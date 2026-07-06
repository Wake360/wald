"""End-to-end narrative eval over a 3-notebook mini-corpus (real corpus files)
with stub backends: exact confusion arithmetic, the heldout structural raise,
gate_evidence, per-recipe G3 kill arithmetic."""

import json
import shutil
from pathlib import Path

import nbformat
import pytest

from wald.cli import _heldout_refusal
from wald.detect import run_static
from wald.eval import evaluate_narrative
from wald.ingest import parse_notebook
from wald.llm import PINNED_DETECTOR_MODEL, PINNED_VERIFIER_MODEL, ReplayBackend
from wald.mutate import SurvivorshipMutation

CORPUS = Path(__file__).parent.parent / "corpus"

pytestmark = pytest.mark.skipif(
    not (CORPUS / "MANIFEST.json").exists(),
    reason="corpus not built (run: wald corpus build)",
)

SCOPED_QUOTE = "These results describe customers in this dataset only."


class DetStub:
    """Routes on a marker substring of the packaged notebook text."""

    provider = "stub-detector"
    model = "stub"
    kind = "stub"
    gate_eligible = False

    def __init__(self, routes=()):
        self.routes = list(routes)
        self.calls = 0

    def complete(self, system, user, schema=None):
        self.calls += 1
        for marker, response in self.routes:
            if marker in user:
                return response
        return {"claims": [], "findings": []}


class VerStub:
    """Supports a finding iff a known-true marker appears in the prompt
    (mutant conclusions are true; authored negatives are not)."""

    provider = "stub-verifier"
    model = "stub"
    kind = "stub"
    gate_eligible = False

    def __init__(self, support_markers=()):
        self.support_markers = tuple(support_markers)
        self.calls = 0

    def complete(self, system, user, schema=None):
        self.calls += 1
        if any(m in user for m in self.support_markers):
            return {"verdict": "supported", "reason": "stub: true flag"}
        return {"verdict": "unsupported", "reason": "stub: seeded false flag"}


def _finding(flaw_id, claim_cell, claim_quote, code_cell, code_quote):
    return {
        "flaw_id": flaw_id,
        "claim_span": {"cell": claim_cell, "quote": claim_quote},
        "code_span": {"cell": code_cell, "quote": code_quote},
        "failure_scenario": "fs",
        "fix": "fix",
        "model_confidence": 0.7,
    }


def _dev_mutant(manifest, flaw_id, require=None):
    for e in manifest["mutants"]:
        if e["flaw_id"] != flaw_id or e["split"] != "dev":
            continue
        if require is not None:
            nb = nbformat.read(str(CORPUS / e["file"]), as_version=4)
            src = "\n".join(c["source"] for c in nb.cells if c["cell_type"] == "code")
            if require not in src:
                continue
        return e
    raise AssertionError(f"no dev mutant for {flaw_id}")


def _claim_cell(file, conclusion):
    nb = nbformat.read(str(CORPUS / file), as_version=4)
    return next(
        i for i, c in enumerate(nb.cells)
        if c["cell_type"] == "markdown" and conclusion in c["source"]
    )


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    root = tmp_path_factory.mktemp("mini-corpus")
    manifest = json.loads((CORPUS / "MANIFEST.json").read_text())
    clean = next(e for e in manifest["clean"]
                 if e["split"] == "dev" and e["family"] == "churn")
    subscript = SurvivorshipMutation.FILTERS[0]
    surv = _dev_mutant(manifest, "selection-survivorship-cohort", require=subscript)
    rtm = _dev_mutant(manifest, "regression-to-mean-claim")
    for e in (clean, surv, rtm):
        dst = root / e["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(CORPUS / e["file"], dst)
    (root / "MANIFEST.json").write_text(json.dumps({
        "built": manifest["built"], "clean": [clean],
        "mutants": [surv, rtm], "discarded": [],
    }))

    # survivorship: cite the static candidate's cell so the fused rule fires
    cand = next(f for f in run_static(parse_notebook(root / surv["file"]))
                if f.flaw_id == "selection-survivorship-cohort")
    surv_finding = _finding(
        "selection-survivorship-cohort",
        _claim_cell(surv["file"], surv["conclusion"]), surv["conclusion"],
        cand.cell, subscript,
    )
    # RTM: no static candidate exists, the finding fires solo at 0.80
    rtm_nb = nbformat.read(str(CORPUS / rtm["file"]), as_version=4)
    sel_cell = next(i for i, c in enumerate(rtm_nb.cells)
                    if c["cell_type"] == "code" and "flagged = " in c["source"])
    sel_line = rtm_nb.cells[sel_cell]["source"].split("\n")[0]
    rtm_finding = _finding(
        "regression-to-mean-claim",
        _claim_cell(rtm["file"], rtm["conclusion"]), rtm["conclusion"],
        sel_cell, sel_line,
    )
    det_routes = [
        (surv["conclusion"], {"claims": [], "findings": [surv_finding]}),
        (rtm["conclusion"], {"claims": [], "findings": [rtm_finding]}),
    ]

    # stub negatives: scoped-claim recipe half-killable by construction (one
    # entry cites the clean notebook's scoped conclusion cell, which the stub
    # verifier supports), the other recipe fully killed; one heldout entry
    # must be excluded from the dev run
    def neg(flaw_id, recipe, split, claim_cell, claim_quote):
        return {
            "flaw_id": flaw_id, "recipe": recipe, "split": split,
            "source_file": clean["file"],
            "claim_span": {"cell": claim_cell, "quote": claim_quote},
            "code_span": {"cell": 4, "quote": "X = df[num_cols]"},
        }

    neg_dir = root / "negative"
    neg_dir.mkdir()
    (neg_dir / "MANIFEST.json").write_text(json.dumps({"flags": [
        neg("selection-survivorship-cohort", "scoped-claim", "dev",
            0, "# Churn model — activity features"),
        neg("selection-survivorship-cohort", "scoped-claim", "dev",
            7, SCOPED_QUOTE),
        neg("significance-meaningless", "effect-size-present", "dev",
            0, "# Churn model — activity features"),
        neg("significance-meaningless", "effect-size-present", "heldout",
            7, SCOPED_QUOTE),
    ]}))

    support = (surv["conclusion"], rtm["conclusion"], SCOPED_QUOTE)
    return {"root": root, "det_routes": det_routes, "support": support}


@pytest.fixture(scope="module")
def results(mini):
    det = DetStub(mini["det_routes"])
    ver = VerStub(mini["support"])
    return evaluate_narrative(mini["root"], det, ver, split="dev")


def test_confusion_arithmetic(results):
    surv = results["narrative_classes"]["selection-survivorship-cohort"]
    rtm = results["narrative_classes"]["regression-to-mean-claim"]
    sig = results["narrative_classes"]["significance-meaningless"]
    assert (surv["tp"], surv["fn"], surv["fp"]) == (1, 0, 0)
    assert (surv["precision"], surv["recall"], surv["f1"]) == (1.0, 1.0, 1.0)
    assert (rtm["tp"], rtm["fn"], rtm["fp"]) == (1, 0, 0)
    assert (rtm["precision"], rtm["recall"], rtm["f1"]) == (1.0, 1.0, 1.0)
    # no significance mutant in the mini-manifest and no spurious flag
    assert (sig["tp"], sig["fn"], sig["fp"]) == (0, 0, 0)
    assert sig["precision"] is None and sig["recall"] is None and sig["f1"] is None
    assert results["missed_mutants"] == []


def test_clean_fp_and_dropped(results):
    assert results["n_clean"] == 1 and results["n_clean_real"] == 0
    assert results["clean_fp_rate"] == 0.0
    assert results["clean_fp_files"] == []
    assert results["dropped_ungrounded"] == {
        "dropped": 0, "raw_findings": 2, "rate": 0.0,
    }


def test_g3_per_recipe_kill_arithmetic(results):
    assert results["g3_per_recipe"] == {
        "scoped-claim": {"killed": 1, "total": 2, "kill_rate": 0.5},
        "effect-size-present": {"killed": 1, "total": 1, "kill_rate": 1.0},
    }


def test_true_flag_survival(results):
    assert results["true_flag_survival"] == {"supported": 2, "total": 2, "rate": 1.0}


def test_gate_evidence_false_for_stub_run(results):
    assert results["gate_evidence"] is False
    assert results["split"] == "dev"
    assert results["usage"] == {"detector": None, "verifier": None}


def test_evaluate_narrative_continues_past_backend_error(mini):
    from wald.llm import BackendError

    rtm_marker = mini["det_routes"][1][0]  # the RTM mutant's conclusion text

    class FlakyDet(DetStub):
        def complete(self, system, user, schema=None):
            if rtm_marker in user:
                raise BackendError("simulated 503")
            return super().complete(system, user, schema)

    det = FlakyDet(mini["det_routes"])
    ver = VerStub(mini["support"])
    res = evaluate_narrative(mini["root"], det, ver, split="dev")

    manifest = json.loads((mini["root"] / "MANIFEST.json").read_text())
    rtm_file = next(e["file"] for e in manifest["mutants"]
                    if e["flaw_id"] == "regression-to-mean-claim")

    # the errored file lands in backend_errors and contributes to no bucket
    assert len(res["backend_errors"]) == 1
    assert res["backend_errors"][0]["file"] == rtm_file
    assert "simulated 503" in res["backend_errors"][0]["error"]
    rtm = res["narrative_classes"]["regression-to-mean-claim"]
    assert (rtm["tp"], rtm["fn"], rtm["fp"]) == (0, 0, 0)
    assert rtm_file not in res["missed_mutants"]
    # the other mutant is still scored, and any backend error voids the gate
    surv = res["narrative_classes"]["selection-survivorship-cohort"]
    assert (surv["tp"], surv["fn"], surv["fp"]) == (1, 0, 0)
    assert res["gate_evidence"] is False


def test_eval_progress_lines(mini, monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    det = DetStub(mini["det_routes"])
    ver = VerStub(mini["support"])
    evaluate_narrative(mini["root"], det, ver, split="dev")
    err = capsys.readouterr().err
    # 2 mutants + 1 clean in the mini corpus: all three ticks appear
    assert "checking 1/3" in err
    assert "checking 3/3" in err


def test_eval_progress_absent_when_piped(mini, capsys):
    det = DetStub(mini["det_routes"])
    ver = VerStub(mini["support"])
    evaluate_narrative(mini["root"], det, ver, split="dev")
    assert "checking" not in capsys.readouterr().err


def test_heldout_raises_before_any_call(mini):
    det = DetStub(mini["det_routes"])
    ver = VerStub(mini["support"])
    with pytest.raises(RuntimeError, match="heldout"):
        evaluate_narrative(mini["root"], det, ver, split="heldout")
    assert det.calls == 0 and ver.calls == 0


def test_heldout_raises_for_prepopulated_replay_cache(mini, tmp_path):
    # a ReplayBackend over a cache dir that already has every answer on disk
    # has served_from_disk == 0 (nothing served *yet*) and gate_eligible ==
    # True at this pre-run check, but would go on to serve the whole run from
    # disk — the guard must key off the static `kind`, not `gate_eligible`.
    class FakeApi:
        provider = "anthropic"
        model = PINNED_DETECTOR_MODEL
        kind = "api"
        gate_eligible = True

        def complete(self, system, user, schema=None):
            raise AssertionError("must not be called: guard should raise first")

    det = ReplayBackend(tmp_path / "detector", FakeApi())
    ver = ReplayBackend(tmp_path / "verifier", FakeApi())
    assert det.gate_eligible and ver.gate_eligible  # true today, the trap
    with pytest.raises(RuntimeError, match="heldout"):
        evaluate_narrative(mini["root"], det, ver, split="heldout")


def test_gate_evidence_false_when_model_not_pinned(mini):
    class ApiStub:
        kind = "api"
        gate_eligible = True

        def __init__(self, provider, model):
            self.provider, self.model = provider, model

        def complete(self, system, user, schema=None):
            return {"claims": [], "findings": []}

    det = ApiStub("anthropic", "not-the-pinned-model")
    ver = ApiStub("openai", PINNED_VERIFIER_MODEL)
    results = evaluate_narrative(mini["root"], det, ver, split="dev")
    assert results["gate_evidence"] is False


def test_heldout_refusal_blocks_non_api_backend(tmp_path):
    (tmp_path / "clean").mkdir()
    (tmp_path / "clean" / "foo.ipynb").write_text("{}")
    (tmp_path / "MANIFEST.json").write_text(json.dumps({
        "clean": [{"file": "clean/foo.ipynb", "split": "heldout"}], "mutants": [],
    }))
    det = ReplayBackend(tmp_path / "d")
    ver = ReplayBackend(tmp_path / "v")
    msg = _heldout_refusal(tmp_path / "clean" / "foo.ipynb", det, ver)
    assert msg is not None and "held-out" in msg


def test_heldout_refusal_allows_api_backend_and_dev_split(tmp_path):
    class ApiStub:
        kind = "api"

    (tmp_path / "clean").mkdir()
    (tmp_path / "clean" / "foo.ipynb").write_text("{}")
    (tmp_path / "MANIFEST.json").write_text(json.dumps({
        "clean": [{"file": "clean/foo.ipynb", "split": "heldout"}], "mutants": [],
    }))
    assert _heldout_refusal(tmp_path / "clean" / "foo.ipynb", ApiStub(), ApiStub()) is None

    (tmp_path / "MANIFEST.json").write_text(json.dumps({
        "clean": [{"file": "clean/foo.ipynb", "split": "dev"}], "mutants": [],
    }))
    det = ReplayBackend(tmp_path / "d2")
    ver = ReplayBackend(tmp_path / "v2")
    assert _heldout_refusal(tmp_path / "clean" / "foo.ipynb", det, ver) is None


def test_heldout_refusal_blocks_real_corpus_notebook(tmp_path):
    """Finding cli.py:25: corpus/real/* is held-out gate-only material even
    though its manifest carries no split field and prefixes paths with real/."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "nb.ipynb").write_text("{}")
    # mirrors corpus/real/MANIFEST.json: "real/"-prefixed paths, no split key
    (real / "MANIFEST.json").write_text(json.dumps({
        "clean": [{"file": "real/nb.ipynb", "repo": "x/y"}], "mutants": [],
    }))
    det = ReplayBackend(tmp_path / "d3")
    ver = ReplayBackend(tmp_path / "v3")
    msg = _heldout_refusal(real / "nb.ipynb", det, ver)
    assert msg is not None and "held-out" in msg
