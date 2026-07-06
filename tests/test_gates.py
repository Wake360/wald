"""Golden gates G0 + G1 (need a built corpus: `wald corpus build`)."""

import json
import re
from pathlib import Path

import nbformat
import pytest

from wald.corpus import MUTATION_SEEDS
from wald.detect import run_static
from wald.eval import REAL_FP_CAVEAT, evaluate, render_report
from wald.ingest import parse_notebook

CORPUS = Path(__file__).parent.parent / "corpus"

pytestmark = pytest.mark.skipif(
    not ((CORPUS / "MANIFEST.json").exists()
         and (CORPUS / "negative" / "MANIFEST.json").exists()),
    reason="corpus not built (run: wald corpus build)",
)


def manifest():
    return json.loads((CORPUS / "MANIFEST.json").read_text())


def negative_manifest():
    return json.loads((CORPUS / "negative" / "MANIFEST.json").read_text())


def cell_source(file: str, index: int) -> str:
    nb = nbformat.read(str(CORPUS / file), as_version=4)
    return nb.cells[index]["source"]


def test_g0_corpus_size_and_verification():
    m = manifest()
    assert len(m["clean"]) >= 40
    assert len(m["mutants"]) >= 150
    assert all(e["verified"] for e in m["mutants"])
    for e in m["mutants"]:
        assert (CORPUS / e["file"]).exists()
    assert "provenance" in m


def test_g0_every_flaw_class_with_mutation_is_represented():
    labels = {e["flaw_id"] for e in manifest()["mutants"]}
    assert {
        "leakage-fit-before-split",
        "testing-multiple-uncorrected",
        "baserate-accuracy-imbalanced",
        "selection-survivorship-cohort",
        "significance-meaningless",
        "regression-to-mean-claim",
        "leakage-temporal-shuffle",
    } <= labels


def test_g0_split_on_every_entry_and_no_base_in_both_splits():
    m = manifest()
    split_of_base = {}
    for e in m["clean"] + m["mutants"]:
        assert e["split"] in {"dev", "heldout"}, e["file"]
        base = e.get("base", e["file"])
        split_of_base.setdefault(base, set()).add(e["split"])
    crossed = {b for b, s in split_of_base.items() if len(s) > 1}
    assert not crossed, crossed


def test_g0_conclusion_phrasings_disjoint_across_splits():
    m = manifest()
    dev = {e["conclusion"] for e in m["mutants"]
           if "conclusion" in e and e["split"] == "dev"}
    heldout = {e["conclusion"] for e in m["mutants"]
               if "conclusion" in e and e["split"] == "heldout"}
    assert dev and heldout
    assert not dev & heldout


def test_g0_mutation_seeds_disjoint_across_splits():
    # third of the three plan-mandated disjointnesses (base files, phrasing
    # strings, mutation seeds); read from the built MANIFEST, not the constant
    seeds_by = {}
    for e in manifest()["mutants"]:
        by = seeds_by.setdefault(e["flaw_id"], {"dev": set(), "heldout": set()})
        by[e["split"]].add(e["mutation_seed"])
    for flaw_id in MUTATION_SEEDS:
        dev = seeds_by.get(flaw_id, {}).get("dev", set())
        heldout = seeds_by.get(flaw_id, {}).get("heldout", set())
        assert dev and heldout, flaw_id
        assert not dev & heldout, (flaw_id, dev & heldout)


def test_g0_negative_recipes_size_and_split():
    by_recipe = {}
    for f in negative_manifest()["flags"]:
        by_recipe.setdefault(f["recipe"], []).append(f)
    assert set(by_recipe) == {
        "scoped-claim", "effect-size-present", "control-group-present",
        "wrong-code-span", "legit-cv-generalization",
    }
    for recipe, fs in by_recipe.items():
        splits = {f["split"] for f in fs}
        if recipe == "legit-cv-generalization":
            # reserved: never tuned on — cites only the 4 held-out program nbs
            assert len(fs) >= 4, recipe
            assert splits == {"heldout"}
        else:
            assert len(fs) >= 6, recipe
            assert splits == {"dev", "heldout"}, recipe


def test_g0_legit_cv_flags_cite_only_heldout_notebooks():
    # regression: the reserved held-out recipe must never cite a dev notebook,
    # or a prompt tuned on that dev file would pass a "held-out" G3 check
    split_of = {e["file"]: e["split"] for e in manifest()["clean"]}
    legit = [f for f in negative_manifest()["flags"]
             if f["recipe"] == "legit-cv-generalization"]
    assert legit
    for f in legit:
        assert f["split"] == "heldout", f
        assert split_of.get(f["source_file"]) == "heldout", f


def test_g0_significance_mutant_outputs_reflect_mutated_code():
    # regression: mutants are re-executed before writing, so the ttest cell's
    # stored p-value is from the inflated n=40000 run (p<0.05), not a stale
    # clean-run p that would contradict the significance-meaningless label
    sig = [e for e in manifest()["mutants"]
           if e["flaw_id"] == "significance-meaningless"]
    assert sig
    for e in sig:
        nb = nbformat.read(str(CORPUS / e["file"]), as_version=4)
        code = [c for c in nb.cells if c["cell_type"] == "code"]
        assert any("n = 40000" in c["source"] for c in code), e["file"]
        out_text = "\n".join(
            o.get("text", "") for c in code
            for o in c.get("outputs", []) if o.get("output_type") == "stream"
        )
        pm = re.search(r"p=([0-9.]+)", out_text)
        assert pm, (e["file"], out_text)
        assert float(pm.group(1)) < 0.05, (e["file"], pm.group(1))


def test_g1_narrative_only_mutants_are_accounted():
    # regression: narrative-layer-only mutants are out of static-recall scope
    # but must be counted, not silently dropped from the headline total
    results = evaluate(CORPUS)
    static_measured = sum(r["tp"] + r["fn"] for r in results["static_classes"].values())
    cand = results["candidate_classes"]["selection-survivorship-cohort"]
    candidate_measured = cand["tp"] + cand["fn"]
    total = static_measured + candidate_measured + results["n_mutants_narrative_only"]
    assert total == results["n_mutants"], (
        static_measured, candidate_measured,
        results["n_mutants_narrative_only"], results["n_mutants"])
    assert set(results["narrative_only_classes"]) == {
        "regression-to-mean-claim", "significance-meaningless"}
    assert "Narrative-layer-only mutants" in render_report(results)


def test_g0_negative_quotes_are_verbatim_spans():
    for f in negative_manifest()["flags"]:
        for span in (f["claim_span"], f["code_span"]):
            assert span["quote"] in cell_source(f["source_file"], span["cell"]), f


def test_g0_negative_falseness_is_mechanical():
    for f in negative_manifest()["flags"]:
        claim_cell = cell_source(f["source_file"], f["claim_span"]["cell"])
        code_cell = cell_source(f["source_file"], f["code_span"]["cell"])
        if f["recipe"] == "scoped-claim":
            assert re.search(r"active|retained|surviv|complet", claim_cell, re.IGNORECASE), f
        elif f["recipe"] == "effect-size-present":
            assert "effect size" in claim_cell, f
        elif f["recipe"] == "control-group-present":
            assert "control" in code_cell, f
        elif f["recipe"] == "wrong-code-span":
            assert "import" in code_cell and "status" not in code_cell, f
        elif f["recipe"] == "legit-cv-generalization":
            assert "Pipeline(" in code_cell and "cross_val_score" in code_cell, f


def test_g1_static_precision_recall_and_clean_fp():
    results = evaluate(CORPUS)
    for cls, r in results["static_classes"].items():
        assert r["precision"] is None or r["precision"] >= 0.9, (cls, r)
        # recall is unmeasurable (None) for a registered static class that has
        # no corpus mutants yet; a measured recall must still clear the bar
        assert r["recall"] is None or r["recall"] >= 0.7, (cls, r)
    assert results["clean_fp_rate"] <= 0.05, results["clean_fp_files"]


def test_g1_temporal_zero_flags_on_real():
    for f in sorted((CORPUS / "real").glob("*.ipynb")):
        flags = run_static(parse_notebook(f))
        assert not any(fl.flaw_id == "leakage-temporal-shuffle" for fl in flags), f


def test_g1_report_date_pinned_by_env(monkeypatch):
    monkeypatch.setenv("WALD_BUILD_DATE", "1999-12-31")
    results = evaluate(CORPUS)
    assert results["date"] == "1999-12-31"


def test_g1_real_fp_caveat_appears_when_real_notebooks_present():
    results = evaluate(CORPUS)
    assert results["n_clean_real"] > 0
    assert REAL_FP_CAVEAT in render_report(results)


def test_g1_survivorship_candidate_recall():
    # the static candidate now sees both filter idioms (subscript and
    # .query), so the gate holds it accountable for every survivorship
    # mutant: full candidate recall, promotion to a flag stays the
    # narrative layer's job
    hits = total = 0
    for e in manifest()["mutants"]:
        if e["flaw_id"] != "selection-survivorship-cohort":
            continue
        total += 1
        flags = run_static(parse_notebook(CORPUS / e["file"]))
        hits += any(f.flaw_id == "selection-survivorship-cohort" for f in flags)
    assert total >= 16
    assert hits == total, (hits, total)
