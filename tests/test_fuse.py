import pytest

from wald.detect import Flag
from wald.fuse import fuse, run_full
from wald.ingest import Cell, ParsedNotebook
from wald.narrative import Claim, NarrativeFinding, NarrativeResult
from wald.taxonomy import load_fusion_rules

RULES = load_fusion_rules()


def survivorship_candidate(cell=2, confidence=0.55) -> Flag:
    return Flag(
        flaw_id="selection-survivorship-cohort",
        severity="high",
        confidence=confidence,
        cell=cell,
        line=1,
        evidence="cohort filtered on `status`",
        failure_scenario="",
        fix="",
        extra={"column": "status"},
    )


def leakage_candidate(cell=3, sink_cell=5, confidence=0.75) -> Flag:
    extra = {"leaked_names": ["X"]}
    if sink_cell is not None:
        extra["sink_cell"] = sink_cell
    return Flag(
        flaw_id="leakage-fit-before-split",
        severity="high",
        confidence=confidence,
        cell=cell,
        line=1,
        evidence="scaler fitted before cross_val_score",
        failure_scenario="",
        fix="",
        extra=extra,
    )


def survivorship_finding(code_cell=2, claim_cell=0, quote="q") -> NarrativeFinding:
    return NarrativeFinding(
        flaw_id="selection-survivorship-cohort",
        claim_cell=claim_cell,
        claim_quote=quote,
        code_cell=code_cell,
        code_line_start=1,
        code_line_end=1,
        code_quote="df = df[df.status == 'active']",
        failure_scenario="survivor-only metric",
        fix="compute on full cohort",
        model_confidence=0.7,
    )


def result(claims=(), findings=()) -> NarrativeResult:
    return NarrativeResult(claims=list(claims), findings=list(findings), dropped=[])


class StubBackend:
    def __init__(self, provider, response):
        self.provider = provider
        self.model = "stub"
        self.kind = "api"
        self.gate_eligible = True
        self._response = response

    def complete(self, system, user, schema=None):
        return self._response


def survivorship_nb() -> ParsedNotebook:
    return ParsedNotebook(
        path=None,
        cells=[
            Cell(index=0, cell_type="markdown", source="LTV grows across the whole population."),
            Cell(index=1, cell_type="code", source="df = load()"),
            Cell(
                index=2,
                cell_type="code",
                source="df = df[df.status == 'active']\nltv = df.ltv.mean()",
            ),
        ],
    )


NARRATIVE_RAW = {
    "claims": [],
    "findings": [
        {
            "flaw_id": "selection-survivorship-cohort",
            "claim_span": {"cell": 0, "quote": "LTV grows across the whole population."},
            "code_span": {"cell": 2, "quote": "df = df[df.status == 'active']"},
            "failure_scenario": "survivor-only metric",
            "fix": "compute on full cohort",
            "model_confidence": 0.7,
        }
    ],
}


# -- fuse (pure) --------------------------------------------------------------


def test_survivorship_fuses_to_091_with_both_spans():
    flags = fuse([survivorship_candidate()], result(findings=[survivorship_finding()]), RULES)
    assert len(flags) == 1
    f = flags[0]
    assert f.flaw_id == "selection-survivorship-cohort"
    assert f.confidence == 0.91
    assert f.extra["claim_span"] == {"cell": 0, "quote": "q"}
    assert f.extra["code_span"]["cell"] == 2


def test_scoped_claim_counter_no_finding_emits_nothing():
    # candidate present but the narrative layer confirmed no population claim
    flags = fuse([survivorship_candidate()], result(findings=[]), RULES)
    assert flags == []


def test_survivorship_needs_same_cell():
    # finding cites a different code cell than the candidate -> no fusion; it
    # falls through to the solo rule at 0.80 rather than promoting to 0.91
    flags = fuse([survivorship_candidate(cell=2)], result(findings=[survivorship_finding(code_cell=9)]), RULES)
    assert [f.confidence for f in flags] == [0.80]


def test_candidate_out_of_band_does_not_fuse():
    flags = fuse([survivorship_candidate(confidence=0.9)], result(findings=[survivorship_finding()]), RULES)
    # 0.9 is above the survivorship band; the finding fires solo instead
    assert [f.confidence for f in flags] == [0.80]


def test_precv_fires_only_when_claim_cites_sink():
    claim = Claim(cell=1, quote="model generalizes", kind="performance-generalization", code_cells=[5])
    flags = fuse([leakage_candidate(sink_cell=5)], result(claims=[claim]), RULES)
    assert len(flags) == 1
    assert flags[0].flaw_id == "leakage-fit-before-split"
    assert flags[0].confidence == 0.88
    assert flags[0].extra["claim_span"]["cell"] == 1


def test_precv_no_fire_when_claim_misses_sink():
    claim = Claim(cell=1, quote="model generalizes", kind="performance-generalization", code_cells=[9])
    flags = fuse([leakage_candidate(sink_cell=5)], result(claims=[claim]), RULES)
    assert flags == []


def test_precv_two_claims_same_sink_dedup_keeps_first():
    # two performance-generalization claims both cite the same sink cell;
    # dedup by (flaw_id, cell) must deterministically keep the claim that
    # comes first in narrative.claims, not whichever fires last.
    first = Claim(cell=1, quote="first claim", kind="performance-generalization", code_cells=[5])
    second = Claim(cell=2, quote="second claim", kind="performance-generalization", code_cells=[5])
    flags = fuse([leakage_candidate(sink_cell=5)], result(claims=[first, second]), RULES)
    assert len(flags) == 1
    assert flags[0].extra["claim_span"] == {"cell": 1, "quote": "first claim"}


def test_precv_no_fire_without_recorded_sink():
    claim = Claim(cell=1, quote="model generalizes", kind="performance-generalization", code_cells=[5])
    flags = fuse([leakage_candidate(sink_cell=None)], result(claims=[claim]), RULES)
    assert flags == []


def test_solo_finding_emits_080():
    flags = fuse([], result(findings=[survivorship_finding()]), RULES)
    assert len(flags) == 1
    assert flags[0].confidence == 0.80
    assert flags[0].extra["rule"] == "solo-narrative"


def test_fused_finding_not_also_solo():
    flags = fuse([survivorship_candidate()], result(findings=[survivorship_finding()]), RULES)
    # exactly one flag: the fused 0.91, no duplicate 0.80 solo for the same finding
    assert [f.confidence for f in flags] == [0.91]


def test_dedup_collapses_same_key_keeping_fused():
    findings = [survivorship_finding(quote="a"), survivorship_finding(quote="b")]
    flags = fuse([survivorship_candidate()], result(findings=findings), RULES)
    assert len(flags) == 1
    assert flags[0].confidence == 0.91


# -- run_full (pipeline) ------------------------------------------------------


def test_run_full_trace_shape_supported():
    det = StubBackend("anthropic", NARRATIVE_RAW)
    ver = StubBackend("openai", {"verdict": "supported", "reason": "no scoping present"})
    flags = run_full(survivorship_nb(), det, ver)
    assert len(flags) == 1
    assert flags[0].confidence == 0.91
    assert "claim_span" in flags[0].extra and "code_span" in flags[0].extra
    assert flags[0].extra["verdict_reason"] == "no scoping present"


def test_run_full_unsupported_verdict_kills():
    det = StubBackend("anthropic", NARRATIVE_RAW)
    ver = StubBackend("openai", {"verdict": "unsupported", "reason": "claim is scoped"})
    flags = run_full(survivorship_nb(), det, ver)
    assert flags == []


def test_run_full_same_provider_raises():
    det = StubBackend("openai", NARRATIVE_RAW)
    ver = StubBackend("openai", {"verdict": "supported", "reason": "x"})
    with pytest.raises(ValueError):
        run_full(survivorship_nb(), det, ver)
