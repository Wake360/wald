from wald import narrative
from wald.ingest import Cell, ParsedNotebook
from wald.llm import BackendError


def make_nb(specs):
    """specs: list of (cell_type, source[, outputs_text])."""
    cells = []
    for i, spec in enumerate(specs):
        cell_type, source = spec[0], spec[1]
        outputs = spec[2] if len(spec) > 2 else ""
        cells.append(Cell(index=i, cell_type=cell_type, source=source, outputs_text=outputs))
    return ParsedNotebook(path=None, cells=cells)


class StubBackend:
    """Protocol-conforming; returns one canned dict, no network."""

    provider = "stub-detector"
    model = "stub-model"
    kind = "api"

    def __init__(self, response):
        self._response = response
        self.calls = 0

    @property
    def gate_eligible(self):
        return True

    def complete(self, system, user, schema=None):
        self.calls += 1
        return self._response


class ErrorBackend:
    provider = "stub-detector"
    model = "stub-model"
    kind = "api"

    @property
    def gate_eligible(self):
        return True

    def complete(self, system, user, schema=None):
        raise BackendError("boom")


CODE_CELL = "df = load()\nactive = df[df.status == 'active']\navg = active.spend.mean()"
CLAIM_MD = "Users spend more, so we should invest in retention."


def _happy_nb():
    return make_nb([
        ("code", "import pandas as pd"),
        ("markdown", CLAIM_MD),
        ("code", CODE_CELL),
    ])


def test_happy_path_produces_claim_and_finding():
    backend = StubBackend({
        "claims": [
            {"cell": 1, "quote": "we should invest", "kind": "population", "code_cells": [2]}
        ],
        "findings": [
            {
                "flaw_id": "selection-survivorship-cohort",
                "claim_span": {"cell": 1, "quote": "we should invest"},
                "code_span": {"cell": 2, "quote": "active = df[df.status == 'active']"},
                "failure_scenario": "survivor-only metric",
                "fix": "compute on full cohort",
                "model_confidence": 0.7,
            }
        ],
    })
    result = narrative.detect_narrative(_happy_nb(), backend)
    assert backend.calls == 1
    assert result.dropped == []
    assert len(result.claims) == 1
    assert result.claims[0].kind == "population"
    assert result.claims[0].code_cells == [2]
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.flaw_id == "selection-survivorship-cohort"
    assert f.code_cell == 2
    assert (f.code_line_start, f.code_line_end) == (2, 2)
    assert f.model_confidence == 0.7


def test_paraphrased_claim_quote_dropped():
    backend = StubBackend({
        "claims": [
            {"cell": 1, "quote": "the analysts recommend more spending", "kind": "population", "code_cells": []}
        ],
        "findings": [],
    })
    result = narrative.detect_narrative(_happy_nb(), backend)
    assert result.claims == []
    assert len(result.dropped) == 1
    assert "not verbatim in cell 1" in result.dropped[0]


def test_wrong_cell_index_dropped():
    backend = StubBackend({
        "claims": [],
        "findings": [
            {
                "flaw_id": "selection-survivorship-cohort",
                "claim_span": {"cell": 1, "quote": "we should invest"},
                "code_span": {"cell": 1, "quote": "we should invest"},
                "model_confidence": 0.5,
            }
        ],
    })
    result = narrative.detect_narrative(_happy_nb(), backend)
    assert result.findings == []
    assert any("code cell 1 is not a code cell" in d for d in result.dropped)


def test_non_enabled_flaw_id_dropped():
    backend = StubBackend({
        "claims": [],
        "findings": [
            {
                "flaw_id": "leakage-future-feature",
                "claim_span": {"cell": 1, "quote": "we should invest"},
                "code_span": {"cell": 2, "quote": "active = df[df.status == 'active']"},
                "model_confidence": 0.9,
            }
        ],
    })
    result = narrative.detect_narrative(_happy_nb(), backend)
    assert result.findings == []
    assert any("not narrative-enabled" in d for d in result.dropped)


def test_oversized_code_quote_dropped():
    backend = StubBackend({
        "claims": [],
        "findings": [
            {
                "flaw_id": "selection-survivorship-cohort",
                "claim_span": {"cell": 1, "quote": "we should invest"},
                "code_span": {"cell": 2, "quote": "x" * 401},
                "model_confidence": 0.6,
            }
        ],
    })
    result = narrative.detect_narrative(_happy_nb(), backend)
    assert result.findings == []
    assert any("exceeds 400 chars" in d for d in result.dropped)


def test_backend_error_returns_empty_with_dropped_entry():
    result = narrative.detect_narrative(_happy_nb(), ErrorBackend())
    assert result.claims == []
    assert result.findings == []
    assert len(result.dropped) == 1
    assert "backend error" in result.dropped[0]


def test_confidence_clamped():
    backend = StubBackend({
        "claims": [],
        "findings": [
            {
                "flaw_id": "selection-survivorship-cohort",
                "claim_span": {"cell": 1, "quote": "we should invest"},
                "code_span": {"cell": 2, "quote": "active = df[df.status == 'active']"},
                "model_confidence": 5.0,
            }
        ],
    })
    result = narrative.detect_narrative(_happy_nb(), backend)
    assert result.findings[0].model_confidence == 1.0


def test_unknown_kind_coerced_to_descriptive():
    backend = StubBackend({
        "claims": [
            {"cell": 1, "quote": "we should invest", "kind": "made-up-kind", "code_cells": []}
        ],
        "findings": [],
    })
    result = narrative.detect_narrative(_happy_nb(), backend)
    assert result.claims[0].kind == "descriptive"


def test_system_prompt_snapshot():
    prompt = narrative.build_system_prompt()
    for flaw_id in (
        "selection-survivorship-cohort",
        "significance-meaningless",
        "regression-to-mean-claim",
    ):
        assert prompt.count(flaw_id) == 1
    assert "explicitly scoped to the filtered group" in prompt
    assert "linearity-extrapolation" not in prompt
    assert "leakage-future-feature" not in prompt


def test_package_notebook_markdown_never_truncated_when_code_is():
    md = "CONCLUSION: revenue rose because the active cohort spent more. " * 4
    big_code = "value = compute_something_really_long()  # padding\n" * 200
    nb = make_nb([("markdown", md), ("code", big_code)])
    packaged = narrative.package_notebook(nb, max_chars=1000)
    assert md in packaged  # markdown survives whole
    assert "truncated" in packaged  # code was shortened
    assert "### Cell 0 (markdown)" in packaged
    assert "### Cell 1 (code)" in packaged
    assert len(packaged) < len("\n".join([md, big_code]))


def test_package_notebook_deterministic_and_true_indices():
    nb = make_nb([
        ("code", "import numpy as np"),
        ("markdown", "Intro text."),
        ("code", "x = 1\ny = 2"),
    ])
    first = narrative.package_notebook(nb)
    second = narrative.package_notebook(nb)
    assert first == second
    assert "### Cell 0 (code)" in first
    assert "### Cell 1 (markdown)" in first
    assert "### Cell 2 (code)" in first
    assert "x = 1" in first  # code rendered as raw source, no line-number gutter
    assert "y = 2" in first
    assert "1| x = 1" not in first  # gutter removed so quotes can match raw source


def test_rendered_code_quote_grounds_against_raw_source():
    """Finding 1: with the gutter removed, a verbatim quote of the rendered code
    is a substring of the raw cell source, so grounding succeeds."""
    nb = make_nb([("markdown", CLAIM_MD), ("code", CODE_CELL)])
    packaged = narrative.package_notebook(nb)
    quoted = "active = df[df.status == 'active']"
    assert quoted in packaged  # rendered verbatim, no "N| " prefix to copy
    backend = StubBackend({
        "claims": [],
        "findings": [
            {
                "flaw_id": "selection-survivorship-cohort",
                "claim_span": {"cell": 0, "quote": CLAIM_MD},
                "code_span": {"cell": 1, "quote": quoted},
                "model_confidence": 0.5,
            }
        ],
    })
    result = narrative.detect_narrative(nb, backend)
    assert result.dropped == []
    assert len(result.findings) == 1
    assert (result.findings[0].code_line_start, result.findings[0].code_line_end) == (2, 2)


def test_line_span_multiline_quote_first_line_long():
    """Finding 2: a multi-line quote whose first line is longer than
    len(q_norm)+200 must still match via the normalized fallback."""
    long_first = "avg = data[" + "col " * 80 + "].mean()"  # >300 chars
    source = long_first + "\n\nresult = avg * 2"
    quote = "].mean() result = avg * 2"  # spans long line into a later line
    assert quote not in source  # direct find fails → normalized fallback path
    assert len(narrative._norm(long_first)) > len(narrative._norm(quote)) + 200
    span = narrative._line_span(quote, source)
    assert span == (1, 3)


def _short_claim_backend(quote):
    return StubBackend({
        "claims": [{"cell": 1, "quote": quote, "kind": "population", "code_cells": []}],
        "findings": [],
    })


def test_empty_claim_quote_dropped():
    result = narrative.detect_narrative(_happy_nb(), _short_claim_backend(""))
    assert result.claims == []
    assert any("quote too short" in d for d in result.dropped)


def test_whitespace_only_claim_quote_dropped():
    result = narrative.detect_narrative(_happy_nb(), _short_claim_backend("   \n  \t "))
    assert result.claims == []
    assert any("quote too short" in d for d in result.dropped)


def test_just_under_threshold_claim_quote_dropped():
    quote = "we should inve"  # 14 chars, a real substring of CLAIM_MD
    assert len(quote) == 14 and quote in CLAIM_MD
    result = narrative.detect_narrative(_happy_nb(), _short_claim_backend(quote))
    assert result.claims == []
    assert any("quote too short" in d for d in result.dropped)


def test_short_code_quote_dropped():
    backend = StubBackend({
        "claims": [],
        "findings": [
            {
                "flaw_id": "selection-survivorship-cohort",
                "claim_span": {"cell": 1, "quote": "we should invest"},
                "code_span": {"cell": 2, "quote": "avg"},  # 3 chars, < 10
                "model_confidence": 0.5,
            }
        ],
    })
    result = narrative.detect_narrative(_happy_nb(), backend)
    assert result.findings == []
    assert any("code quote too short" in d for d in result.dropped)


def test_empty_code_quote_dropped():
    backend = StubBackend({
        "claims": [],
        "findings": [
            {
                "flaw_id": "selection-survivorship-cohort",
                "claim_span": {"cell": 1, "quote": "we should invest"},
                "code_span": {"cell": 2, "quote": "   "},  # whitespace-only, < 10
                "model_confidence": 0.5,
            }
        ],
    })
    result = narrative.detect_narrative(_happy_nb(), backend)
    assert result.findings == []
    assert any("code quote too short" in d for d in result.dropped)


def test_curly_quote_markdown_grounds():
    """Finding 4: typographic variants in markdown fold to their ASCII forms
    on both sides of the match."""
    md = "The team’s “win rate” rose — we should invest heavily."
    nb = make_nb([("code", "import pandas as pd"), ("markdown", md), ("code", CODE_CELL)])
    quote = "The team's \"win rate\" rose - we should invest heavily."
    assert quote not in md  # only the folded normalized forms coincide
    backend = StubBackend({
        "claims": [{"cell": 1, "quote": quote, "kind": "descriptive", "code_cells": []}],
        "findings": [],
    })
    result = narrative.detect_narrative(nb, backend)
    assert result.dropped == []
    assert len(result.claims) == 1


def test_duplicate_code_quote_resolves_to_first_occurrence():
    """Finding 5: a code quote appearing twice pins to the first occurrence,
    deterministically."""
    dup = "value = compute()"  # >= 10 chars, appears on lines 1 and 3
    code = "value = compute()\ny = 2\nvalue = compute()"
    nb = make_nb([("markdown", CLAIM_MD), ("code", code)])
    backend = StubBackend({
        "claims": [],
        "findings": [
            {
                "flaw_id": "selection-survivorship-cohort",
                "claim_span": {"cell": 0, "quote": CLAIM_MD},
                "code_span": {"cell": 1, "quote": dup},
                "model_confidence": 0.5,
            }
        ],
    })
    result = narrative.detect_narrative(nb, backend)
    assert len(result.findings) == 1
    f = result.findings[0]
    assert (f.code_line_start, f.code_line_end) == (1, 1)
