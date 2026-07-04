"""Layer B narrative detector: one LLM call per notebook, mechanical grounding.

The model reads every cell and returns claims + findings. Nothing it says is
trusted on faith: a finding survives only if both its quotes are verbatim
substrings of the cells they cite (after whitespace normalization), the cited
cells have the right type, the flaw id is narrative-enabled, and the claim kind
is in the closed enum. Line numbers are recomputed from the matched code quote —
the model's line numbers are never read. Wrong cell = drop, no repair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .ingest import ParsedNotebook
from .llm import BackendError
from .taxonomy import load_taxonomy

CLAIM_KINDS = (
    "population",
    "causal",
    "performance-generalization",
    "forecast",
    "significance",
    "descriptive",
)
MAX_CODE_QUOTE = 400
MIN_CLAIM_QUOTE = 15
MIN_CODE_QUOTE = 10
OUTPUT_CAP = 400
MIN_CODE_BLOCK = 200

# Typographic variants folded before matching so curly quotes / dashes in real
# markdown do not defeat verbatim grounding. Applied to both sides of the match.
_TYPO_FOLD = {
    "‘": "'", "’": "'",  # ‘ ’
    "“": '"', "”": '"',  # “ ”
    "–": "-", "—": "-",  # – —
    " ": " ",                 # nbsp
}

NARRATIVE_SCHEMA = {
    "claims": [
        {
            "cell": "int (true cell index)",
            "quote": "str (verbatim from that markdown cell)",
            "kind": "|".join(CLAIM_KINDS),
            "code_cells": ["int (cell indices the claim rests on)"],
        }
    ],
    "findings": [
        {
            "flaw_id": "str (one of the flaw classes above)",
            "claim_span": {"cell": "int", "quote": "str (verbatim markdown)"},
            "code_span": {"cell": "int", "quote": "str (verbatim code, <=400 chars)"},
            "failure_scenario": "str",
            "fix": "str",
            "model_confidence": "float in [0, 1]",
        }
    ],
}


@dataclass
class Claim:
    cell: int
    quote: str
    kind: str
    code_cells: list[int]


@dataclass
class NarrativeFinding:
    flaw_id: str
    claim_cell: int
    claim_quote: str
    code_cell: int
    code_line_start: int
    code_line_end: int
    code_quote: str
    failure_scenario: str
    fix: str
    model_confidence: float


@dataclass
class NarrativeResult:
    claims: list[Claim]
    findings: list[NarrativeFinding]
    dropped: list[str]


PREAMBLE = (
    "You audit a Jupyter notebook for statistical-integrity flaws in its written "
    "conclusions. You are given every cell in document order with its true index. "
    "Markdown holds the analyst's claims; code holds what was actually computed."
)

KIND_RULE = (
    "A claim kind is exactly one of: " + ", ".join(CLAIM_KINDS) + "."
)

CLAIMS_INVENTORY_RULE = (
    "First inventory claims: list every analytical conclusion the notebook states "
    "in markdown, each as a claim with its cell, an exact quote, its kind, and the "
    "code cells it rests on. Then emit a finding only for a claim that matches one "
    "of the flaw classes above."
)

EVIDENCE_RULE = (
    "A flaw exists only when you can supply a verbatim quote for BOTH the written "
    "claim AND the code it misreads. Copy the characters exactly from the cell "
    "text; do not paraphrase, summarize, or fix typos. Code cells are shown as "
    "raw source with no line-number prefixes, so quote the source line verbatim. "
    "A code quote must be at most 400 characters."
)

CLOSED_SET_RULE = (
    "These flaw ids are the only ones that exist. Never invent a flaw id. If a "
    "claim does not clearly match a class, or if a disqualifier applies, emit no "
    "finding for it."
)


def build_system_prompt() -> str:
    lines = [PREAMBLE, "", "FLAW CLASSES (the closed set — no other flaw exists):", ""]
    for d in load_taxonomy().values():
        if not d.narrative_enabled:
            continue
        lines.append(f"## {d.id}")
        lines.append(f"Definition: {d.definition}")
        lines.append(f"Failure scenario: {d.failure_scenario}")
        lines.append("NOT this flaw when:")
        for dq in d.disqualifiers:
            lines.append(f"  - {dq}")
        lines.append("")
    lines += [
        EVIDENCE_RULE,
        "",
        CLOSED_SET_RULE,
        "",
        KIND_RULE,
        "",
        CLAIMS_INVENTORY_RULE,
        "",
        "Output JSON of this shape:",
        json.dumps(NARRATIVE_SCHEMA, indent=2),
    ]
    return "\n".join(lines)


def _render_code(cell) -> str:
    rendered = cell.source
    if cell.outputs_text:
        capped = cell.outputs_text[:OUTPUT_CAP]
        if len(cell.outputs_text) > OUTPUT_CAP:
            capped += " …[output truncated]"
        rendered += f"\n[output]\n{capped}"
    return rendered


def _truncate(text: str, target: int) -> str:
    marker = f"\n… [{len(text) - target} chars truncated] …\n"
    keep = max(0, target - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def package_notebook(nb: ParsedNotebook, max_chars: int = 48_000) -> str:
    blocks = []
    for c in nb.cells:
        header = f"### Cell {c.index} ({c.cell_type})"
        body = c.source if c.cell_type == "markdown" else _render_code(c)
        blocks.append((c.cell_type, f"{header}\n{body}"))

    total = sum(len(b) for _, b in blocks) + len(blocks)  # + newline joins
    if total > max_chars:
        excess = total - max_chars
        code_order = sorted(
            (i for i, (t, _) in enumerate(blocks) if t == "code"),
            key=lambda i: len(blocks[i][1]),
            reverse=True,
        )
        for i in code_order:
            if excess <= 0:
                break
            block = blocks[i][1]
            cut = min(excess, len(block) - MIN_CODE_BLOCK)
            if cut <= 0:
                continue
            blocks[i] = ("code", _truncate(block, len(block) - cut))
            excess -= cut

    return "\n".join(b for _, b in blocks)


def _norm(s: str) -> str:
    for variant, plain in _TYPO_FOLD.items():
        s = s.replace(variant, plain)
    return " ".join(s.split())


def _matches(quote: str, source: str) -> bool:
    return quote in source or _norm(quote) in _norm(source)


def _cell(nb: ParsedNotebook, idx, expected_type: str):
    if not isinstance(idx, int) or isinstance(idx, bool):
        return None
    if idx < 0 or idx >= len(nb.cells):
        return None
    cell = nb.cells[idx]
    return cell if cell.cell_type == expected_type else None


def _line_span(quote: str, source: str):
    """(start, end) 1-indexed lines the quote occupies, or None. Recomputed from
    the match position; the model's own line numbers are never consulted."""
    idx = source.find(quote)
    if idx != -1:
        start = source.count("\n", 0, idx) + 1
        return start, start + quote.count("\n")
    q_norm = _norm(quote)
    lines = source.split("\n")
    for i in range(len(lines)):
        # A match that starts on line i ends within len(line_i) + len(q_norm)
        # chars of the window, so bound the extension by the FIRST line's length
        # — not a fixed slack that a single long line would blow past.
        limit = len(_norm(lines[i])) + len(q_norm) + 1
        for j in range(i, len(lines)):
            joined = _norm(" ".join(lines[i : j + 1]))
            if q_norm in joined:
                return i + 1, j + 1
            if len(joined) > limit:
                break
    return None


def _ground_claim(raw, nb, dropped):
    cell = raw.get("cell")
    quote = raw.get("quote", "")
    if len(_norm(quote)) < MIN_CLAIM_QUOTE:
        dropped.append(f"claim dropped: quote too short in cell {cell}")
        return None
    md = _cell(nb, cell, "markdown")
    if md is None:
        dropped.append(f"claim dropped: cell {cell} is not a markdown cell")
        return None
    if not _matches(quote, md.source):
        dropped.append(f"claim dropped: quote not verbatim in cell {cell}")
        return None
    kind = raw.get("kind", "")
    if kind not in CLAIM_KINDS:
        kind = "descriptive"
    code_cells = [
        x for x in raw.get("code_cells", []) if isinstance(x, int) and not isinstance(x, bool)
    ]
    return Claim(cell=cell, quote=quote, kind=kind, code_cells=code_cells)


def _ground_finding(raw, nb, enabled, taxonomy, dropped):
    flaw_id = raw.get("flaw_id", "")
    if flaw_id not in enabled:
        dropped.append(f"finding dropped: flaw_id {flaw_id!r} is not narrative-enabled")
        return None
    claim = raw.get("claim_span") or {}
    code = raw.get("code_span") or {}
    claim_cell = claim.get("cell")
    claim_quote = claim.get("quote", "")
    if len(_norm(claim_quote)) < MIN_CLAIM_QUOTE:
        dropped.append(f"finding dropped: claim quote too short in cell {claim_cell}")
        return None
    md = _cell(nb, claim_cell, "markdown")
    if md is None:
        dropped.append(f"finding dropped: claim cell {claim_cell} is not a markdown cell")
        return None
    if not _matches(claim_quote, md.source):
        dropped.append(f"finding dropped: claim quote not verbatim in cell {claim_cell}")
        return None
    code_quote = code.get("quote", "")
    if len(_norm(code_quote)) < MIN_CODE_QUOTE:
        dropped.append(f"finding dropped: code quote too short in cell {code.get('cell')}")
        return None
    if len(code_quote) > MAX_CODE_QUOTE:
        dropped.append(f"finding dropped: code quote exceeds {MAX_CODE_QUOTE} chars")
        return None
    code_cell = code.get("cell")
    cc = _cell(nb, code_cell, "code")
    if cc is None:
        dropped.append(f"finding dropped: code cell {code_cell} is not a code cell")
        return None
    span = _line_span(code_quote, cc.source)
    if span is None:
        dropped.append(f"finding dropped: code quote not verbatim in cell {code_cell}")
        return None
    try:
        conf = float(raw.get("model_confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    d = taxonomy[flaw_id]
    return NarrativeFinding(
        flaw_id=flaw_id,
        claim_cell=claim_cell,
        claim_quote=claim_quote,
        code_cell=code_cell,
        code_line_start=span[0],
        code_line_end=span[1],
        code_quote=code_quote,
        failure_scenario=str(raw.get("failure_scenario", d.failure_scenario)),
        fix=str(raw.get("fix", d.fix)),
        model_confidence=conf,
    )


def detect_narrative(nb: ParsedNotebook, backend) -> NarrativeResult:
    system = build_system_prompt()
    user = package_notebook(nb)
    try:
        raw = backend.complete(system, user, NARRATIVE_SCHEMA)
    except BackendError as exc:
        return NarrativeResult(claims=[], findings=[], dropped=[f"backend error: {exc}"])

    taxonomy = load_taxonomy()
    enabled = {fid for fid, d in taxonomy.items() if d.narrative_enabled}
    dropped: list[str] = []

    claims = []
    for c in raw.get("claims", []) or []:
        grounded = _ground_claim(c, nb, dropped)
        if grounded is not None:
            claims.append(grounded)

    findings = []
    for f in raw.get("findings", []) or []:
        grounded = _ground_finding(f, nb, enabled, taxonomy, dropped)
        if grounded is not None:
            findings.append(grounded)

    return NarrativeResult(claims=claims, findings=findings, dropped=dropped)
