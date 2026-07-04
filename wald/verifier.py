"""Verifier pass: a second, skeptical model kills unsupported findings.

Skeptical by design (Příloha C.2) — the verifier's job is to kill, not
confirm; fail-closed on any parse trouble so a broken backend cannot
silently promote a flag.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ingest import ParsedNotebook
from .llm import Backend, BackendError
from .taxonomy import load_taxonomy

SYSTEM_PROMPT = """\
You are verifying a single flagged statistical flaw. You receive the
flaw definition, the quoted claim, and the quoted code.
Question: does the quoted evidence FULLY establish the flaw as defined?
Answer strictly: {"verdict": "supported" | "unsupported",
                  "reason": "<one sentence>"}
Be skeptical: if the surrounding text or code admits a legitimate
reading (scoped claim, control group present, correction applied,
effect size reported), answer "unsupported"."""


@dataclass
class Verdict:
    supported: bool
    reason: str


def _build_prompt(finding, nb: ParsedNotebook) -> str:
    d = load_taxonomy()[finding.flaw_id]
    disqualifiers = "\n".join(f"- {dq}" for dq in d.disqualifiers)
    claim_cell_source = next((c.source for c in nb.cells if c.index == finding.claim_cell), "")
    code_cell_source = next((c.source for c in nb.cells if c.index == finding.code_cell), "")
    return f"""FLAW DEFINITION ({finding.flaw_id}): {d.definition}

DISQUALIFIERS (answer unsupported if any applies):
{disqualifiers}

CLAIM QUOTE (cell {finding.claim_cell}): {finding.claim_quote}

FULL SOURCE OF CLAIM CELL {finding.claim_cell}:
{claim_cell_source}

CODE QUOTE (cell {finding.code_cell}): {finding.code_quote}

FULL SOURCE OF CODE CELL {finding.code_cell}:
{code_cell_source}"""


def verify_finding(finding, nb: ParsedNotebook, backend: Backend) -> Verdict:
    prompt = _build_prompt(finding, nb)
    try:
        response = backend.complete(SYSTEM_PROMPT, prompt)
    except BackendError as exc:
        return Verdict(supported=False, reason=f"backend error: {exc}")

    verdict = response.get("verdict")
    reason = response.get("reason", "")
    if verdict == "supported":
        return Verdict(supported=True, reason=reason)
    if verdict == "unsupported":
        return Verdict(supported=False, reason=reason)
    return Verdict(supported=False, reason=f"unparseable verdict: {verdict!r}")
