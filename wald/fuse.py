"""Fusion: join static candidates with narrative findings into promoted flags.

Two join predicates, expressed by the rules in taxonomy/fusion.yaml:
survivorship (static candidate cell == narrative finding's code cell) and
pre-CV (a performance-generalization claim cites the candidate's CV sink cell).
A fused rule consumes its narrative finding so it cannot also fire solo; the
solo rule emits the rest. Fused wins over solo on the same (flaw_id, code cell).

`run_full` wires the pipeline: static -> narrative -> fuse -> verify. Confident
static flags are mechanical evidence and are never sent to the verifier; every
narrative-derived flag is, and an unsupported verdict drops it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .detect import DEFAULT_CONFIDENCE_FLOOR, Flag, run_static
from .narrative import NarrativeResult, detect_narrative
from .taxonomy import load_fusion_rules, load_taxonomy
from .verifier import verify_finding


@dataclass
class _VerifyTarget:
    flaw_id: str
    claim_cell: int
    claim_quote: str
    code_cell: int
    code_quote: str


def _emit_flaw_id(rule, source_flaw_id: str) -> str:
    return source_flaw_id if rule.emit_flaw_id == "inherit" else rule.emit_flaw_id


def _fused_survivorship(rule, cand: Flag, finding) -> Flag:
    flaw_id = _emit_flaw_id(rule, cand.flaw_id)
    d = load_taxonomy()[flaw_id]
    return Flag(
        flaw_id=flaw_id,
        severity=d.severity,
        confidence=rule.emit_confidence,
        cell=finding.code_cell,
        line=finding.code_line_start,
        evidence=(
            f"static survivorship filter (cell {cand.cell}) confirmed by a narrative "
            f"population claim (cell {finding.claim_cell})"
        ),
        failure_scenario=finding.failure_scenario,
        fix=finding.fix,
        extra={
            "rule": rule.id,
            "narrative_derived": True,
            "claim_span": {"cell": finding.claim_cell, "quote": finding.claim_quote},
            "code_span": {"cell": finding.code_cell, "quote": finding.code_quote},
            "static_confidence": cand.confidence,
            "model_confidence": finding.model_confidence,
        },
    )


def _fused_precv(rule, cand: Flag, claim) -> Flag:
    flaw_id = _emit_flaw_id(rule, cand.flaw_id)
    d = load_taxonomy()[flaw_id]
    return Flag(
        flaw_id=flaw_id,
        severity=d.severity,
        confidence=rule.emit_confidence,
        cell=cand.cell,
        line=cand.line,
        evidence=(
            f"pre-CV fit candidate (cell {cand.cell}) confirmed by a "
            f"performance-generalization claim (cell {claim.cell})"
        ),
        failure_scenario=d.failure_scenario,
        fix=d.fix,
        extra={
            "rule": rule.id,
            "narrative_derived": True,
            "claim_span": {"cell": claim.cell, "quote": claim.quote},
            "code_span": {"cell": cand.cell, "quote": cand.evidence},
            "static_confidence": cand.confidence,
        },
    )


def _solo(rule, finding) -> Flag:
    flaw_id = _emit_flaw_id(rule, finding.flaw_id)
    d = load_taxonomy()[flaw_id]
    return Flag(
        flaw_id=flaw_id,
        severity=d.severity,
        confidence=rule.emit_confidence,
        cell=finding.code_cell,
        line=finding.code_line_start,
        evidence=f"narrative finding for `{flaw_id}` (no static candidate)",
        failure_scenario=finding.failure_scenario,
        fix=finding.fix,
        extra={
            "rule": rule.id,
            "narrative_derived": True,
            "claim_span": {"cell": finding.claim_cell, "quote": finding.claim_quote},
            "code_span": {"cell": finding.code_cell, "quote": finding.code_quote},
            "model_confidence": finding.model_confidence,
        },
    )


def fuse(static_flags: list[Flag], narrative: NarrativeResult, rules) -> list[Flag]:
    consumed: set[int] = set()
    fused: list[Flag] = []
    solo: list[Flag] = []

    for rule in rules:
        if rule.narrative.join == "code-cell-equals-candidate-cell":
            for cand in (
                f
                for f in static_flags
                if f.flaw_id == rule.static.flaw_id
                and rule.static.confidence_min <= f.confidence <= rule.static.confidence_max
            ):
                for finding in narrative.findings:
                    if finding.flaw_id == rule.narrative.flaw_id and finding.code_cell == cand.cell:
                        fused.append(_fused_survivorship(rule, cand, finding))
                        consumed.add(id(finding))
        elif rule.narrative.join == "claim-cites-sink-cell":
            for cand in (
                f
                for f in static_flags
                if f.flaw_id == rule.static.flaw_id
                and rule.static.confidence_min <= f.confidence <= rule.static.confidence_max
            ):
                sink_cell = cand.extra.get("sink_cell")
                if sink_cell is None:
                    continue
                for claim in narrative.claims:
                    if claim.kind == rule.narrative.claim_kind and sink_cell in claim.code_cells:
                        fused.append(_fused_precv(rule, cand, claim))
        elif rule.narrative.any_enabled_finding:
            for finding in narrative.findings:
                if id(finding) not in consumed:
                    solo.append(_solo(rule, finding))

    # dedup by (flaw_id, code cell); fused precedes solo, so fused wins
    result: list[Flag] = []
    seen: set[tuple[str, int]] = set()
    for flag in fused + solo:
        key = (flag.flaw_id, flag.cell)
        if key in seen:
            continue
        seen.add(key)
        result.append(flag)
    return result


def run_full_traced(nb, det_backend, ver_backend) -> tuple[list[Flag], NarrativeResult, list[Flag]]:
    """(survivors, narrative result, pre-verify fused flags). The eval needs
    the intermediates — dropped_ungrounded from the NarrativeResult, true-flag
    survival from fused-vs-survivors — without a second detector call."""
    if det_backend.provider == ver_backend.provider:
        raise ValueError(
            "detector and verifier must use distinct providers "
            f"(both are {det_backend.provider!r})"
        )
    static_flags = run_static(nb)
    narrative = detect_narrative(nb, det_backend)
    fused = fuse(static_flags, narrative, load_fusion_rules())

    survivors = [f for f in static_flags if f.confidence >= DEFAULT_CONFIDENCE_FLOOR]
    for flag in fused:
        span = flag.extra["claim_span"]
        code = flag.extra["code_span"]
        target = _VerifyTarget(
            flaw_id=flag.flaw_id,
            claim_cell=span["cell"],
            claim_quote=span["quote"],
            code_cell=code["cell"],
            code_quote=code["quote"],
        )
        verdict = verify_finding(target, nb, ver_backend)
        if verdict.supported:
            flag.extra["verdict_reason"] = verdict.reason
            survivors.append(flag)
    return survivors, narrative, fused


def run_full(nb, det_backend, ver_backend) -> list[Flag]:
    return run_full_traced(nb, det_backend, ver_backend)[0]
