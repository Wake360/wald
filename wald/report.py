"""Flag rendering (markdown + JSON) and exit-code policy."""

from __future__ import annotations

import json
from dataclasses import asdict

from .detect import DEFAULT_CONFIDENCE_FLOOR, Flag
from .taxonomy import load_taxonomy

SEVERITY_ORDER = {"info": 0, "medium": 1, "high": 2}


def exit_code(flags: list[Flag], floor: float = DEFAULT_CONFIDENCE_FLOOR,
              severity_gate: str = "high") -> int:
    """0 = clean, 1 = medium findings, 2 = at or above the gate severity.
    Only flags at/above the confidence floor count."""
    confident = [f for f in flags if f.confidence >= floor]
    if not confident:
        return 0
    worst = max(SEVERITY_ORDER[f.severity] for f in confident)
    return 2 if worst >= SEVERITY_ORDER[severity_gate] else 1


def checked_classes(flags: list[Flag], floor: float) -> list[str]:
    """Static classes checked and found clean (negative assurance)."""
    from .detect import STATIC_DECIDABLE

    flagged = {f.flaw_id for f in flags if f.confidence >= floor}
    return sorted(STATIC_DECIDABLE - flagged)


def to_json(path: str, flags: list[Flag], floor: float = DEFAULT_CONFIDENCE_FLOOR) -> str:
    return json.dumps(
        {
            "notebook": path,
            "flags": [asdict(f) for f in flags if f.confidence >= floor],
            "candidates": [asdict(f) for f in flags if f.confidence < floor],
            "clean_classes": checked_classes(flags, floor),
            "exit_code": exit_code(flags, floor),
        },
        indent=2,
    )


def to_markdown(path: str, flags: list[Flag], floor: float = DEFAULT_CONFIDENCE_FLOOR) -> str:
    taxonomy = load_taxonomy()
    confident = sorted(
        (f for f in flags if f.confidence >= floor),
        key=lambda f: -SEVERITY_ORDER[f.severity],
    )
    candidates = [f for f in flags if f.confidence < floor]

    lines = [f"# Wald report — {path}", ""]
    n_high = sum(1 for f in confident if f.severity == "high")
    n_med = sum(1 for f in confident if f.severity == "medium")
    lines.append(f"verdict: {n_high} high, {n_med} medium | static layer (no LLM)")
    lines.append("")

    for f in confident:
        d = taxonomy[f.flaw_id]
        lines += [
            f"## {f.severity.upper()}: {f.flaw_id}",
            f"- **Where:** cell {f.cell}, line {f.line}",
            f"- **Evidence:** {f.evidence}",
            f"- **Why it matters:** {d.definition}",
            f"- **Failure scenario:** {f.failure_scenario}",
            f"- **Fix:** {f.fix}",
            f"- **Confidence:** {f.confidence:.2f}",
            "",
        ]
    if candidates:
        lines.append("## Candidates (below confidence floor, need narrative layer)")
        for f in candidates:
            lines.append(f"- {f.flaw_id} (conf {f.confidence:.2f}, cell {f.cell}): {f.evidence}")
        lines.append("")
    clean = checked_classes(flags, floor)
    if clean:
        lines.append(f"## CLEAN (checked): {', '.join(clean)}")
        lines.append("")
    return "\n".join(lines)
