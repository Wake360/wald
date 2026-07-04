from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

FLAWS_PATH = Path(__file__).parent / "flaws.yaml"
FUSION_PATH = Path(__file__).parent / "fusion.yaml"


@dataclass(frozen=True)
class FlawDef:
    id: str
    flaw_class: str
    layer: str
    severity: str
    definition: str
    failure_scenario: str
    fix: str
    book_anchor: str = ""
    narrative_enabled: bool = False
    disqualifiers: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, FlawDef]:
    raw = yaml.safe_load(FLAWS_PATH.read_text())
    defs = {}
    for r in raw:
        defs[r["id"]] = FlawDef(
            id=r["id"],
            flaw_class=r["class"],
            layer=r["layer"],
            severity=r["severity"],
            definition=r["definition"].strip(),
            failure_scenario=r["failure_scenario"].strip(),
            fix=r["fix"].strip(),
            book_anchor=r.get("book_anchor", ""),
            narrative_enabled=r.get("narrative_enabled", False),
            disqualifiers=tuple(r.get("disqualifiers", ())),
        )
    return defs


@dataclass(frozen=True)
class StaticMatch:
    flaw_id: str
    confidence_min: float
    confidence_max: float


@dataclass(frozen=True)
class NarrativeMatch:
    join: str = ""
    flaw_id: str = ""
    claim_kind: str = ""
    any_enabled_finding: bool = False


@dataclass(frozen=True)
class FusionRule:
    id: str
    static: StaticMatch | None
    narrative: NarrativeMatch
    emit_flaw_id: str
    emit_confidence: float
    verify: bool


@lru_cache(maxsize=1)
def load_fusion_rules() -> tuple[FusionRule, ...]:
    raw = yaml.safe_load(FUSION_PATH.read_text())
    rules = []
    for r in raw:
        static = r["static"]
        rules.append(
            FusionRule(
                id=r["id"],
                static=StaticMatch(**static) if static else None,
                narrative=NarrativeMatch(**r["narrative"]),
                emit_flaw_id=r["emit"]["flaw_id"],
                emit_confidence=r["emit"]["confidence"],
                verify=r["verify"],
            )
        )
    return tuple(rules)
