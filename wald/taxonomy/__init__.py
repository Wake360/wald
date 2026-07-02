from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

FLAWS_PATH = Path(__file__).parent / "flaws.yaml"


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
        )
    return defs
