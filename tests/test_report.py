import json

from wald.detect import Flag
from wald.report import exit_code, to_json, to_markdown


def flag(flaw_id="leakage-fit-before-split", severity="high", confidence=0.92):
    return Flag(
        flaw_id=flaw_id, severity=severity, confidence=confidence,
        cell=5, line=2, evidence="ev", failure_scenario="fs", fix="fx",
    )


def test_exit_codes():
    assert exit_code([]) == 0
    assert exit_code([flag(confidence=0.5)]) == 0  # below floor
    assert exit_code([flag("testing-multiple-uncorrected", "medium")]) == 1
    assert exit_code([flag()]) == 2


def test_json_report_separates_candidates():
    data = json.loads(to_json("nb.ipynb", [flag(), flag(confidence=0.55)]))
    assert len(data["flags"]) == 1
    assert len(data["candidates"]) == 1
    assert data["exit_code"] == 2


def test_markdown_report_sections():
    md = to_markdown("nb.ipynb", [flag()])
    assert "HIGH: leakage-fit-before-split" in md
    assert "Failure scenario" in md
    assert "CLEAN (checked)" in md  # negative assurance for the other classes
