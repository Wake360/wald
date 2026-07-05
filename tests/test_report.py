import json
from pathlib import Path

from wald.cli import main
from wald.detect import Flag
from wald.report import exit_code, parse_warning, to_json, to_markdown, to_sarif
from wald.taxonomy import load_taxonomy

LEAKY = str(Path(__file__).parent.parent / "examples" / "leaky.ipynb")


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


def test_exit_code_info_severity_is_clean():
    # a confident info-severity flag must not read as a medium finding
    assert exit_code([flag("some-info-flaw", "info", 0.95)]) == 0
    assert exit_code([flag("some-info-flaw", "info", 0.95), flag()]) == 2


def test_to_json_honors_severity_gate():
    med = flag("testing-multiple-uncorrected", "medium")
    assert json.loads(to_json("nb.ipynb", [med]))["exit_code"] == 1  # default high gate
    assert json.loads(to_json("nb.ipynb", [med], severity_gate="medium"))["exit_code"] == 2


def test_parse_warning_surfaced_in_reports():
    assert parse_warning(0, 3) is None
    warning = parse_warning(2, 3)
    assert warning == "warning: 2 of 3 code cells could not be parsed; results are partial"
    assert warning in to_markdown("nb.ipynb", [], warning=warning)
    assert json.loads(to_json("nb.ipynb", [], warning=warning))["parse_warning"] == warning


def test_cli_missing_file_exits_3(capsys):
    assert main(["check", "/tmp/wald-does-not-exist-xyz.ipynb"]) == 3
    assert "no such file" in capsys.readouterr().err


def test_cli_floor_out_of_range_exits_3(capsys):
    assert main(["check", "--floor", "5", LEAKY]) == 3
    assert "between 0 and 1" in capsys.readouterr().err


def test_cli_llm_without_keys_exits_3(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["check", "--llm", LEAKY]) == 3
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "OPENAI_API_KEY" in err


def test_cli_eval_llm_without_keys_exits_3(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["eval", "--llm"]) == 3
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "OPENAI_API_KEY" in err


def test_cli_multi_notebook_json_is_one_array(capsys):
    rc = main(["check", "--format", "json", LEAKY, LEAKY])
    out = capsys.readouterr().out
    data = json.loads(out)  # single valid JSON document, not concatenated objects
    assert isinstance(data, list) and len(data) == 2
    assert rc == 2  # leaky is high-severity; a valid later parse must not downgrade


def test_cli_single_notebook_json_is_bare_object(capsys):
    main(["check", "--format", "json", LEAKY])
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, dict) and data["notebook"] == LEAKY


# --- design 8: structurally-broken-but-incomplete notebook nodes must lint,
# not crash (source/outputs null, missing keys, source as a line list)


def test_from_nbnode_tolerates_null_and_missing_fields():
    from nbformat import NotebookNode

    from wald.detect import run_static
    from wald.ingest import from_nbnode

    node = NotebookNode(cells=[
        NotebookNode(cell_type="code", source=None, outputs=None),  # both null
        NotebookNode(cell_type="code"),                             # both keys missing
        NotebookNode(cell_type="code", source=["x = ", "1\n"], outputs=[]),  # line-list source
    ])
    nb = from_nbnode(node)  # must not raise AttributeError/TypeError
    assert nb.cells[0].source == "" and nb.cells[0].outputs_text == ""
    assert nb.cells[1].source == ""
    assert nb.cells[2].source == "x = 1\n"  # line list joined
    assert run_static(nb) == []  # libcst must not choke on a coerced-empty source


def test_cli_null_source_notebook_exits_clean(tmp_path, capsys):
    # a file nbformat can read but whose cell carries "source": null used to
    # crash deep in libcst; it must now lint to a clean pass instead
    nb = {"cells": [
        {"cell_type": "code", "source": None, "outputs": [],
         "execution_count": None, "metadata": {}},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "null_source.ipynb"
    p.write_text(json.dumps(nb))
    assert main(["check", str(p)]) == 0


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


def test_candidate_class_not_certified_clean():
    # a below-floor flag is still a flag: its class must not appear as clean
    cand = flag(confidence=0.55)
    data = json.loads(to_json("nb.ipynb", [cand]))
    assert "leakage-fit-before-split" not in data["clean_classes"]
    md = to_markdown("nb.ipynb", [cand])
    clean_line = next(l for l in md.splitlines() if l.startswith("## CLEAN"))
    assert "leakage-fit-before-split" not in clean_line


def test_verdict_and_header_acknowledge_candidates():
    md = to_markdown("nb.ipynb", [flag(confidence=0.55)])
    assert "verdict: clean, 1 candidate below floor | static layer (no LLM)" in md
    assert "## Candidates (below confidence floor 0.80)" in md


def test_definition_bullet_labeled_flaw():
    md = to_markdown("nb.ipynb", [flag()])
    assert "- **Flaw:**" in md
    assert "Why it matters" not in md


def test_sarif_structure():
    sarif = json.loads(to_sarif([("nb.ipynb", [flag()])]))
    assert sarif["version"] == "2.1.0"
    assert len(sarif["runs"]) == 1
    driver = sarif["runs"][0]["tool"]["driver"]
    assert driver["name"] == "wald"
    assert driver["informationUri"] == "https://github.com/Wake360/wald"
    assert driver["version"]  # non-empty, from package metadata


def test_sarif_rule_table_completeness():
    sarif = json.loads(to_sarif([("nb.ipynb", [])]))
    rules = {r["id"]: r for r in sarif["runs"][0]["tool"]["driver"]["rules"]}
    taxonomy = load_taxonomy()
    assert set(rules) == set(taxonomy)
    for flaw_id, d in taxonomy.items():
        assert rules[flaw_id]["fullDescription"]["text"] == d.definition
        assert rules[flaw_id]["help"]["text"] == d.fix
        assert d.definition.startswith(rules[flaw_id]["shortDescription"]["text"].rstrip("."))


def test_sarif_level_mapping():
    high = flag()
    medium = flag("testing-multiple-uncorrected", "medium")
    sarif = json.loads(to_sarif([("nb.ipynb", [high, medium])]))
    levels = {r["ruleId"]: r["level"] for r in sarif["runs"][0]["results"]}
    assert levels["leakage-fit-before-split"] == "error"
    assert levels["testing-multiple-uncorrected"] == "warning"


def test_sarif_candidate_exclusion():
    cand = flag(confidence=0.55)
    sarif = json.loads(to_sarif([("nb.ipynb", [cand])]))
    run = sarif["runs"][0]
    assert run["results"] == []
    assert len(run["properties"]["candidates"]) == 1
    assert run["properties"]["candidates"][0]["ruleId"] == "leakage-fit-before-split"


def test_sarif_multi_notebook_single_log():
    sarif = json.loads(to_sarif([("a.ipynb", [flag()]), ("b.ipynb", [flag()])]))
    assert len(sarif["runs"]) == 1
    uris = {r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
            for r in sarif["runs"][0]["results"]}
    assert uris == {"a.ipynb", "b.ipynb"}


def test_sarif_result_fields():
    sarif = json.loads(to_sarif([("nb.ipynb", [flag()])]))
    r = sarif["runs"][0]["results"][0]
    assert r["message"]["text"] == "ev | Failure: fs | Fix: fx (cell 5, line 2)"
    assert r["locations"][0]["physicalLocation"]["region"]["startLine"] == 2
    assert r["properties"] == {"cell": 5, "confidence": 0.92}
