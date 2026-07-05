import json
from pathlib import Path

import pytest

from wald.cli import main


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_valid_json_missing_cells_exits_3(tmp_path, capsys):
    # valid JSON but not a notebook (no top-level "cells"): nbformat raises a
    # jsonschema ValidationError, which must still map to the exit-3 error path
    p = _write(tmp_path, "nocells.ipynb", '{"hello": "world"}')
    assert main(["check", p]) == 3
    err = capsys.readouterr().err
    assert err.startswith("wald: ")
    assert "Traceback" not in err


def test_valid_json_missing_cells_json_format_exits_3(tmp_path, capsys):
    p = _write(tmp_path, "nocells.ipynb", '{"hello": "world"}')
    assert main(["check", "--format", "json", p]) == 3


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.startswith("wald ")


def test_version_short_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["-V"])
    assert e.value.code == 0
    assert capsys.readouterr().out.startswith("wald ")


def test_format_sarif_accepted_and_valid_json(tmp_path, capsys):
    nb = {"cells": [
        {"cell_type": "code", "source": "x = 1\n", "outputs": [],
         "execution_count": None, "metadata": {}},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = _write(tmp_path, "clean.ipynb", json.dumps(nb))
    rc = main(["check", "--format", "sarif", p])
    data = json.loads(capsys.readouterr().out)  # must be a single valid JSON document
    assert data["version"] == "2.1.0"
    assert rc == 0


def test_heldout_refusal_has_prefix_and_exits_3(tmp_path, capsys):
    (tmp_path / "clean").mkdir()
    (tmp_path / "clean" / "foo.ipynb").write_text(
        '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
    )
    (tmp_path / "MANIFEST.json").write_text(json.dumps({
        "clean": [{"file": "clean/foo.ipynb", "split": "heldout"}], "mutants": [],
    }))
    (tmp_path / "replay" / "detector").mkdir(parents=True)
    (tmp_path / "replay" / "verifier").mkdir(parents=True)
    rc = main([
        "check", "--llm", "--replay-dir", str(tmp_path / "replay"),
        str(tmp_path / "clean" / "foo.ipynb"),
    ])
    assert rc == 3
    assert capsys.readouterr().err.startswith("wald: ")
