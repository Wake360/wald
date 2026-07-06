import json
from pathlib import Path

import pytest

from wald.cli import main
from wald.llm import BackendError

LEAKY = str(Path(__file__).parent.parent / "examples" / "leaky.ipynb")


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


class _Det:
    provider = "anthropic"
    model = "det"
    kind = "api"

    @property
    def gate_eligible(self):
        return True

    def complete(self, system, user, schema=None):
        raise BackendError("upstream 503")


class _Ver:
    provider = "openai"
    model = "ver"
    kind = "api"

    @property
    def gate_eligible(self):
        return True

    def complete(self, system, user, schema=None):
        return {"verdict": "unsupported", "reason": "x"}


class _OkDet:
    provider = "anthropic"
    model = "det"
    kind = "api"

    @property
    def gate_eligible(self):
        return True

    def complete(self, system, user, schema=None):
        return {"claims": [], "findings": []}


def test_cli_llm_run_reports_narrative_provenance(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    monkeypatch.setattr("wald.cli._llm_backends", lambda rd: (_OkDet(), _Ver()))
    nb = {"cells": [
        {"cell_type": "code", "source": "x = 1\n", "outputs": [],
         "execution_count": None, "metadata": {}},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = _write(tmp_path, "clean.ipynb", json.dumps(nb))
    rc = main(["check", "--llm", p])
    out = capsys.readouterr().out
    assert "static + narrative layers" in out  # provenance reflects the --llm run
    assert rc == 0


def test_cli_llm_backend_failure_exits_3_fail_loud(monkeypatch, capsys):
    # an api outage during `check --llm` must not read like a clean notebook:
    # the dropped backend error propagates to the CLI, which exits 3.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    monkeypatch.setattr("wald.cli._llm_backends", lambda rd: (_Det(), _Ver()))
    assert main(["check", "--llm", LEAKY]) == 3
    err = capsys.readouterr().err
    assert "narrative layer failed" in err
    assert "Traceback" not in err


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
