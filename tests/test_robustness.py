import json

from wald.cli import main


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _one_cell_nb(source):
    return {"cells": [
        {"cell_type": "code", "source": source, "outputs": [],
         "execution_count": None, "metadata": {}},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def test_deep_nesting_cell_does_not_crash(tmp_path, capsys):
    # a 3000-deep nested list literal SIGSEGVs cst.parse_module natively; the
    # depth guard skips the cell, so the run completes and exits 0 (no findings)
    source = "x = " + "[" * 3000 + "1" + "]" * 3000 + "\n"
    p = _write(tmp_path, "deep.ipynb", json.dumps(_one_cell_nb(source)))
    rc = main(["check", p])
    assert rc == 0
    assert "Traceback" not in capsys.readouterr().err


def test_long_operator_chain_does_not_crash(tmp_path, capsys):
    # a 4000-term binary-op chain raises RecursionError from MetadataWrapper.visit;
    # the RecursionError catch skips the cell, so the run completes and exits 0
    source = "x = " + "+".join(["1"] * 4000) + "\n"
    p = _write(tmp_path, "chain.ipynb", json.dumps(_one_cell_nb(source)))
    rc = main(["check", p])
    assert rc == 0
    assert "Traceback" not in capsys.readouterr().err


def test_oversized_file_rejected_exit_3(tmp_path, capsys):
    p = tmp_path / "huge.ipynb"
    p.write_bytes(b"x" * (20 * 1024 * 1024 + 1))
    rc = main(["check", str(p)])
    assert rc == 3
    err = capsys.readouterr().err
    assert err.startswith("wald: ")
    assert "20 MB cap" in err
    assert "Traceback" not in err


def test_too_many_cells_rejected_exit_3(tmp_path, capsys):
    cells = [{"cell_type": "code", "source": "x = 1\n", "outputs": [],
              "execution_count": None, "metadata": {}} for _ in range(5001)]
    nb = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = _write(tmp_path, "manycells.ipynb", json.dumps(nb))
    rc = main(["check", str(p)])
    assert rc == 3
    err = capsys.readouterr().err
    assert err.startswith("wald: ")
    assert "5000-cell cap" in err
    assert "Traceback" not in err
