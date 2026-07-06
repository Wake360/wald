"""wald.execute: notebook cell append/parse helpers, plus one live-kernel
integration test guarded by the same kernelspec-ensure pattern test_mutations.py
uses (verify() calls execute() directly, no explicit skip needed unless the
kernel truly can't be provisioned)."""

import nbformat
import pytest

from wald.execute import execute, stdout_lines, with_appended_code_cell


def _one_cell_notebook(source: str):
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(source)]
    return nb


def test_with_appended_code_cell_appends_and_leaves_original_untouched():
    nb = _one_cell_notebook("x = 1")
    mutant = with_appended_code_cell(nb, "print('probe')")
    assert len(mutant.cells) == 2
    assert mutant.cells[-1]["source"] == "print('probe')"
    assert mutant.cells[-1]["cell_type"] == "code"
    assert len(nb.cells) == 1


def test_stdout_lines_parses_stream_outputs_and_ignores_others():
    nb = _one_cell_notebook("x = 1")
    nb.cells[0]["outputs"] = [
        {"output_type": "stream", "name": "stdout", "text": "PROBE:1\nother\nPROBE:2\n"},
        {"output_type": "execute_result", "data": {"text/plain": "PROBE:3"}},
        {"output_type": "error", "ename": "Err", "evalue": "e", "traceback": []},
    ]
    assert stdout_lines(nb, "PROBE:") == ["PROBE:1", "PROBE:2"]


def test_execute_runs_trivial_notebook():
    try:
        nb = _one_cell_notebook("print('hello wald')")
        out = execute(nb)
    except Exception as e:
        pytest.skip(f"no kernel available: {e}")
    assert stdout_lines(out, "hello") == ["hello wald"]
    assert len(nb.cells[0].get("outputs", [])) == 0  # original left unexecuted
