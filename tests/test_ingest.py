import warnings

import pytest

from wald.ingest import parse_notebook, parse_script


def test_parse_script_plain_single_cell(tmp_path):
    p = tmp_path / "plain.py"
    p.write_text("x = 1\ny = 2\n")
    nb = parse_script(str(p))
    assert len(nb.cells) == 1
    c = nb.cells[0]
    assert c.index == 0 and c.cell_type == "code" and c.start_line == 1
    assert c.source == "x = 1\ny = 2\n"


def test_parse_script_percent_split_code_and_markdown(tmp_path):
    p = tmp_path / "pct.py"
    p.write_text(
        "# %% Intro [markdown]\n"
        "# Title\n"
        "#\n"
        "# Body line\n"
        "# %%\n"
        "a = 1\n"
        "b = 2\n"
    )
    nb = parse_script(str(p))
    assert [c.index for c in nb.cells] == list(range(len(nb.cells)))
    assert len(nb.cells) == 2
    md, code = nb.cells
    assert md.cell_type == "markdown"
    assert md.source == "Title\n\nBody line"  # leading "# "/"#" stripped per line
    assert md.start_line == 2  # 1-indexed file line of "# Title"
    assert code.cell_type == "code"
    assert code.source == "a = 1\nb = 2"
    assert code.start_line == 6  # 1-indexed file line of "a = 1"


def test_parse_script_leading_code_before_first_marker(tmp_path):
    p = tmp_path / "lead.py"
    p.write_text("import os\n# %%\nx = 1\n")
    nb = parse_script(str(p))
    assert [c.index for c in nb.cells] == [0, 1]
    assert nb.cells[0].source == "import os" and nb.cells[0].start_line == 1
    assert nb.cells[1].source == "x = 1" and nb.cells[1].start_line == 3


def test_parse_script_crlf(tmp_path):
    p = tmp_path / "crlf.py"
    p.write_bytes(b"# %%\r\na = 1\r\n# %%\r\nb = 2\r\n")
    nb = parse_script(str(p))
    assert [c.source for c in nb.cells] == ["a = 1", "b = 2"]
    assert [c.start_line for c in nb.cells] == [2, 4]


def test_parse_script_markdown_strips_comment_prefix(tmp_path):
    p = tmp_path / "md.py"
    p.write_text("# %% [markdown]\n# Heading\n#\n#no-space\nplain\n")
    nb = parse_script(str(p))
    assert nb.cells[0].cell_type == "markdown"
    # "# " and bare "#" stripped; "#no-space" loses only the "#"; non-comment kept
    assert nb.cells[0].source == "Heading\n\nno-space\nplain"


def test_parse_script_start_line_offsets(tmp_path):
    p = tmp_path / "offsets.py"
    p.write_text("# %%\na = 1\nb = 2\n# %%\nc = 3\n")
    nb = parse_script(str(p))
    assert [c.start_line for c in nb.cells] == [2, 5]  # first body line of each cell


def test_parse_script_index_equals_position(tmp_path):
    p = tmp_path / "idx.py"
    p.write_text("# %% [markdown]\n# doc\n# %%\na = 1\n# %%\nb = 2\n")
    nb = parse_script(str(p))
    assert [c.index for c in nb.cells] == list(range(len(nb.cells)))


def test_parse_script_non_utf8_raises_unicodedecode(tmp_path):
    p = tmp_path / "bad.py"
    p.write_bytes(b"# %%\nx = '\xff\xfe'\n")  # invalid UTF-8 bytes
    with pytest.raises(UnicodeDecodeError):
        parse_script(str(p))


def test_cells_null_raises_valueerror(tmp_path):
    # "cells": null reaches nbformat's rejoin_lines and iterates None; the CLI
    # maps ValueError to a clean exit 3 rather than an uncaught TypeError
    nb = tmp_path / "cells_null.ipynb"
    nb.write_text('{"nbformat":4,"nbformat_minor":5,"cells":null}')
    with pytest.raises(ValueError):
        parse_notebook(str(nb))


def test_missing_cell_id_no_warning(tmp_path):
    nb = tmp_path / "no_id.ipynb"
    nb.write_text(
        '{"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":'
        '[{"cell_type":"code","source":"a = 1\\n","metadata":{},'
        '"outputs":[],"execution_count":null}]}'
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_notebook(str(nb))
    assert [w for w in caught if "MissingIDField" in w.category.__name__] == []
