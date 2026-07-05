import warnings

import pytest

from wald.ingest import parse_notebook


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
