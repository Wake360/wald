import libcst as cst

from wald.dataflow import analyze, expr_names
from wald.ingest import Cell, ParsedNotebook


def nb_from_sources(*sources: str) -> ParsedNotebook:
    cells = [Cell(index=i, cell_type="code", source=s) for i, s in enumerate(sources)]
    return ParsedNotebook(path=None, cells=cells)


def test_expr_names_attribute_base_only():
    node = cst.parse_expression("df.status == other[x].mean()")
    assert expr_names(node) == {"df", "other", "x"}


def test_ancestors_transitive():
    flow = analyze(nb_from_sources("a = raw * 2", "b = a + 1", "c = b"))
    assert {"a", "raw", "b", "c"} <= flow.ancestors({"c"})


def test_call_sites_receiver_and_loop_depth():
    flow = analyze(nb_from_sources(
        "scaler.fit_transform(X)",
        "for i in range(3):\n    stats.ttest_ind(a, b)",
    ))
    fit = next(c for c in flow.calls if c.name == "fit_transform")
    assert fit.receiver == "scaler" and fit.arg_names == {"X"} and fit.loop_depth == 0
    test = next(c for c in flow.calls if c.name == "ttest_ind")
    assert test.loop_depth == 1


def test_split_outputs_are_not_ancestors_of_inputs():
    flow = analyze(nb_from_sources(
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)",
        "X_tr = scaler.fit_transform(X_tr)",
    ))
    assert "X_tr" not in flow.ancestors({"X", "y"})


def test_magics_are_stripped_not_fatal():
    flow = analyze(nb_from_sources("%matplotlib inline\na = 1"))
    assert not flow.parse_errors
    assert "a" in flow.deps
