import libcst as cst

from wald.dataflow import analyze, expr_names
from wald.ingest import Cell, ParsedNotebook


def nb_from_sources(*sources: str) -> ParsedNotebook:
    cells = [Cell(index=i, cell_type="code", source=s) for i, s in enumerate(sources)]
    return ParsedNotebook(path=None, cells=cells)


def chain_names(flow, names, at):
    _, bindings = flow.chain(set(names), at)
    return {n for n, _ in bindings}


def test_expr_names_attribute_base_only():
    node = cst.parse_expression("df.status == other[x].mean()")
    assert expr_names(node) == {"df", "other", "x"}


def test_chain_transitive():
    flow = analyze(nb_from_sources("a = raw * 2", "b = a + 1", "c = b"))
    assert {"a", "raw", "b", "c"} <= chain_names(flow, {"c"}, (3, 1))


def test_call_sites_receiver_and_loop_depth():
    flow = analyze(nb_from_sources(
        "scaler.fit_transform(X)",
        "for i in range(3):\n    stats.ttest_ind(a, b)",
    ))
    fit = next(c for c in flow.calls if c.name == "fit_transform")
    assert fit.receiver == "scaler" and fit.arg_names == {"X"} and fit.loop_depth == 0
    test = next(c for c in flow.calls if c.name == "ttest_ind")
    assert test.loop_depth == 1


def test_constructor_chained_call_yields_call_site():
    flow = analyze(nb_from_sources("X = StandardScaler().fit_transform(X)"))
    fit = next(c for c in flow.calls if c.name == "fit_transform")
    assert fit.receiver == "StandardScaler()" and fit.arg_names == {"X"}
    assert any(ev.call is fit for ev in flow.assigns)


def test_loop_vars_recorded_on_call_sites():
    flow = analyze(nb_from_sources(
        "for c in cols:\n    ttest_ind(a[c], b[c])\nttest_ind(a, b)",
    ))
    looped = next(c for c in flow.calls if c.loop_depth == 1)
    top = next(c for c in flow.calls if c.loop_depth == 0)
    assert looped.loop_vars == {"c"} and top.loop_vars == set()


def test_split_outputs_do_not_reach_split_input_chain():
    flow = analyze(nb_from_sources(
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y)",
        "X_tr = scaler.fit_transform(X_tr)",
    ))
    split = next(c for c in flow.calls if c.name == "train_test_split")
    assert "X_tr" not in chain_names(flow, {"X", "y"}, (split.cell, split.line))


def test_magics_are_stripped_not_fatal():
    flow = analyze(nb_from_sources("%matplotlib inline\na = 1"))
    assert not flow.parse_errors
    assert any("a" in ev.targets for ev in flow.assigns)


def test_time_line_magic_keeps_wrapped_statement():
    flow = analyze(nb_from_sources(
        "%matplotlib inline\n%time X = scaler.fit_transform(X)",
    ))
    assert not flow.parse_errors
    assert any(c.name == "fit_transform" for c in flow.calls)
    assert any("X" in ev.targets for ev in flow.assigns)


def test_timeit_with_options_falls_back_to_pass():
    # "-n 100 f(x)" is not Python; stripping must not create a parse error
    flow = analyze(nb_from_sources("%matplotlib inline\n%timeit -n 100 f(x)\na = 1"))
    assert not flow.parse_errors
    assert any("a" in ev.targets for ev in flow.assigns)


def test_indented_magic_preserves_block():
    flow = analyze(nb_from_sources("if COLAB:\n    %pip install x\na = 1"))
    assert not flow.parse_errors
    assert any("a" in ev.targets for ev in flow.assigns)


def test_percent_continuation_line_not_treated_as_magic():
    flow = analyze(nb_from_sources("print('x %s'\n      % (val,))\na = val"))
    assert not flow.parse_errors
    assert any("a" in ev.targets for ev in flow.assigns)


def test_percent_continuation_survives_alongside_real_magic():
    flow = analyze(nb_from_sources("%matplotlib inline\ntotal = (a\n % b)"))
    assert not flow.parse_errors
    assert any("total" in ev.targets for ev in flow.assigns)


def test_nonpython_cell_magic_skipped_silently():
    flow = analyze(nb_from_sources('%%writefile x.proto\nsyntax = "proto3";\nmessage P {}'))
    assert not flow.parse_errors


def test_python_cell_magic_body_still_analyzed():
    flow = analyze(nb_from_sources("%%time\na = raw * 2"))
    assert not flow.parse_errors
    assert any("a" in ev.targets for ev in flow.assigns)


def test_python_cell_magic_with_syntax_error_is_recorded():
    flow = analyze(nb_from_sources("%%time\nfor i in range(3)\n    pass"))
    assert flow.parse_errors == [0]


def test_chain_kills_on_reassign():
    flow = analyze(nb_from_sources(
        "X = scaler.fit_transform(X)",
        "X = load_other()",
        "split(X)",
    ))
    names = chain_names(flow, {"X"}, (2, 1))
    assert "X" in names and "scaler" not in names


def test_function_local_does_not_kill_global_binding():
    flow = analyze(nb_from_sources(
        "X = scaler.fit_transform(X)",
        "def make_demo():\n    X = synth()\n    return X",
        "train_test_split(X, y)",
    ))
    assert "scaler" in chain_names(flow, {"X"}, (2, 1))


def test_subscript_receivers_keep_literal_keys():
    flow = analyze(nb_from_sources("models['clf'].fit(X, y)\nmodels['scaler'].transform(X)"))
    fit = next(c for c in flow.calls if c.name == "fit")
    transform = next(c for c in flow.calls if c.name == "transform")
    assert fit.receiver != transform.receiver
    assert fit.receiver_base == "models"


def test_assign_event_shares_call_identity():
    flow = analyze(nb_from_sources("Xa = s1.fit_transform(A); Xb = s2.fit_transform(B)"))
    fits = [c for c in flow.calls if c.name == "fit_transform"]
    linked = [ev.call for ev in flow.assigns if ev.call is not None]
    assert len(fits) == 2 and fits[0] in linked and fits[1] in linked


def test_oversized_cell_skipped():
    from wald.dataflow import MAX_CELL_SOURCE_BYTES

    big = "x = fit(y)\n" + "a = 1  # " + "A" * (MAX_CELL_SOURCE_BYTES + 1)
    flow = analyze(nb_from_sources(big, "z = fit(w)"))
    # oversized cell 0 is skipped, cell 1 is still analyzed
    assert all(c.cell == 1 for c in flow.calls)
    assert flow.parse_errors == []
