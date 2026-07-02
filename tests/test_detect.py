from wald.detect import run_static
from wald.ingest import Cell, ParsedNotebook


def nb(cells: list[tuple[str, str]], outputs: dict[int, str] | None = None) -> ParsedNotebook:
    outputs = outputs or {}
    return ParsedNotebook(
        path=None,
        cells=[
            Cell(index=i, cell_type=t, source=s, outputs_text=outputs.get(i, ""))
            for i, (t, s) in enumerate(cells)
        ],
    )


def flag_ids(notebook, floor=0.8):
    return {f.flaw_id for f in run_static(notebook) if f.confidence >= floor}


def test_leakage_flagged_when_fit_precedes_split():
    notebook = nb([
        ("code", "X = scaler.fit_transform(X)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
    ])
    assert "leakage-fit-before-split" in flag_ids(notebook)


def test_leakage_silent_on_fit_after_split():
    notebook = nb([
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
        ("code", "X_tr = scaler.fit_transform(X_tr)\nX_te = scaler.transform(X_te)"),
        ("code", "model.fit(X_tr, y_tr)"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


def test_leakage_flagged_through_derived_name():
    notebook = nb([
        ("code", "feats = vectorizer.fit_transform(texts)"),
        ("code", "X = feats"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
    ])
    assert "leakage-fit-before-split" in flag_ids(notebook)


def test_multiple_testing_flagged_in_loop_without_correction():
    notebook = nb([
        ("code", "from scipy.stats import ttest_ind"),
        ("code", "for c in cols:\n    s, p = ttest_ind(a[c], b[c])"),
    ])
    assert "testing-multiple-uncorrected" in flag_ids(notebook)


def test_multiple_testing_silent_with_correction():
    notebook = nb([
        ("code", "from statsmodels.stats.multitest import multipletests"),
        ("code", "for c in cols:\n    s, p = ttest_ind(a[c], b[c])"),
        ("code", "rej, adj, _, _ = multipletests(pvals, method='holm')"),
    ])
    assert "testing-multiple-uncorrected" not in flag_ids(notebook)


def test_multiple_testing_silent_on_few_tests():
    notebook = nb([("code", "s, p = ttest_ind(a, b)\ns2, p2 = ttest_ind(c, d)")])
    assert "testing-multiple-uncorrected" not in flag_ids(notebook)


VC_IMBALANCED = "0    0.90\n1    0.10\nName: proportion, dtype: float64"
VC_BALANCED = "0    0.52\n1    0.48\nName: proportion, dtype: float64"


def test_baserate_flagged_accuracy_only_imbalanced():
    notebook = nb(
        [
            ("code", 'df["y"].value_counts(normalize=True)'),
            ("code", "acc = accuracy_score(y_te, pred)"),
        ],
        outputs={0: VC_IMBALANCED},
    )
    assert "baserate-accuracy-imbalanced" in flag_ids(notebook)


def test_baserate_silent_when_auc_present():
    notebook = nb(
        [
            ("code", 'df["y"].value_counts(normalize=True)'),
            ("code", "acc = accuracy_score(y_te, pred)\nauc = roc_auc_score(y_te, proba)"),
        ],
        outputs={0: VC_IMBALANCED},
    )
    assert "baserate-accuracy-imbalanced" not in flag_ids(notebook)


def test_baserate_silent_on_balanced_classes():
    notebook = nb(
        [
            ("code", 'df["y"].value_counts(normalize=True)'),
            ("code", "acc = accuracy_score(y_te, pred)"),
        ],
        outputs={0: VC_BALANCED},
    )
    assert "baserate-accuracy-imbalanced" not in flag_ids(notebook)


def test_baserate_candidate_without_balance_evidence():
    notebook = nb([("code", "acc = accuracy_score(y_te, pred)")])
    flags = run_static(notebook)
    match = [f for f in flags if f.flaw_id == "baserate-accuracy-imbalanced"]
    assert match and match[0].confidence < 0.8


def test_survivorship_candidate_below_floor():
    notebook = nb([
        ("code", 'df = df[df["status"] == "active"]'),
        ("code", 'df.groupby("cohort")["ltv"].mean()'),
    ])
    flags = run_static(notebook)
    match = [f for f in flags if f.flaw_id == "selection-survivorship-cohort"]
    assert match and match[0].confidence < 0.8


def test_survivorship_silent_on_scoped_variable_name():
    notebook = nb([
        ("code", 'active = df[df["status"] == "active"]'),
        ("code", 'active.groupby("cohort")["ltv"].mean()'),
    ])
    assert not [
        f for f in run_static(notebook) if f.flaw_id == "selection-survivorship-cohort"
    ]
