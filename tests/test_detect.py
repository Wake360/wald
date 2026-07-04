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


# -- behaviors fixed by the 2026-07-04 dogfood review (see evals/) --


def test_leakage_silent_on_estimator_fit_full_data():
    # model fit for visualization is not preprocessing leakage
    notebook = nb([
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
        ("code", "clf = KNeighborsClassifier()\nclf.fit(X, y)"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


def test_leakage_silent_on_label_encoder():
    notebook = nb([
        ("code", "le = LabelEncoder()\ny = le.fit_transform(y)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


def test_leakage_silent_on_name_reuse_across_sections():
    # second dataset reuses the name X; the sections are unrelated
    notebook = nb([
        ("code", "X = scaler.fit_transform(X)"),
        ("code", "X = load_other_dataset()"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


def test_leakage_silent_on_unassigned_fit_transform():
    # result discarded: fitted statistics never reach the split
    notebook = nb([
        ("code", "scaler.fit_transform(X)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


def test_leakage_flagged_full_fit_then_transform_split_part():
    notebook = nb([
        ("code", "scaler = StandardScaler()"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
        ("code", "scaler.fit(X)\nX_te_s = scaler.transform(X_te)"),
    ])
    assert "leakage-fit-before-split" in flag_ids(notebook)


def test_leakage_supervised_selector_before_cv_flagged():
    notebook = nb([
        ("code", "skb = SelectKBest(chi2, k=15)\nskb.fit(X, y)"),
        ("code", "Xs = skb.transform(X)"),
        ("code", "scores = cross_val_score(lr, Xs, y, cv=5)"),
    ])
    assert "leakage-fit-before-split" in flag_ids(notebook)


def test_leakage_unsupervised_fit_before_cv_is_candidate():
    notebook = nb([
        ("code", "scaler = StandardScaler()\nXs = scaler.fit_transform(X)"),
        ("code", "scores = cross_val_score(clf, Xs, y, cv=5)"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-fit-before-split"]
    assert flags and all(f.confidence < 0.8 for f in flags)


def test_leakage_silent_on_pipeline_estimator_in_cv():
    # a transformer inside the CV'd pipeline is refit per fold: correct usage
    notebook = nb([
        ("code", "knn_transformer = FeatureFromRegressor(knn)\nknn_transformer.fit_transform(X, y)"),
        ("code", "pipe = Pipeline([('prep', knn_transformer), ('svr', SVR())])"),
        ("code", "scores = cross_val_score(pipe, X, y, cv=3)"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


def test_leakage_self_statistic_imputation_before_split():
    notebook = nb([
        ("code", "data['a'] = data['a'].replace(0, data['a'].median())"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(data[cols], data['y'])"),
    ])
    assert "leakage-fit-before-split" in flag_ids(notebook)


def test_leakage_imputation_one_flag_per_cell():
    src = "\n".join(
        f"data['{c}'] = data['{c}'].replace(0, data['{c}'].median())" for c in "abc"
    )
    notebook = nb([
        ("code", src),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(data[cols], data['y'])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-fit-before-split"]
    assert len(flags) == 1


def test_leakage_imputation_on_train_split_only_silent():
    notebook = nb([
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
        ("code", "X_tr = X_tr.fillna(X_tr.median())"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


# -- behaviors fixed by the 2026-07-04 code review --


def test_leakage_unsupervised_multi_name_arg_stays_candidate():
    # df[num_cols] mentions two names but is one argument: not "fitted with labels"
    notebook = nb([
        ("code", "scaler = StandardScaler()\nXs = scaler.fit_transform(df[num_cols])"),
        ("code", "scores = cross_val_score(clf, Xs, y, cv=5)"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-fit-before-split"]
    assert flags and all(f.confidence < 0.8 for f in flags)


def test_leakage_supervised_selector_single_frame_spelling_flagged():
    notebook = nb([
        ("code", "skb = SelectKBest(chi2, k=15)\nskb.fit(data.drop(columns=[t]), data[t])"),
        ("code", "Xs = skb.transform(data)"),
        ("code", "scores = cross_val_score(lr, Xs, data[t], cv=5)"),
    ])
    assert "leakage-fit-before-split" in flag_ids(notebook)


def test_leakage_imputation_silent_after_frame_rebind():
    notebook = nb([
        ("code", "data['a'] = data['a'].replace(0, data['a'].median())"),
        ("code", "data = load_other_dataset()"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(data[cols], data['y'])"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


def test_leakage_receiver_rebind_does_not_whitelist_earlier_fit():
    notebook = nb([
        ("code", "enc = StandardScaler()\nenc.fit(X)\nXs = enc.transform(X)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(Xs, y)"),
        ("code", "enc = LabelEncoder()\ny2 = enc.fit_transform(labels)"),
    ])
    assert "leakage-fit-before-split" in flag_ids(notebook)


def test_leakage_silent_on_estimator_in_shared_container():
    notebook = nb([
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
        ("code", "models['clf'].fit(X, y)"),
        ("code", "Xs = models['scaler'].transform(X_te)"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


def test_leakage_two_fits_same_line_two_flags():
    notebook = nb([
        ("code", "Xa = s1.fit_transform(A); Xb = s2.fit_transform(B)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(Xa, Xb)"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-fit-before-split"]
    assert len(flags) == 2


def test_leakage_imputation_one_flag_per_frame_across_cells():
    notebook = nb([
        ("code", "df['a'] = df['a'].fillna(df['a'].median())"),
        ("code", "df['b'] = df['b'].fillna(df['b'].median())"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-fit-before-split"]
    assert len(flags) == 1


def test_leakage_cv_groups_kwarg_not_a_data_seed():
    notebook = nb([
        ("code", "groups = df['user_id']\ndf['s'] = scaler.fit_transform(df[cols])"),
        ("code", "scores = cross_val_score(clf, X_other, y_other, groups=groups)"),
    ])
    assert not [f for f in run_static(notebook) if f.flaw_id == "leakage-fit-before-split"]
