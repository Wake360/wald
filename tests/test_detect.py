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


def test_leakage_evidence_renders_names_plainly():
    notebook = nb([
        ("code", "X = scaler.fit_transform(X)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
    ])
    ev = next(f.evidence for f in run_static(notebook)
              if f.flaw_id == "leakage-fit-before-split")
    assert "consumes X," in ev
    assert "['X']" not in ev


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


def test_multiple_testing_not_suppressed_by_stockholm():
    # 'stockholm' contains 'holm'; a word-boundary match must not read it as a correction
    tests = "\n".join(f"s{i}, p{i} = ttest_ind(stockholm_df[a{i}], stockholm_df[b{i}])" for i in range(6))
    notebook = nb([("code", "stockholm_df = load('stockholm.csv')"), ("code", tests)])
    assert "testing-multiple-uncorrected" in flag_ids(notebook)


def test_multiple_testing_not_suppressed_by_fdr_named_variable():
    tests = "\n".join(f"s{i}, p{i} = ttest_ind(a{i}, b{i})" for i in range(6))
    notebook = nb([("code", "fdr_level = 0.1"), ("code", tests)])
    assert "testing-multiple-uncorrected" in flag_ids(notebook)


def test_multiple_testing_loop_evidence_not_fabricated():
    # one site parameterized by the loop var: confident, but no invented count
    notebook = nb([
        ("code", "from scipy.stats import ttest_ind"),
        ("code", "for c in cols:\n    s, p = ttest_ind(a[c], b[c])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "testing-multiple-uncorrected"]
    assert flags and flags[0].confidence == 0.9
    assert flags[0].extra["n_tests"] == 1 and "fwer" not in flags[0].extra
    assert "iteration count not statically known" in flags[0].evidence
    assert ">= 10" not in flags[0].evidence


def test_multiple_testing_resampling_loop_is_candidate():
    # permutation loop: loop var never reaches the test args — one hypothesis
    notebook = nb([
        ("code", "from scipy.stats import ttest_ind"),
        ("code", "for i in range(1000):\n    t, p = ttest_ind(a[perm[:m]], a[perm[m:]])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "testing-multiple-uncorrected"]
    assert flags and all(f.confidence < 0.8 for f in flags)


VC_IMBALANCED = "0    0.90\n1    0.10\nName: proportion, dtype: float64"
VC_BALANCED = "0    0.52\n1    0.48\nName: proportion, dtype: float64"


SPLIT_ON_Y = 'X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df["y"])'


def test_baserate_flagged_accuracy_only_imbalanced():
    notebook = nb(
        [
            ("code", 'df["y"].value_counts(normalize=True)'),
            ("code", SPLIT_ON_Y),
            ("code", "acc = accuracy_score(y_te, pred)"),
        ],
        outputs={0: VC_IMBALANCED},
    )
    assert "baserate-accuracy-imbalanced" in flag_ids(notebook)


def test_baserate_silent_when_auc_present():
    notebook = nb(
        [
            ("code", 'df["y"].value_counts(normalize=True)'),
            ("code", SPLIT_ON_Y),
            ("code", "acc = accuracy_score(y_te, pred)\nauc = roc_auc_score(y_te, proba)"),
        ],
        outputs={0: VC_IMBALANCED},
    )
    assert "baserate-accuracy-imbalanced" not in flag_ids(notebook)


def test_baserate_silent_on_balanced_classes():
    notebook = nb(
        [
            ("code", 'df["y"].value_counts(normalize=True)'),
            ("code", SPLIT_ON_Y),
            ("code", "acc = accuracy_score(y_te, pred)"),
        ],
        outputs={0: VC_BALANCED},
    )
    assert "baserate-accuracy-imbalanced" not in flag_ids(notebook)


def test_baserate_unlinked_skewed_feature_not_confident():
    # a 90/10 feature (gender) must not certify the target as imbalanced
    notebook = nb(
        [
            ("code", 'df["gender"].value_counts(normalize=True)'),
            ("code", SPLIT_ON_Y),
            ("code", "acc = accuracy_score(y_te, pred)"),
        ],
        outputs={0: VC_IMBALANCED},
    )
    flags = [f for f in run_static(notebook) if f.flaw_id == "baserate-accuracy-imbalanced"]
    assert flags and all(f.confidence < 0.8 for f in flags)


def test_baserate_balanced_feature_does_not_mask_imbalanced_target():
    notebook = nb(
        [
            ("code", 'df["gender"].value_counts(normalize=True)'),
            ("code", 'df["y"].value_counts(normalize=True)'),
            ("code", SPLIT_ON_Y),
            ("code", "acc = accuracy_score(y_te, pred)"),
        ],
        outputs={0: VC_BALANCED, 1: VC_IMBALANCED},
    )
    assert "baserate-accuracy-imbalanced" in flag_ids(notebook)


def test_baserate_plain_name_receiver_links_via_dataflow():
    notebook = nb(
        [
            ("code", 'y = df["target"]\ny.value_counts(normalize=True)'),
            ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
            ("code", "acc = accuracy_score(y_te, pred)"),
        ],
        outputs={0: VC_IMBALANCED},
    )
    assert "baserate-accuracy-imbalanced" in flag_ids(notebook)


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


def test_survivorship_skips_oversized_cell():
    from wald.dataflow import MAX_CELL_SOURCE_BYTES

    # an oversized cell is skipped by the dataflow caps; the survivorship
    # detector must skip it too. scanning its raw source is O(n^2) on a long
    # word-character run and hangs (regression: a 200 KB single-cell .py).
    payload = 'df = df.query("status == \'active\'")  # ' + "A" * (MAX_CELL_SOURCE_BYTES + 1)
    notebook = nb([
        ("code", payload),
        ("code", 'df.groupby("cohort")["ltv"].mean()'),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "selection-survivorship-cohort"]
    assert flags == []


def test_survivorship_candidate_flagged_on_query_idiom():
    notebook = nb([
        ("code", "df = df.query(\"status == 'active'\")"),
        ("code", 'df.groupby("cohort")["ltv"].mean()'),
    ])
    flags = run_static(notebook)
    match = [f for f in flags if f.flaw_id == "selection-survivorship-cohort"]
    assert match and match[0].confidence < 0.8


def test_survivorship_flagged_on_chained_query():
    notebook = nb([
        ("code", "df = df.query(\"cohort == '2023'\").query(\"status == 'active'\")"),
        ("code", 'df.groupby("cohort")["ltv"].mean()'),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "selection-survivorship-cohort"]
    assert flags and flags[0].extra["column"] == "status"


def test_survivorship_flagged_on_backtick_column():
    notebook = nb([
        ("code", 'df = df.query("`is_active` == 1")'),
        ("code", 'df.groupby("cohort")["ltv"].mean()'),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "selection-survivorship-cohort"]
    assert flags and flags[0].extra["column"] == "is_active"


def test_survivorship_fstring_query_skipped_silently():
    notebook = nb([
        ("code", "df = df.query(f\"status == '{val}'\")"),
        ("code", 'df.groupby("cohort")["ltv"].mean()'),
    ])
    assert not [
        f for f in run_static(notebook) if f.flaw_id == "selection-survivorship-cohort"
    ]


def test_survivorship_silent_on_query_non_risk_column():
    notebook = nb([
        ("code", "df = df.query(\"region == 'west'\")"),
        ("code", 'df.groupby("cohort")["ltv"].mean()'),
    ])
    assert not [
        f for f in run_static(notebook) if f.flaw_id == "selection-survivorship-cohort"
    ]


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


def test_leakage_flagged_one_line_constructor_fit():
    # the canonical one-liner: constructor-chained fit_transform on full data
    notebook = nb([
        ("code", "X = StandardScaler().fit_transform(X)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
    ])
    assert "leakage-fit-before-split" in flag_ids(notebook)


def test_leakage_silent_one_line_constructor_on_train_only():
    notebook = nb([
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
        ("code", "X_tr = StandardScaler().fit_transform(X_tr)"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


def test_leakage_silent_on_one_line_label_encoder():
    notebook = nb([
        ("code", "y = LabelEncoder().fit_transform(y)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


def test_leakage_factory_fit_stays_conservative():
    # clone()/make_pipeline() receivers resolve to the factory name, which is
    # not a known transformer class: a bare .fit must not flag
    notebook = nb([
        ("code", "clone(scaler).fit(X)"),
        ("code", "make_pipeline(StandardScaler(), clf).fit(X, y)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(X, y)"),
    ])
    assert "leakage-fit-before-split" not in flag_ids(notebook)


def test_leakage_cv_groups_kwarg_not_a_data_seed():
    notebook = nb([
        ("code", "groups = df['user_id']\ndf['s'] = scaler.fit_transform(df[cols])"),
        ("code", "scores = cross_val_score(clf, X_other, y_other, groups=groups)"),
    ])
    assert not [f for f in run_static(notebook) if f.flaw_id == "leakage-fit-before-split"]


# -- leakage-temporal-shuffle: FROZEN SPEC, detector not implemented yet --
# these 15 tests are expected to fail until the detector lands.


def test_temporal_flagged_lag_features_shuffled_split():
    notebook = nb([
        ("code", "df['date'] = pd.to_datetime(df['date'])"),
        ("code", "df['l1'] = df['y'].shift(1)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    assert "leakage-temporal-shuffle" in flag_ids(notebook, floor=0.8)


def test_temporal_flagged_rolling_chain():
    notebook = nb([
        ("code", "df['date'] = pd.to_datetime(df['date'])"),
        ("code", "df['roll'] = df['y'].rolling(14).mean()"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    assert "leakage-temporal-shuffle" in flag_ids(notebook, floor=0.8)


def test_temporal_silent_with_shuffle_false():
    notebook = nb([
        ("code", "df['date'] = pd.to_datetime(df['date'])"),
        ("code", "df['l1'] = df['y'].shift(1)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'], shuffle=False)"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert not flags


def test_temporal_silent_with_timeseries_split_cv():
    lag_src = "df['date'] = pd.to_datetime(df['date'])\ndf['l1'] = df['y'].shift(1)"

    tscv = nb([
        ("code", "from sklearn.model_selection import TimeSeriesSplit"),
        ("code", lag_src),
        ("code", "tscv = TimeSeriesSplit(5)"),
        ("code", "scores = cross_val_score(model, df[cols], df['y'], cv=tscv)"),
    ])
    flags = [f for f in run_static(tscv) if f.flaw_id == "leakage-temporal-shuffle"]
    assert not flags

    # the leftover TimeSeriesSplit import must not suppress a genuinely shuffled
    # KFold in a different variant: suppression is binding-based, not token-based
    kfold_shuffled = nb([
        ("code", "from sklearn.model_selection import TimeSeriesSplit"),
        ("code", lag_src),
        ("code", "kf = KFold(5, shuffle=True)"),
        ("code", "scores = cross_val_score(model, df[cols], df['y'], cv=kf)"),
    ])
    assert "leakage-temporal-shuffle" in flag_ids(kfold_shuffled)


def test_temporal_flagged_shuffled_kfold_cv():
    notebook = nb([
        ("code", "df['date'] = pd.to_datetime(df['date'])"),
        ("code", "df['l1'] = df['y'].shift(1)"),
        ("code", "kf = KFold(5, shuffle=True)"),
        ("code", "scores = cross_val_score(model, df[cols], df['y'], cv=kf)"),
    ])
    assert "leakage-temporal-shuffle" in flag_ids(notebook, floor=0.8)


def test_temporal_plain_cv_is_candidate():
    notebook = nb([
        ("code", "df['date'] = pd.to_datetime(df['date'])"),
        ("code", "df['l1'] = df['y'].shift(1)"),
        ("code", "scores = cross_val_score(model, df[cols], df['y'], cv=10)"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert flags and all(f.confidence < 0.8 for f in flags)


def test_temporal_silent_date_column_alone():
    # bike-sharing shape (corpus/real notebooks 21/28): datetime column and
    # categorical casts, shuffled split and CV, but no lag/window features
    notebook = nb([
        ("code", "df['dteday'] = pd.to_datetime(df['dteday'])"),
        ("code", "df['season'] = df['season'].astype('category')\n"
                  "df['weathersit'] = df['weathersit'].astype('category')"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['cnt'])"),
        ("code", "scores = cross_val_score(model, df[cols], df['cnt'], cv=5)"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert not flags


def test_temporal_silent_bare_scipy_shift():
    notebook = nb([
        ("code", "df['date'] = pd.to_datetime(df['date'])"),
        ("code", "from scipy.ndimage import shift"),
        ("code", "df['shifted'] = shift(df['y'].values, 1)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert not flags


def test_temporal_silent_sklearn_resample():
    notebook = nb([
        ("code", "df['date'] = pd.to_datetime(df['date'])"),
        ("code", "from sklearn.utils import resample"),
        ("code", "df_r = resample(df, n_samples=100)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df_r[cols], df_r['y'])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert not flags


def test_temporal_silent_np_diff():
    notebook = nb([
        ("code", "df['date'] = pd.to_datetime(df['date'])"),
        ("code", "df['d'] = np.diff(df['y'].values, prepend=0)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert not flags


def test_temporal_silent_lag_on_other_frame():
    notebook = nb([
        ("code", "ts_df['date'] = pd.to_datetime(ts_df['date'])\nts_df['l1'] = ts_df['y'].shift(1)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert not flags


def test_temporal_silent_after_frame_rebind():
    notebook = nb([
        ("code", "df['date'] = pd.to_datetime(df['date'])\ndf['l1'] = df['y'].shift(1)"),
        ("code", "df = pd.DataFrame(other_data)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert not flags


def test_temporal_datetime_index_no_lags_is_candidate():
    notebook = nb([
        ("code", "df = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert len(flags) == 1
    assert flags[0].confidence < 0.7


def test_temporal_silent_sort_values_non_date():
    notebook = nb([
        ("code", "corr = df.corr()\ncorr['v'] = corr['v'].sort_values()"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert not flags


def test_temporal_evidence_points_at_sink():
    notebook = nb([
        ("code", "df['date'] = pd.to_datetime(df['date'])"),
        ("code", "df['l1'] = df['y'].shift(1)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    flag = next(f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle")
    assert (flag.cell, flag.line) == (2, 1)
    assert "shift" in flag.evidence


def test_temporal_silent_value_sorted_a1_only_shuffled_split():
    # A1-only (an unused parsed date) on a frame sorted by a VALUE column: the
    # .diff is cross-sectional over score order, not a temporal lag — no leak
    notebook = nb([
        ("code", "df = df.sort_values('score')"),
        ("code", "df['signup_date'] = pd.to_datetime(df['signup_date'])"),
        ("code", "df['gap'] = df['score'].diff()"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert not flags


def test_temporal_silent_value_sorted_a1_only_shuffled_cv():
    # same root cause reaching the CV sink: income-sorted frame, unused dob date,
    # rolling feature, KFold(shuffle=True) must not fire
    notebook = nb([
        ("code", "df = df.sort_values('income')"),
        ("code", "df['dob'] = pd.to_datetime(df['dob'])"),
        ("code", "df['roll'] = df['income'].rolling(3).mean()"),
        ("code", "kf = KFold(5, shuffle=True)"),
        ("code", "scores = cross_val_score(model, df[cols], df['y'], cv=kf)"),
    ])
    flags = [f for f in run_static(notebook) if f.flaw_id == "leakage-temporal-shuffle"]
    assert not flags


def test_temporal_flagged_parse_dates_survives_value_sort():
    # a strong temporal signal (A2, parse_dates) is real time-order even when a
    # later non-date sort is present: suppression only gates the A1-only case
    notebook = nb([
        ("code", "df = pd.read_csv('d.csv', parse_dates=['date'])"),
        ("code", "df = df.sort_values('amount')"),
        ("code", "df['l1'] = df['y'].shift(1)"),
        ("code", "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['y'])"),
    ])
    assert "leakage-temporal-shuffle" in flag_ids(notebook, floor=0.8)
