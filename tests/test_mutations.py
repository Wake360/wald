"""Mutation structure tests — apply() without kernel execution.
Full verify() (execution fingerprints) is covered by corpus build + gate G0."""

from wald.corpus import FAMILIES, churn_notebook, cohort_notebook, fraud_notebook
from wald.mutate import (
    BaserateAccuracyMutation,
    FitBeforeSplitMutation,
    MultipleTestingMutation,
    MUTATIONS,
    SurvivorshipMutation,
)


def test_fit_before_split_reorders_statements():
    clean = churn_notebook(7)
    m = FitBeforeSplitMutation()
    assert m.applicable(clean)
    mutant = m.apply(clean, 0)
    lines = mutant.cells[5]["source"].splitlines()
    fit_line = next(i for i, l in enumerate(lines) if "fit_transform" in l)
    split_line = next(i for i, l in enumerate(lines) if "train_test_split" in l)
    assert fit_line < split_line
    assert "X = scaler.fit_transform(X)" in lines[fit_line]
    assert not any(".transform(X_te)" in l for l in lines)


def test_multiple_testing_inserts_uncorrected_loop():
    clean = churn_notebook(7)
    m = MultipleTestingMutation()
    assert m.applicable(clean)
    mutant = m.apply(clean, 0)
    src = "\n".join(c["source"] for c in mutant.cells if c["cell_type"] == "code")
    assert "ttest_ind" in src and "for c in cols_to_test" in src
    assert "multipletests" not in src


def test_multiple_testing_seed_changes_columns():
    clean = churn_notebook(7)
    m = MultipleTestingMutation()
    s0 = "\n".join(c["source"] for c in m.apply(clean, 0).cells if c["cell_type"] == "code")
    s1 = "\n".join(c["source"] for c in m.apply(clean, 1).cells if c["cell_type"] == "code")
    assert s0 != s1


def test_baserate_strips_all_auc_mentions():
    clean = fraud_notebook(7)
    m = BaserateAccuracyMutation()
    assert m.applicable(clean)
    mutant = m.apply(clean, 0)
    src = "\n".join(c["source"] for c in mutant.cells if c["cell_type"] == "code")
    assert "roc_auc" not in src and "predict_proba" not in src
    assert "accuracy_score" in src


def test_baserate_not_applicable_on_balanced_family():
    assert not BaserateAccuracyMutation().applicable(churn_notebook(7))


def test_survivorship_filter_and_population_claim():
    clean = cohort_notebook(7)
    m = SurvivorshipMutation()
    assert m.applicable(clean)
    for seed in (0, 1):
        mutant = m.apply(clean, seed)
        src = "\n".join(c["source"] for c in mutant.cells if c["cell_type"] == "code")
        assert 'df = df[df["status"] == "active"]' in src
        md = "\n".join(c["source"] for c in mutant.cells if c["cell_type"] == "markdown")
        assert "customer value over time" in md


def test_every_family_has_at_least_one_applicable_mutation():
    for family, builder in FAMILIES.items():
        clean = builder(7)
        assert any(m.applicable(clean) for m in MUTATIONS), family
