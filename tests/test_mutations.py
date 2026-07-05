"""Mutation structure tests — apply() without kernel execution.
Full verify() (execution fingerprints) is covered by corpus build + gate G0,
except the one execution-based verify test below that pins the three
regression-to-mean selection idioms."""

import re

import pytest

from wald.corpus import (
    FAMILIES,
    abtest_notebook,
    churn_notebook,
    cohort_notebook,
    forecast_notebook,
    fraud_notebook,
    program_notebook,
)
from wald.detect import DEFAULT_CONFIDENCE_FLOOR, run_static
from wald.ingest import from_nbnode
from wald.mutate import (
    BaserateAccuracyMutation,
    FitBeforeSplitMutation,
    MultipleTestingMutation,
    MUTATIONS,
    RegressionToMeanMutation,
    SignificanceMeaninglessMutation,
    SurvivorshipMutation,
    TemporalShuffleMutation,
    phrasing_variant,
)


def code_of(nb):
    return "\n".join(c["source"] for c in nb.cells if c["cell_type"] == "code")


def md_of(nb):
    return "\n".join(c["source"] for c in nb.cells if c["cell_type"] == "markdown")


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
    src = code_of(mutant)
    assert "ttest_ind" in src and "for c in cols_to_test" in src
    assert "multipletests" not in src


def test_multiple_testing_seed_changes_columns():
    clean = churn_notebook(7)
    m = MultipleTestingMutation()
    s0 = code_of(m.apply(clean, 0))
    s1 = code_of(m.apply(clean, 1))
    assert s0 != s1


def test_baserate_strips_all_auc_mentions():
    clean = fraud_notebook(7)
    m = BaserateAccuracyMutation()
    assert m.applicable(clean)
    mutant = m.apply(clean, 0)
    src = code_of(mutant)
    assert "roc_auc" not in src and "predict_proba" not in src
    assert "accuracy_score" in src


def test_baserate_not_applicable_on_balanced_family():
    assert not BaserateAccuracyMutation().applicable(churn_notebook(7))


def test_phrasing_variant_split_partition():
    dev = {phrasing_variant(s, b) for s in (0, 1) for b in range(30)}
    heldout = {phrasing_variant(s, b) for s in (2, 3) for b in range(30)}
    assert dev == {0, 1}
    assert heldout == {2, 3, 4}


def test_survivorship_filter_idioms_and_phrasing_pool():
    clean = cohort_notebook(7)
    m = SurvivorshipMutation()
    assert m.applicable(clean)
    for seed in (0, 1, 2, 3):
        mutant = m.apply(clean, seed)
        assert m.FILTERS[seed % 2] in code_of(mutant)
        md = md_of(mutant)
        assert m.conclusion(clean, seed) in md
        assert not re.search(r"active|retained|surviv|complet", md, re.IGNORECASE)


def test_significance_inflates_n_and_keeps_effect_size_in_code():
    clean = abtest_notebook(7)
    m = SignificanceMeaninglessMutation()
    assert m.applicable(clean)
    for seed in (0, 2):
        mutant = m.apply(clean, seed)
        datagen = mutant.cells[2]["source"]
        assert "n = 40000" in datagen and "* 0.17" in datagen
        assert "pooled_sd" in code_of(mutant)  # d stays; the prose ignores it
        md = md_of(mutant)
        assert m.conclusion(clean, seed) in md
        assert "effect size" not in md.lower()


def test_regression_to_mean_varies_selection_and_has_no_control():
    clean = churn_notebook(7)
    m = RegressionToMeanMutation()
    assert m.applicable(clean)
    assert not m.applicable(abtest_notebook(7))
    selectors = set()
    for seed in (0, 1, 2):
        mutant = m.apply(clean, seed)
        src = code_of(mutant)
        selectors |= {s for s in ("nsmallest", "nlargest", "quantile") if s in src}
        assert "control" not in src.lower()
        md = md_of(mutant)
        assert m.conclusion(clean, seed) in md
        assert re.search(r"caused|effect of|drove|produced|worked", md, re.IGNORECASE)
        assert not re.search(r"control|comparison group", md, re.IGNORECASE)
    assert selectors == {"nsmallest", "nlargest", "quantile"}


@pytest.mark.parametrize("seed,selector", [
    (0, "nsmallest"), (1, "nlargest"), (2, "quantile"),
])
def test_regression_to_mean_verify_each_selection_idiom(seed, selector):
    # seed % 3 picks the selection idiom; execute apply()+verify() so each
    # of the three branches is actually run and its fingerprint asserted
    clean = churn_notebook(7)
    m = RegressionToMeanMutation()
    mutant = m.apply(clean, seed)
    assert selector in code_of(mutant)
    ok, evidence = m.verify(mutant)
    assert ok, evidence
    assert evidence["causal_claim"] and not evidence["control_language"]


def test_churn_second_period_column_is_static_silent():
    clean = churn_notebook(7)
    assert "monthly_spend_q2" in clean.cells[2]["source"]
    flags = run_static(from_nbnode(clean))
    assert [f for f in flags if f.confidence >= DEFAULT_CONFIDENCE_FLOOR] == []


def test_program_family_is_static_silent():
    flags = run_static(from_nbnode(program_notebook(7)))
    assert [f for f in flags if f.confidence >= DEFAULT_CONFIDENCE_FLOOR] == []


def test_every_family_has_at_least_one_applicable_mutation():
    for family, builder in FAMILIES.items():
        clean = builder(7)
        assert any(m.applicable(clean) for m in MUTATIONS), family


def test_normalize_strips_execution_and_is_serialization_stable():
    # T15: deterministic cell ids + no transient execution timestamps, so a
    # second build of an unchanged notebook is byte-identical
    import nbformat

    from wald.corpus import _normalize

    nb = churn_notebook(7)
    nb.cells[1]["metadata"]["execution"] = {"iopub.status.busy": "2026-01-01T00:00:00Z"}
    _normalize(nb, "churn-s7")
    assert nb.cells[1]["id"] == "churn-s7-1"
    assert "execution" not in nb.cells[1]["metadata"]
    other = _normalize(churn_notebook(7), "churn-s7")  # fresh random ids pre-normalize
    assert nbformat.writes(nb) == nbformat.writes(other)


def test_temporal_mutation_applicable_only_on_forecast_family():
    m = TemporalShuffleMutation()
    assert m.applicable(forecast_notebook(7))
    for family, builder in FAMILIES.items():
        if family == "forecast":
            continue
        assert not m.applicable(builder(7)), family


def test_temporal_split_rewrite_drops_shuffle_false():
    clean = forecast_notebook(7)
    m = TemporalShuffleMutation()
    mutant = m.apply(clean, 0)
    src = mutant.cells[4]["source"]
    assert "shuffle=False" not in src
    assert "random_state=40" in src


def test_temporal_cv_rewrite_installs_shuffled_kfold():
    clean = forecast_notebook(7)
    m = TemporalShuffleMutation()
    mutant = m.apply(clean, 1)
    cv_src = mutant.cells[5]["source"]
    assert "KFold" in cv_src
    assert "cv = TimeSeriesSplit" not in cv_src
    assert mutant.cells[1]["source"] == clean.cells[1]["source"]


@pytest.mark.parametrize("seed", [0, 1])
def test_temporal_verify_each_variant(seed):
    clean = forecast_notebook(7)
    m = TemporalShuffleMutation()
    mutant = m.apply(clean, seed)
    ok, evidence = m.verify(mutant)
    assert ok, evidence
    assert "train_max_date" in evidence and "test_min_date" in evidence


def test_forecast_family_is_static_silent():
    flags = run_static(from_nbnode(forecast_notebook(7)))
    assert flags == []


@pytest.mark.parametrize("seed", [0, 1])
def test_forecast_mutants_are_confidently_flagged(seed):
    clean = forecast_notebook(7)
    m = TemporalShuffleMutation()
    mutant = m.apply(clean, seed)
    ok, evidence = m.verify(mutant)
    assert ok, evidence
    flags = run_static(from_nbnode(mutant))
    assert any(
        f.flaw_id == "leakage-temporal-shuffle" and f.confidence >= DEFAULT_CONFIDENCE_FLOOR
        for f in flags
    ), flags


def test_render_report_survives_zero_clean_notebooks():
    from wald.eval import render_report

    results = {
        "date": "2026-01-01", "corpus_built": "2026-01-01",
        "n_clean": 0, "n_clean_real": 0, "n_mutants": 1, "n_discarded": 0,
        "confidence_floor": 0.8, "static_classes": {}, "candidate_classes": {},
        "clean_fp_rate": None, "clean_fp_files": [], "missed_mutants": [],
    }
    out = render_report(results)  # must not raise on None clean_fp_rate
    assert "no clean notebooks" in out


def test_evaluate_missing_corpus_raises_systemexit(tmp_path):
    from wald.eval import evaluate

    with pytest.raises(SystemExit):
        evaluate(tmp_path / "nonexistent-corpus")
