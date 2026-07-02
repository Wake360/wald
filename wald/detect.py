"""Static (layer A) detectors: deterministic, key-free, evidence-bearing.

Every flag carries an exact location and mechanical evidence. Confidence is
a property of the detection (static facts ~0.9+, heuristic candidates
below the default floor); severity is a property of the flaw class and
comes from the taxonomy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .dataflow import NotebookDataflow, analyze
from .ingest import ParsedNotebook
from .taxonomy import load_taxonomy

DEFAULT_CONFIDENCE_FLOOR = 0.8

TEST_FUNCS = {
    "ttest_ind", "ttest_rel", "ttest_1samp", "mannwhitneyu", "wilcoxon",
    "ks_2samp", "chi2_contingency", "fisher_exact", "pearsonr", "spearmanr",
    "kendalltau", "f_oneway", "kruskal", "binomtest", "proportions_ztest",
}
CORRECTION_RE = re.compile(r"multipletests|bonferroni|holm|sidak|fdr", re.IGNORECASE)
OTHER_METRIC_RE = re.compile(
    r"roc_auc|f1_score|precision_score|recall_score|balanced_accuracy"
    r"|average_precision|classification_report|confusion_matrix|log_loss"
)
SURVIVOR_COLUMNS = {"status", "active", "is_active", "churned", "retained", "completed", "survived"}
# how many tests a single call site inside a loop is assumed to run (documented heuristic)
LOOP_TEST_WEIGHT = 10
MULTIPLE_TESTING_THRESHOLD = 5
IMBALANCE_THRESHOLD = 0.8


@dataclass
class Flag:
    flaw_id: str
    severity: str
    confidence: float
    cell: int
    line: int
    evidence: str
    failure_scenario: str
    fix: str
    extra: dict = field(default_factory=dict)


def _make_flag(flaw_id: str, confidence: float, cell: int, line: int, evidence: str, **extra) -> Flag:
    d = load_taxonomy()[flaw_id]
    return Flag(
        flaw_id=flaw_id,
        severity=d.severity,
        confidence=confidence,
        cell=cell,
        line=line,
        evidence=evidence,
        failure_scenario=d.failure_scenario,
        fix=d.fix,
        extra=extra,
    )


def detect_leakage_fit_before_split(nb: ParsedNotebook, flow: NotebookDataflow) -> list[Flag]:
    splits = [c for c in flow.calls if c.name == "train_test_split"]
    if not splits:
        return []
    split_inputs: set[str] = set()
    for s in splits:
        split_inputs |= s.arg_names
    pre_split = flow.ancestors(split_inputs)

    flags = []
    for call in flow.calls:
        if call.name not in {"fit", "fit_transform"} or call.receiver is None:
            continue
        leaked = call.arg_names & pre_split
        if leaked:
            flags.append(
                _make_flag(
                    "leakage-fit-before-split",
                    confidence=0.92,
                    cell=call.cell,
                    line=call.line,
                    evidence=(
                        f"`{call.func}(...)` consumes {sorted(leaked)}, which feed "
                        f"`train_test_split` (cell {splits[0].cell}); the transformer "
                        f"is fitted on data containing the test set"
                    ),
                    leaked_names=sorted(leaked),
                )
            )
    return flags


def detect_multiple_testing(nb: ParsedNotebook, flow: NotebookDataflow) -> list[Flag]:
    sites = [c for c in flow.calls if c.name in TEST_FUNCS]
    if not sites:
        return []
    if CORRECTION_RE.search(nb.full_source()):
        return []
    effective_n = sum(LOOP_TEST_WEIGHT if c.loop_depth > 0 else 1 for c in sites)
    if effective_n <= MULTIPLE_TESTING_THRESHOLD:
        return []
    fwer = 1 - 0.95 ** effective_n
    looped = any(c.loop_depth > 0 for c in sites)
    first = sites[0]
    return [
        _make_flag(
            "testing-multiple-uncorrected",
            confidence=0.9,
            cell=first.cell,
            line=first.line,
            evidence=(
                f"{len(sites)} test call site(s)"
                + (" including tests inside a loop" if looped else "")
                + f"; estimated >= {effective_n} tests, no correction found; "
                f"FWER at alpha=0.05: {fwer:.0%}"
            ),
            n_tests=effective_n,
            fwer=round(fwer, 3),
        )
    ]


def _imbalance_from_outputs(nb: ParsedNotebook) -> float | None:
    """Parse value_counts-style stored outputs; return majority share if found."""
    for cell in nb.code_cells:
        if "value_counts" not in cell.source or not cell.outputs_text:
            continue
        values = []
        for line in cell.outputs_text.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    values.append(float(parts[-1]))
                except ValueError:
                    continue
        if len(values) >= 2 and sum(values) > 0:
            return max(values) / sum(values)
    return None


def detect_baserate_accuracy(nb: ParsedNotebook, flow: NotebookDataflow) -> list[Flag]:
    acc_calls = [c for c in flow.calls if c.name == "accuracy_score"]
    if not acc_calls:
        return []
    if OTHER_METRIC_RE.search(nb.full_source()):
        return []
    majority = _imbalance_from_outputs(nb)
    first = acc_calls[0]
    if majority is not None and majority < IMBALANCE_THRESHOLD:
        return []  # accuracy-only is defensible on balanced classes
    if majority is None:
        confidence, note = 0.55, "class balance not visible in outputs (candidate)"
    else:
        confidence, note = 0.9, f"majority class share {majority:.0%} visible in value_counts output"
    return [
        _make_flag(
            "baserate-accuracy-imbalanced",
            confidence=confidence,
            cell=first.cell,
            line=first.line,
            evidence=f"accuracy is the only classification metric; {note}",
            majority_share=majority,
        )
    ]


SURVIVOR_FILTER_RE = re.compile(
    r"(\w+)\s*=\s*\1\[\s*\1(?:\.(\w+)|\[\s*[\"'](\w+)[\"']\s*\])\s*(?:==|!=)",
)


def detect_survivorship_candidate(nb: ParsedNotebook, flow: NotebookDataflow) -> list[Flag]:
    """Static half of the survivorship pair: a filter on a risk-vocabulary
    column followed by aggregation. Low confidence by design — promotion to
    a real flag requires the narrative layer confirming a population claim."""
    flags = []
    for cell in nb.code_cells:
        for m in SURVIVOR_FILTER_RE.finditer(cell.source):
            column = (m.group(2) or m.group(3) or "").lower()
            if column not in SURVIVOR_COLUMNS:
                continue
            aggregates_later = any(
                c.name in {"groupby", "mean", "agg", "sum", "median", "pivot_table"}
                and c.cell >= cell.index
                for c in flow.calls
            )
            if not aggregates_later:
                continue
            line = cell.source[: m.start()].count("\n") + 1
            flags.append(
                _make_flag(
                    "selection-survivorship-cohort",
                    confidence=0.55,
                    cell=cell.index,
                    line=line,
                    evidence=(
                        f"cohort filtered on `{column}` (a survival-correlated column) "
                        f"and aggregated afterwards; whether the conclusion is scoped "
                        f"to survivors needs the narrative layer"
                    ),
                    column=column,
                )
            )
    return flags


DETECTORS = [
    detect_leakage_fit_before_split,
    detect_multiple_testing,
    detect_baserate_accuracy,
    detect_survivorship_candidate,
]

# flaw classes the static layer can decide on its own (measured by gate G1)
STATIC_DECIDABLE = {
    "leakage-fit-before-split",
    "testing-multiple-uncorrected",
    "baserate-accuracy-imbalanced",
}


def run_static(nb: ParsedNotebook) -> list[Flag]:
    flow = analyze(nb)
    flags: list[Flag] = []
    for detector in DETECTORS:
        flags.extend(detector(nb, flow))
    return flags
