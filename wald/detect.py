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


# classes whose .fit learns statistics that leak when computed on full data;
# estimators (.fit-only models) are a different flaw class and are not v1
TRANSFORMER_CLASSES = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer",
    "QuantileTransformer", "PowerTransformer", "PolynomialFeatures",
    "PCA", "KernelPCA", "TruncatedSVD", "NMF",
    "SimpleImputer", "KNNImputer", "IterativeImputer",
    "OneHotEncoder", "OrdinalEncoder",
    "CountVectorizer", "TfidfVectorizer", "TfidfTransformer",
    "SelectKBest", "SelectPercentile", "RFE", "VarianceThreshold",
}
# fitting a label encoding on full y is standard practice, not leakage
NON_LEAKY_TRANSFORMERS = {"LabelEncoder", "LabelBinarizer"}


# cross-validation calls are evaluation sinks too: a transformer fitted on
# the full data whose output feeds CV has seen every fold's test rows
CV_SINKS = {"cross_val_score", "cross_val_predict", "cross_validate"}


def detect_leakage_fit_before_split(nb: ParsedNotebook, flow: NotebookDataflow) -> list[Flag]:
    sinks = [c for c in flow.calls if c.name == "train_test_split" or c.name in CV_SINKS]
    if not sinks:
        return []

    # names assigned from a train_test_split call (X_train, X_test, ...)
    split_outputs: set[str] = set()
    for ev in flow.assigns:
        if ev.call is not None and ev.call.name == "train_test_split":
            split_outputs |= ev.targets
    # every .transform call, for binding-aware receiver checks
    transform_calls = [c for c in flow.calls if c.name == "transform" and c.receiver]
    assign_event_of: dict[int, "object"] = {
        id(ev.call): ev for ev in flow.assigns if ev.call is not None
    }

    def recv_class(call) -> str | None:
        """Class the receiver was constructed from, resolved at the call's
        position (a later rebind of the name must not rewrite history)."""
        ev = flow.last_assign(call.receiver, (call.cell, call.line))
        return ev.call.name if ev is not None and ev.call is not None else None

    def same_receiver(a_call, b_call) -> bool:
        """Both calls act on the same object: same receiver name bound to
        the same event at each site (name reuse across sections differs)."""
        return (
            a_call.receiver == b_call.receiver
            and flow.binding(a_call.receiver, (a_call.cell, a_call.line))
            == flow.binding(b_call.receiver, (b_call.cell, b_call.line))
        )

    # flow-sensitive dependency chain per sink (kill-on-reassign: name reuse
    # across notebook sections no longer links unrelated datasets)
    chains = []
    for s in sinks:
        if s.name in CV_SINKS:
            # only the data args (X, y): the estimator is cloned and refit per
            # fold, and cv=/groups=/scoring= carry no transformed data
            seed = set().union(
                set(), *s.pos_args[1:3],
                s.kw_args.get("X", set()), s.kw_args.get("y", set()),
            )
        else:
            seed = s.arg_names
        events, bindings = flow.chain(seed, (s.cell, s.line))
        chains.append((s, {id(e) for e in events}, bindings))

    best: dict[object, Flag] = {}

    def emit(call, confidence, leaked, evidence, key=None, sink_cell=None):
        key = key if key is not None else id(call)
        if key in best and best[key].confidence >= confidence:
            return
        extra = {"leaked_names": sorted(leaked)}
        if sink_cell is not None:
            extra["sink_cell"] = sink_cell  # CV sink cell, read by the pre-CV fusion rule
        best[key] = _make_flag(
            "leakage-fit-before-split",
            confidence=confidence,
            cell=call.cell,
            line=call.line,
            evidence=evidence,
            **extra,
        )

    for call in flow.calls:
        pos = (call.cell, call.line)

        # -- transformer fitted on data that reaches an evaluation sink --
        if call.name in {"fit", "fit_transform"} and call.receiver is not None:
            if "[?]" in call.receiver:
                continue  # dynamic container element: object identity unknown
            cls = recv_class(call)
            if cls in NON_LEAKY_TRANSFORMERS:
                continue
            is_transformer = (
                call.name == "fit_transform"
                or cls in TRANSFORMER_CLASSES
                or any(same_receiver(call, t) for t in transform_calls)
            )
            if not is_transformer:
                continue  # estimator fit: not this flaw class
            my_event = assign_event_of.get(id(call))
            arg_binds = {flow.binding(n, pos) for n in call.arg_names}
            recv_bind = flow.binding(call.receiver, pos)
            transforms_split_part = any(
                same_receiver(call, t) and t.arg_names & split_outputs
                for t in transform_calls
            )
            for sink, chain_ids, bindings in chains:
                leaked = {name for name, ev_id in arg_binds if (name, ev_id) in bindings}
                if not leaked:
                    continue
                # the fitted data reaches the sink either through this call's
                # assigned result, or through the receiver's .transform output
                # (a bare `skb.fit(X, y)` produces no assign event)
                on_chain = (my_event is not None and id(my_event) in chain_ids) or recv_bind in bindings
                if sink.name == "train_test_split":
                    # (a) its output feeds the split, or (b) the fitted
                    # receiver later transforms a split output
                    if not (on_chain or transforms_split_part):
                        continue
                    emit(call, 0.92, leaked, (
                        f"`{call.func}(...)` consumes {sorted(leaked)}, which feed "
                        f"`train_test_split` (cell {sink.cell}); the transformer "
                        f"is fitted on data containing the test set"
                    ))
                else:
                    if not on_chain:
                        continue
                    # supervised selection on full data (fit(X, y)) is the
                    # serious variant; unsupervised pre-CV fits are common
                    # practice and stay below the confidence floor
                    supervised = len(call.pos_args) >= 2 or "y" in call.kw_args
                    emit(call, 0.9 if supervised else 0.75, leaked, (
                        f"`{call.func}(...)` is fitted on {sorted(leaked)} whose "
                        f"transformed output feeds `{sink.name}` (cell {sink.cell}); "
                        f"every CV fold's test rows were in the transformer fit"
                        + (" (fitted with labels)" if supervised else "")
                    ), sink_cell=sink.cell)

        # -- imputation with statistics of the same frame, before the split --
        elif call.name in {"fillna", "replace"} and call.receiver is not None:
            base = call.receiver_base
            base_bind = flow.binding(base, pos)
            _, arg_bindings = flow.chain(call.arg_names, pos)
            if base_bind not in arg_bindings:
                continue  # fill values do not derive from the frame itself
            for sink, _ids, bindings in chains:
                if sink.name != "train_test_split":
                    continue
                if base_bind in bindings and pos < (sink.cell, sink.line):
                    # one flag per frame: imputing N columns is one violation
                    emit(call, 0.85, {base}, (
                        f"`{call.func}(...)` fills values of `{base}` with statistics "
                        f"computed on the full `{base}` before `train_test_split` "
                        f"(cell {sink.cell}); imputation statistics include the test rows"
                    ), key=f"impute:{base}")
                    break

    return list(best.values())


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
# self-filter via .query, e.g. df = df.query("status == 'active'"); the
# quoted column is the first identifier before the comparison operator
SURVIVOR_QUERY_RE = re.compile(
    r"(\w+)\s*=\s*\1\.query\(\s*[\"']\s*(\w+)\s*(?:==|!=)",
)


def _survivor_filter_columns(source: str):
    """Yield (match_start, risk_column) for every self-scoping cohort filter
    in a cell, across both the subscript and .query idioms."""
    for m in SURVIVOR_FILTER_RE.finditer(source):
        yield m.start(), (m.group(2) or m.group(3) or "").lower()
    for m in SURVIVOR_QUERY_RE.finditer(source):
        yield m.start(), m.group(2).lower()


def detect_survivorship_candidate(nb: ParsedNotebook, flow: NotebookDataflow) -> list[Flag]:
    """Static half of the survivorship pair: a filter on a risk-vocabulary
    column followed by aggregation. Low confidence by design — promotion to
    a real flag requires the narrative layer confirming a population claim."""
    flags = []
    for cell in nb.code_cells:
        for start, column in _survivor_filter_columns(cell.source):
            if column not in SURVIVOR_COLUMNS:
                continue
            aggregates_later = any(
                c.name in {"groupby", "mean", "agg", "sum", "median", "pivot_table"}
                and c.cell >= cell.index
                for c in flow.calls
            )
            if not aggregates_later:
                continue
            line = cell.source[:start].count("\n") + 1
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
