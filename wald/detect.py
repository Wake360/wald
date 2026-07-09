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
# word-bounded: 'stockholm' or a variable 'fdr_level' must not read as a correction
CORRECTION_RE = re.compile(
    r"\b(?:multipletests|bonferroni|holm|sidak|fdr_bh|fdr_by|fdr_tsbh|fdr_tsbky)\b",
    re.IGNORECASE,
)
OTHER_METRIC_RE = re.compile(
    r"roc_auc|f1_score|precision_score|recall_score|balanced_accuracy"
    r"|average_precision|classification_report|confusion_matrix|log_loss"
)
SURVIVOR_COLUMNS = {"status", "active", "is_active", "churned", "retained", "completed", "survived"}
# retained for mutate.verify's applicability check; the detector itself no
# longer fabricates an iteration count for loops (see detect_multiple_testing)
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


def _join_names(names) -> str:
    ns = sorted(names)
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]


# -- leakage-temporal-shuffle constants --
DATETIME_MAKERS = {"to_datetime", "date_range", "DatetimeIndex"}
DATE_READERS = {"read_csv", "read_parquet", "read_sql"}
# a lag/window METHOD call: the '.' before the verb is load-bearing — it
# distinguishes df['y'].shift(1) (a series method, real lag) from a bare
# shift(image, ...)/np.diff(...) that merely shares the name
LAG_FUNC_RE = re.compile(r"\.(shift|rolling|resample|diff|pct_change|ewm|expanding|asfreq)\(")
# base segments that make the lag name a namespaced free function, not a
# frame method: scipy.ndimage.shift, np.diff, sklearn's utils are not lags
MODULE_ALIASES = {"np", "numpy", "pd", "pandas", "scipy", "sp", "tf", "torch"}
TEMPORAL_SORT_INDEX_RE = re.compile(
    r"(?:set_index|sort_values)\(\s*(?:by\s*=\s*)?['\"]\w*(?:date|datetime|timestamp|dteday)\w*['\"]",
    re.IGNORECASE,
)
# any set_index/sort_values on a LITERAL column, capturing the column name; a
# match whose column is NOT date-like means the frame's row order is a
# value ordering, not time — so an A1-only datetime column (an unused parsed
# date) does not make a .shift/.diff a temporal lag (see the value-sorted FPs)
SORT_INDEX_COLUMN_RE = re.compile(
    r"(?:set_index|sort_values)\(\s*(?:by\s*=\s*)?['\"](\w+)['\"]",
    re.IGNORECASE,
)
DATE_COLUMN_RE = re.compile(r"date|datetime|timestamp|dteday", re.IGNORECASE)
# splitters that shuffle unconditionally (KFold/StratifiedKFold shuffle only
# when shuffle=True, handled separately; TimeSeriesSplit never does)
SHUFFLED_SPLITTERS = {"ShuffleSplit", "StratifiedShuffleSplit"}
_TEMPORAL_SPLITTERS = SHUFFLED_SPLITTERS | {"KFold", "StratifiedKFold", "TimeSeriesSplit"}


def detect_leakage_temporal_shuffle(nb: ParsedNotebook, flow: NotebookDataflow) -> list[Flag]:
    sinks = [c for c in flow.calls if c.name == "train_test_split" or c.name in CV_SINKS]
    if not sinks:
        return []
    src = {c.index: c.source.splitlines() for c in nb.code_cells}

    def is_lag(ev) -> bool:
        if ev.call is None:
            return False
        if not LAG_FUNC_RE.search(ev.call.func + "("):
            return False
        base = re.split(r"[.\[]", ev.call.func, maxsplit=1)[0]
        return base not in MODULE_ALIASES

    def lag_label(ev) -> str:
        assert ev.call is not None
        m = LAG_FUNC_RE.search(ev.call.func + "(")
        assert m is not None
        return f"`.{m.group(1)}` at cell {ev.cell} line {ev.line}"

    def classify_splitter(ctor) -> str:
        if ctor.name == "TimeSeriesSplit":
            return "skip"
        if ctor.name in SHUFFLED_SPLITTERS:
            return "shuffled"
        if ctor.name in {"KFold", "StratifiedKFold"}:
            return "shuffled" if ctor.kw_args.get("shuffle") == {"True"} else "plain"
        return "plain"

    def cv_state(s) -> str:
        cv = s.kw_args.get("cv")
        if not cv:  # absent, or an int literal (no names) -> default plain CV
            return "plain"
        for name in cv:
            ev = flow.last_assign(name, (s.cell, s.line))
            if ev is not None and ev.call is not None:
                return classify_splitter(ev.call)
        for c in flow.calls:  # inline constructor: cv=KFold(5, shuffle=True)
            if (c.cell, c.line) == (s.cell, s.line) and c.name in _TEMPORAL_SPLITTERS:
                return classify_splitter(c)
        return "plain"

    def sink_state(s) -> str:
        if s.name == "train_test_split":
            shuffle = s.kw_args.get("shuffle")
            if shuffle is None or shuffle == {"True"}:
                return "shuffled"
            if shuffle == {"False"}:
                return "skip"
            return "skip"  # non-literal shuffle: treat as clean (FP discipline)
        return cv_state(s)

    flags: list[Flag] = []
    for s in sinks:
        if s.name in CV_SINKS:
            seed = set().union(
                set(), *s.pos_args[1:3],
                s.kw_args.get("X", set()), s.kw_args.get("y", set()),
            )
        else:
            seed = s.arg_names
        events, _bindings = flow.chain(seed, (s.cell, s.line))
        chain_text = "\n".join(
            src[ev.cell][ev.line - 1]
            for ev in events
            if ev.cell in src and 0 < ev.line <= len(src[ev.cell])
        )

        dt_events = [
            ev for ev in events
            if ev.call is not None and (
                ev.call.name in DATETIME_MAKERS
                or (ev.call.name in DATE_READERS and "parse_dates" in ev.call.kw_args)
            )
        ]
        a1 = any(ev.call is not None and ev.call.name in DATETIME_MAKERS for ev in dt_events)
        a2 = any(ev.call is not None and ev.call.name in DATE_READERS for ev in dt_events)
        a3 = bool(TEMPORAL_SORT_INDEX_RE.search(chain_text))
        a_any = a1 or a2 or a3
        a_strong = a2 or a3
        if not a_any:
            continue
        # A1 (a bare to_datetime column) is weak temporal identity: it can be an
        # unused parsed date on a frame whose rows are ordered by value. When the
        # only signal is A1 and the chain sorts on a non-date column, the frame is
        # value-sorted, not time-sorted, so a lag/window method is cross-sectional,
        # not a temporal lag — emit nothing (kills the value-sorted FPs).
        if a1 and not a_strong and any(
            not DATE_COLUMN_RE.search(m.group(1))
            for m in SORT_INDEX_COLUMN_RE.finditer(chain_text)
        ):
            continue

        lag_events = [ev for ev in events if is_lag(ev)]
        has_lag = bool(lag_events)
        state = sink_state(s)
        if state == "skip":
            continue

        if a_any and has_lag and state == "shuffled":
            confidence = 0.9
        elif a_any and has_lag and state == "plain":
            confidence = 0.75
        elif a_strong and not has_lag and s.name == "train_test_split" and state == "shuffled":
            confidence = 0.6
        else:
            continue

        dt_cell = dt_events[0].cell if dt_events else s.cell
        if has_lag:
            lag_desc = ", ".join(lag_label(ev) for ev in lag_events)
            evidence = (
                f"time-ordered frame (datetime origin in cell {dt_cell}) carries "
                f"lag/window features ({lag_desc}); `{s.name}` (cell {s.cell}) uses a "
                f"{'shuffled' if state == 'shuffled' else 'non-temporal'} split, "
                f"leaking future rows into training"
            )
        else:
            evidence = (
                f"time-ordered frame (datetime origin in cell {dt_cell}) is split by a "
                f"shuffled `train_test_split` (cell {s.cell}); no lag features detected, "
                f"but shuffling time-ordered rows risks leakage (candidate)"
            )
        flags.append(
            _make_flag(
                "leakage-temporal-shuffle",
                confidence=confidence,
                cell=s.cell,
                line=s.line,
                evidence=evidence,
                sink=s.name,
            )
        )
    return flags


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
        if call.receiver.endswith("()"):
            # direct-constructor receiver: StandardScaler().fit(X). Factory
            # calls (clone(x).fit, make_pipeline(...).fit) resolve to the
            # factory name, which is not a known transformer class — so the
            # known-class rule stays conservative for them.
            head = call.receiver[:-2]
            return head if head.isidentifier() else None
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
                        f"`{call.func}(...)` consumes {_join_names(leaked)}, "
                        f"feeding `train_test_split` (cell {sink.cell}) — "
                        f"the fit happened before the split"
                    ))
                else:
                    if not on_chain:
                        continue
                    # supervised selection on full data (fit(X, y)) is the
                    # serious variant; unsupervised pre-CV fits are common
                    # practice and stay below the confidence floor
                    supervised = len(call.pos_args) >= 2 or "y" in call.kw_args
                    emit(call, 0.9 if supervised else 0.75, leaked, (
                        f"`{call.func}(...)` is fitted on {_join_names(leaked)} whose "
                        f"transformed output feeds `{sink.name}` (cell {sink.cell}); "
                        f"every CV fold's test rows were in the transformer fit"
                        + (" (fitted with labels)" if supervised else "")
                    ), sink_cell=sink.cell)

        # -- imputation with statistics of the same frame, before the split --
        elif call.name in {"fillna", "replace"} and call.receiver is not None:
            base = call.receiver_base
            if base is None:
                continue
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
    n_static = len(sites)
    fwer = 1 - 0.95 ** n_static
    # FWER is only reported from the statically counted sites — never from a
    # fabricated per-loop iteration count
    over_threshold = n_static > MULTIPLE_TESTING_THRESHOLD
    param_vars = sorted(set().union(set(), *(c.arg_names & c.loop_vars for c in sites)))
    if param_vars:
        # loop variable reaches the test arguments: the loop enumerates
        # hypotheses, one per iteration
        confidence = 0.9
        evidence = (
            f"{n_static} test call site(s); at least one parameterized by a loop "
            f"over `{param_vars[0]}` — per-iteration hypotheses, iteration count "
            f"not statically known; no correction found"
            + (f"; FWER at alpha=0.05 from the static sites alone: {fwer:.0%}"
               if over_threshold else "")
        )
    elif over_threshold:
        confidence = 0.9
        evidence = (
            f"{n_static} test call site(s), no correction found; "
            f"FWER at alpha=0.05: {fwer:.0%}"
        )
    elif any(c.loop_depth > 0 for c in sites):
        # loop variable never reaches the test arguments: likely a
        # resampling/permutation loop over a single hypothesis
        confidence = 0.75
        evidence = (
            f"{n_static} test call site(s) inside a loop whose variable does not "
            f"reach the test arguments — likely resampling over one hypothesis "
            f"(candidate); no correction found"
        )
    else:
        return []
    extra: dict[str, int | float] = {"n_tests": n_static}
    if over_threshold:
        extra["fwer"] = round(fwer, 3)
    first = sites[0]
    return [
        _make_flag(
            "testing-multiple-uncorrected",
            confidence=confidence,
            cell=first.cell,
            line=first.line,
            evidence=evidence,
            **extra,
        )
    ]


_VC_COLUMN_RE = re.compile(r"\[['\"](\w+)['\"]\]|\.(\w+)$")


def _imbalance_from_outputs(nb: ParsedNotebook, flow: NotebookDataflow,
                            y_names: set[str], at: tuple[int, int]) -> float | None:
    """Majority share parsed from a value_counts stored output, but only when
    the counted series links to the scored target: a plain-name receiver whose
    binding is on y's dependency chain, or a column receiver whose literal
    column name appears in a y-chain assignment line. A skewed unrelated
    feature must not drive the base-rate verdict, and an unrelated balanced
    one must not clear it."""
    events, bindings = flow.chain(y_names, at)
    src = {c.index: c.source.splitlines() for c in nb.code_cells}
    chain_text = "\n".join(
        src[ev.cell][ev.line - 1]
        for ev in events
        if ev.cell in src and 0 < ev.line <= len(src[ev.cell])
    )
    outputs = {c.index: c.outputs_text for c in nb.code_cells}
    for vc in flow.calls:
        if vc.name != "value_counts" or vc.receiver is None:
            continue
        out = outputs.get(vc.cell, "")
        if not out:
            continue
        m = _VC_COLUMN_RE.search(vc.receiver)
        col = (m.group(1) or m.group(2)) if m else None
        if col is None:
            linked = flow.binding(vc.receiver, (vc.cell, vc.line)) in bindings
        else:
            linked = f"'{col}'" in chain_text or f'"{col}"' in chain_text
        if not linked:
            continue
        values = []
        for line in out.splitlines():
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
    first = acc_calls[0]
    y_names = first.pos_args[0] if first.pos_args else first.kw_args.get("y_true", set())
    majority = _imbalance_from_outputs(nb, flow, y_names, (first.cell, first.line))
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
# self-filter via .query, e.g. df = df.query("status == 'active'"), including
# chained .query calls; the column is the first (optionally backtick-quoted)
# identifier before the comparison operator. An f-string or non-literal query
# expression is unresolvable and skipped — no guessing.
SURVIVOR_QUERY_ASSIGN_RE = re.compile(r"(\w+)\s*=\s*\1\b[^\n]*")
SURVIVOR_QUERY_STRING_RE = re.compile(r"\.query\(\s*([A-Za-z]*)([\"'])(.*?)\2")
SURVIVOR_QUERY_COND_RE = re.compile(r"^\s*`?(\w+)`?\s*(?:==|!=)")


def _survivor_filter_columns(source: str):
    """Yield (match_start, risk_column) for every self-scoping cohort filter
    in a cell, across both the subscript and .query idioms."""
    for m in SURVIVOR_FILTER_RE.finditer(source):
        yield m.start(), (m.group(2) or m.group(3) or "").lower()
    for am in SURVIVOR_QUERY_ASSIGN_RE.finditer(source):
        for qm in SURVIVOR_QUERY_STRING_RE.finditer(am.group(0)):
            if "f" in qm.group(1).lower():
                continue  # f-string: the filter expression is dynamic
            cm = SURVIVOR_QUERY_COND_RE.match(qm.group(3))
            if cm:
                yield am.start() + qm.start(), cm.group(1).lower()


def detect_survivorship_candidate(nb: ParsedNotebook, flow: NotebookDataflow) -> list[Flag]:
    """Static half of the survivorship pair: a filter on a risk-vocabulary
    column followed by aggregation. Low confidence by design — promotion to
    a real flag requires the narrative layer confirming a population claim."""
    flags = []
    for cell in nb.code_cells:
        if cell.index in flow.skipped_cells:
            continue  # oversized/deep cell dataflow skipped: keep runtime bounded
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
    detect_leakage_temporal_shuffle,
    detect_multiple_testing,
    detect_baserate_accuracy,
    detect_survivorship_candidate,
]

# flaw classes the static layer can decide on its own (measured by gate G1)
STATIC_DECIDABLE = {
    "leakage-fit-before-split",
    "leakage-temporal-shuffle",
    "testing-multiple-uncorrected",
    "baserate-accuracy-imbalanced",
}


def run_static(nb: ParsedNotebook, flow: NotebookDataflow | None = None) -> list[Flag]:
    if flow is None:
        flow = analyze(nb)
    flags: list[Flag] = []
    for detector in DETECTORS:
        flags.extend(detector(nb, flow))
    return flags
