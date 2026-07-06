# leakage-temporal-shuffle: documented evasions

Four notebook shapes evade the detector (`wald/detect.py`,
`detect_leakage_temporal_shuffle`). Each was confirmed during the
adversarial round that shipped the class (commit 096606f) and ruled
residual risk: the root fix re-adds a measured false-positive class or
needs analysis machinery owned by a later milestone. Dispositions below.
Revisit trigger for all four: a confirmed real-world flaw in this shape
during the wide dogfood (`plans/v1-completion.md`, Static WS-A).

## 1. Variable `shuffle=` argument

Trigger sketch:

```python
do_shuffle = True
X_tr, X_te, y_tr, y_te = train_test_split(X, y, shuffle=do_shuffle)
```

`sink_state` treats a non-literal `shuffle` kwarg as `skip` (clean).
Resolving the variable would need literal-propagation through
`flow.last_assign`, and any unresolvable value (config dict, function
argument, CLI flag) would have to default somewhere: defaulting to
shuffled flags legitimate `shuffle=cfg.shuffle` set to False; defaulting
to clean is the current behavior. Wald's threat model is the honest
mistake, and honest notebooks pass shuffle as a literal — laundering the
flag through a variable is evasion, not error.

Decision: accept. Default-clean on unresolvable values is the FP
discipline that holds the 0.0% clean rate; the miss requires actively
hiding the shuffle from the linter.

## 2. Lag inside a transform lambda

Trigger sketch:

```python
df["lag"] = df.groupby("store")["sales"].transform(lambda s: s.shift(1))
X_tr, X_te, y_tr, y_te = train_test_split(X, y)  # shuffled default
```

`LAG_FUNC_RE` matches lag calls on tracked dataflow events; a `.shift`
inside a lambda body is not an event on the frame's chain, so the lag
signal is lost. Descending into lambda bodies would count every
groupby-transform as a lag carrier, and within-group transforms on
value-sorted frames were one of the four measured dogfood FP shapes.
Group-wise lags also need the group key's semantics (per-entity lags
with a random split over entities can be legitimate).

Decision: accept. Lambda-body analysis re-adds the exact FP class the
rebuild removed; group-aware lag semantics belong to a later detector
iteration with mutants of their own.

## 3. `.loc` slicing without a split sink

Trigger sketch:

```python
train = df.loc[df.index % 5 != 0]  # row-sampled "split", no sink call
test = df.loc[df.index % 5 == 0]
model.fit(train[feats], train.y)
```

The detector anchors on evaluation sinks (`train_test_split`, CV
functions). A manual split via boolean indexing never engages it. The
correct temporal protocol — a cutoff like `df.loc[df.date < "2024"]` —
is byte-for-byte this shape, so firing on sink-less slicing inverts the
signal; telling a random row-sample from a temporal cutoff requires
value-level reasoning about the mask, not syntax.

Decision: accept. A sink-less flag here would flag the fix we recommend.
The narrative layer, which reads the analyst's stated protocol, is the
owning surface if this shape ever shows up in confirmed real flaws.

## 4. CV splitter built in a helper

Trigger sketch:

```python
def make_cv():
    return KFold(5, shuffle=True)

scores = cross_val_score(pipe, X, y, cv=make_cv())
```

`cv_state` resolves the `cv=` binding to a constructor in the same
notebook scope; a splitter built inside a function body is invisible, so
the sink reads as `plain`. The miss is partial: with lag features
present, the plain-CV branch still emits at 0.75 — a below-floor
candidate rather than a confident flag. Full resolution is
function-scoped analysis, which NEXT.md assigns to M2/M3 (the same
decision as groupby-`apply` imputation and function-scoped selection).

Decision: accept. The degradation is to candidate, not silence, and
cross-function resolution is explicitly owned by a later milestone —
adding a special case here is the static-heuristic creep NEXT.md rules
out.
