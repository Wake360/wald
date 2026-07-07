# leakage-fit-before-split — known false positives

False-positive shapes for the fit-before-split detector found in real
notebooks, each with a trigger sketch and a disposition. Same format as
`temporal-shuffle-evasions.md`: `accept` means the FP is tolerated as the
price of catching the true class; `fix` means a root change is warranted.

## 1. RFECV scout-subsample split

Surfaced in the 2026-07-07 wide dogfood (1 of 60 notebooks), the only
confirmed FP in that batch.

```python
X_train_processed = pipeline.fit_transform(X_train)   # fit on the TRAIN split
# ...
X_scout, _, y_scout, _ = train_test_split(X_train, y_train, ...)  # scout subset
selector = RFECV(estimator, cv=StratifiedKFold(...))
selector.fit(X_scout, y_scout)                         # feature selection only
```

The detector links `pipeline.fit_transform` to the downstream
`train_test_split` and flags it as a full-data fit feeding an evaluation
split. But `X_train` is already the training split (the real train/test
split happened upstream), and the second `train_test_split` only carves a
*discarded* scout subsample (`_, _`) for RFECV feature selection — its
output never reaches an evaluation sink. No test rows are in the fit.

Decision: accept for 0.2.x. A precise fix must (a) recognize that the
`fit_transform` receiver is already a split product, and (b) tell an
evaluation split from a discard-half sampling split by whether the split
output flows to a scored `predict`. Both need reach analysis the static
layer does not carry today; adding a shallow heuristic (e.g. suppress when
the second split discards a half) risks re-opening true positives where a
scout split *is* later evaluated. The one-notebook cost does not justify
that risk. Revisit if a wider dogfood shows this idiom recurring.
