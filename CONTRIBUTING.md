# Contributing to Wald

## Clean-corpus criteria

A notebook belongs in the clean corpus only if it satisfies all six:

1. **Scoped claims** — conclusions state the population/sample they apply to, not an unqualified generalization.
2. **Split before fit** — any transformer (scaler, imputer, encoder, feature selector, PCA, vectorizer) is fit only on the training split.
3. **Corrected or ≤3 tests** — multiple-hypothesis tests are either corrected (Bonferroni, BH-FDR, etc.) or there are 3 or fewer.
4. **≥2 classification metrics** — a classifier is judged on at least two of precision/recall/F1/AUC, not accuracy alone.
5. **Imbalance stated** — if classes are imbalanced, the notebook says so before quoting accuracy.
6. **No extrapolation** — no prediction or claim outside the range of the observed data.

A notebook that fails any one of these is a mutation target, not a clean example.

## Contributing a real notebook to the corpus

Real notebooks live in `corpus/real/` and are tracked in `corpus/real/MANIFEST.json`, one entry per file:

```json
{
  "file": "real/00__ageron_handson-ml3__03_classification.ipynb",
  "repo": "ageron/handson-ml3",
  "license": "Apache-2.0",
  "source_path": "03_classification.ipynb",
  "review": "2026-07-04 dogfood (evals/2026-07-04-dogfood.md): hand-reviewed clean",
  "note": "image outputs stripped for size; text outputs preserved"
}
```

Licensing rule: the source repository must be Apache-2.0 or MIT. The contributor supplies the repo URL and the exact `source_path` within that repo; both are recorded in the manifest entry as above. A notebook without a recorded license and source path will not be accepted.

The notebook must also pass the clean-corpus criteria above and a manual review before it's added.

## Contributing a flaw class

A new flaw class needs all three parts:

1. **Taxonomy record** — an entry in `wald/taxonomy/flaws.yaml`: `id`, `class`, `layer` (`static` or `static+narrative`), `severity`, `definition`, `book_anchor` (the statistics/ML reference that names the error), `failure_scenario`, `fix`. Detector code and generated prompts/reports read from this file; flaw semantics are never hardcoded elsewhere.
2. **Detector** — a function in `wald/detect.py` (static layer) and/or `wald/narrative.py` (LLM layer) that emits a `Flag` for the flaw, registered in `DETECTORS`.
3. **Mutation recipe** — a `Mutation` subclass in `wald/mutate.py` implementing `applicable`, `apply`, and `verify`. `verify` must mechanically prove the injected flaw is present (e.g. executing the mutated notebook and checking a fitted transformer's `n_samples_seen_`), not just assert the code pattern was inserted.

All four are required together: a detector without a mutation can't be measured, and a mutation without a book anchor isn't a cited flaw.
