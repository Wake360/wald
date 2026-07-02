# Wald

**Code linters check style. Wald checks whether the analysis is a lie.**

Wald is a statistical-integrity linter for data-analysis notebooks. It reads
code, stored outputs and prose, and flags the finite, catalogued classes of
error that invalidate an analysis: train/test leakage, uncorrected multiple
testing, survivorship bias, accuracy-on-imbalanced-classes and friends.
Every flag carries an exact location, mechanical evidence, a failure
scenario, and a fix.

Named after Abraham Wald, who armored the bullet holes that *weren't* in the
returning planes. The tool's mentality is his: don't ask what the data says —
ask what data is missing and what that does to the conclusion.

```
$ wald check churn_analysis.ipynb

# Wald report — churn_analysis.ipynb
verdict: 1 high, 0 medium | static layer (no LLM)

## HIGH: leakage-fit-before-split
- **Where:** cell 5, line 2
- **Evidence:** `scaler.fit_transform(...)` consumes ['X'], which feed
  `train_test_split` (cell 5); the transformer is fitted on data containing
  the test set
- **Failure scenario:** Test metrics are inflated; the production model will
  underperform the reported numbers.
- **Fix:** Fit the transformer on the training split only; apply transform
  to the test split.
- **Confidence:** 0.92

## CLEAN (checked): baserate-accuracy-imbalanced, testing-multiple-uncorrected
```

Exit codes: `0` clean, `1` medium findings, `2` high findings — usable as a
pre-merge gate.

## How Wald proves it works: mutation testing for statistics

The detector is not measured by demos. A **mutation corpus** exists before
the detectors do: clean notebooks get named flaws programmatically injected
(via libcst, format-preserving), and every mutant must *mechanically prove*
the flaw is present before it enters the corpus — e.g. a leakage mutant is
executed and the scaler's `n_samples_seen_` must equal the full row count
(it saw the test set), not the train count.

Detection quality is then a confusion matrix per flaw class, not an opinion:

| class (static layer) | mutants | TP | FN | FP | precision | recall |
|---|---|---|---|---|---|---|
| leakage-fit-before-split | 12 | 12 | 0 | 0 | 1.00 | 1.00 |
| testing-multiple-uncorrected | 40 | 40 | 0 | 0 | 1.00 | 1.00 |
| baserate-accuracy-imbalanced | 4 | 4 | 0 | 0 | 1.00 | 1.00 |
| selection-survivorship-cohort (candidate) | 8 | 8 | 0 | — | — | 1.00 |

False-positive rate on the 20-notebook clean corpus: **0.0%**. 64/64
mutants passed mechanical verification at build; 0 discarded.
(Eval 2026-07-02, `evals/2026-07-02-eval.md`.)

Reproduce: `wald corpus build && wald eval`. Dated reports live in `evals/`.

**Honest caveat:** the v1 corpus is synthetic and stereotypical by design.
These numbers measure detector correctness on canonical pandas/sklearn
idioms, not real-world recall. Dogfooding on real notebooks is the next
milestone, and the corpus format accepts licensed real notebooks.

## What Wald sees (v1) and what it doesn't

The static layer (deterministic, no API key, runs in CI) decides three
classes on its own:

- `leakage-fit-before-split` — def-use dataflow: a transformer's
  `.fit`/`.fit_transform` consumes an ancestor of the `train_test_split`
  inputs.
- `testing-multiple-uncorrected` — counts statistical test call sites
  (loops weighted), checks for corrections, reports the implied FWER.
- `baserate-accuracy-imbalanced` — accuracy as the only metric while class
  imbalance is visible in stored `value_counts` outputs.

It additionally emits a low-confidence **candidate** for
`selection-survivorship-cohort` (a filter on a survival-correlated column
followed by aggregation). Deciding that class requires the narrative layer
— the flaw is the *pair* (filter, population claim); the filter alone is
legitimate analysis.

**What Wald does not see:** flaws with no imprint in code, outputs or
prose. Survivorship bias that lives in data that never entered the notebook
is invisible to any tool that reads notebooks. Wald claims only the
detectable classes, and the taxonomy (`wald/taxonomy/flaws.yaml`) is the
complete, versioned list of what it checks.

## Architecture

```
notebook ──> ingest (nbformat) ──> layer A: static detectors ──> report
                                    - libcst def-use dataflow      md / json
                                    - deterministic, key-free      exit code
                                    - exact cell:line + evidence

             layer B (planned, M2): LLM narrative detector
             claims <-> computation consistency, closed taxonomy,
             span-grounded, verified by a second provider; fuses with
             layer A candidates (filter + population claim = flag)
```

The hybrid split is the point: leakage is an AST pattern — an LLM has no
business there. The LLM layer (M2) handles only what statics cannot:
whether the prose claims match what the code computes. Until then Wald is
a pure static tool; nothing here calls any API.

Taxonomy is data (`flaws.yaml`): 8 flaw classes with definition, book
anchor, mutation recipe and severity. Adding a class = a record + a
detector + a mutation; prompts and reports generate from the same file, so
documentation and detector cannot drift.

## Install & use

```bash
pip install -e .[corpus,dev]

wald check notebook.ipynb            # markdown report, exit code 0/1/2
wald check notebook.ipynb --format json
wald corpus build                    # build clean corpus + verified mutants
wald eval                            # confusion matrix -> evals/<date>-eval.md
pytest                               # unit tests + golden gates G0/G1
```

Wald never executes *your* notebook (static analysis + stored outputs
only). Only self-authored corpus notebooks are executed, at corpus build
time, to verify mutations.

## Roadmap

- **M2** — LLM narrative layer + cross-provider verifier: survivorship
  decision, regression-to-mean claims, significant-but-meaningless; fusion
  rules in taxonomy. Gates: F1 ≥ 0.7, verifier kills ≥ 80% of seeded false
  flags.
- **M3** — table consistency checks (cohort sums vs. declared population).
- **M4** — GitHub Action with PR annotations, severity calibration,
  dogfood on real (licensed) notebooks.
- Corpus contributions welcome: a clean notebook must satisfy the clean
  criteria in `wald/corpus.py` and pass review; a new flaw class needs a
  taxonomy record, a detector, and a mutation with mechanical `verify()`.

## License

MIT.
