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
$ wald check examples/leaky.ipynb

# Wald report — examples/leaky.ipynb

verdict: 1 high, 0 medium | static layer (no LLM)

## HIGH: leakage-fit-before-split
- **Where:** cell 5, line 2
- **Evidence:** `scaler.fit_transform(...)` consumes X, feeding `train_test_split` (cell 5) — the fit happened before the split
- **Flaw:** A transformation (scaler, imputer, encoder, feature selector, PCA, vectorizer) is fitted on data that contains the test set.
- **Failure scenario:** Test metrics are inflated; the production model will underperform the reported numbers.
- **Fix:** Fit the transformer on the training split only; apply transform to the test split.
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
| leakage-fit-before-split | 24 | 24 | 0 | 0 | 1.00 | 1.00 |
| leakage-temporal-shuffle | 16 | 16 | 0 | 0 | 1.00 | 1.00 |
| testing-multiple-uncorrected | 96 | 96 | 0 | 0 | 1.00 | 1.00 |
| baserate-accuracy-imbalanced | 8 | 8 | 0 | 0 | 1.00 | 1.00 |
| selection-survivorship-cohort (candidate) | 16 | 16 | 0 | — | — | 1.00 |

False-positive rate on the 83-notebook clean corpus (56 synthetic + 27
hand-reviewed real notebooks from Apache-2.0/MIT repositories): **0.0%**.
The 27 real notebooks also shaped the static layer during dogfood (the
rebuild described below), so this is not an out-of-sample test for them —
the real-notebook share of the 0.0% leans optimistic.
192/192 mutants passed mechanical verification at build; 0 discarded.
32 of the 192 are narrative-only mutants (regression-to-mean-claim,
significance-meaningless); they are scored by the `--llm` eval, not by
this table, and no numbers are claimed for them until its gates run.
(Eval 2026-07-05, `evals/2026-07-05-eval.json`.)

Reproduce: `wald corpus build && wald eval`. Dated reports live in `evals/`.
Set `WALD_BUILD_DATE=YYYY-MM-DD` to pin the manifest/report date for a byte-identical rebuild.

**Dogfooded on real notebooks** (`evals/2026-07-04-dogfood.md`): the first
run on 34 real notebooks produced a 50% file flag rate — 119 of 124 flags
were false positives, hand-reviewed one by one. The detector was rebuilt
from that data (flow-sensitive dataflow, transformer/estimator distinction,
CV-aware sinks). After the fix: 3 confident flags on the same 34 notebooks,
all three confirmed real leaks, 0 known false positives. The report keeps
the full failure taxonomy; the 27 clean notebooks entered the corpus with
licenses recorded per file.

**Wider dogfood — first recall and precision numbers**
(`evals/2026-07-07-dogfood-wide.md`): 60 fresh GitHub notebooks, none
from the teaching repos above. wald flagged 13; adversarial review
confirmed 25 of 28 flags real (**0.89 flag precision**, 12 of 13 flagged
notebooks genuinely leaky) and, over a seeded 30-notebook miss-hunt
sample of the un-flagged set, found 8 leaks wald missed — **0.60
notebook-level recall** across the 43-notebook audited subset. 33
confirmed leak instances across 20 notebooks, all in the two leakage
classes (28 `leakage-fit-before-split`, 5 `leakage-temporal-shuffle`).

**Honest caveats:** the recall evidence is leakage-only — this sample
produced zero confirmed real instances of the testing or base-rate
classes, so no real-world recall is claimed for them. The one false
positive (an RFECV scout-subsample pattern) shows the corpus's 0.0% FP
does not carry to messy real code; 0.60 recall is a one-seed subset
estimate, upper-leaning. Ground truth is multi-agent adversarial review,
not expert human labeling.

## What Wald sees (v1) and what it doesn't

The static layer (deterministic, no API key, runs in CI) decides four
classes on its own:

- `leakage-fit-before-split` — flow-sensitive def-use dataflow: a
  *transformer* (not a model) is fitted on data whose transformed output
  feeds `train_test_split`, or transforms a split part after a full-data
  fit. Cross-validation counts as an evaluation sink: supervised selection
  fitted on the CV data (`SelectKBest.fit(X, y)` → `cross_val_score`) is a
  confident flag; unsupervised pre-CV fits (scaler/PCA) stay below the
  floor as candidates. Pandas self-statistic imputation before the split
  (`df[c].replace(0, df[c].median())`) is covered. Pipelines passed to CV,
  fits on the train split, label encoding and estimator fits are not
  flagged — each of those idioms was a measured false-positive class in
  the dogfood report.
- `leakage-temporal-shuffle` — time-ordered data with lag/rolling-window
  features (`shift`, `rolling`, `diff`, `pct_change`, …) split by a
  shuffled protocol — default `train_test_split`, shuffled `KFold` —
  instead of `TimeSeriesSplit`, `shuffle=False`, or a temporal cutoff.
  Signals are evaluated per evaluation sink on that sink's dependency
  chain: a date column alone never fires, bare `scipy.ndimage.shift` /
  `sklearn.utils.resample` / `np.diff` never fire, and a frame sorted on
  a non-date key with neighbor-comparison features never fires.
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

## GitHub Action

`Wake360/wald@v0.2.1` is a composite action that installs `wald-lint`, runs
`wald check` on the paths you pass, writes a SARIF 2.1.0 log, and reports a
`gate-exit-code` mapped through `fail-on` (high, medium, or never). The
action never fails the job itself — it exposes the code and lets your
workflow upload SARIF before gating, so findings show up as inline PR
annotations even on a red run. The consumer must grant `security-events:
write` for the SARIF upload, put Python >= 3.10 on `PATH` (via
`actions/setup-python`), and run on a Linux or macOS runner (the steps are
bash).

```yaml
name: wald
on: [pull_request]

jobs:
  check-notebooks:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Find changed notebooks
        id: changed
        run: |
          files=$(git diff --name-only \
            origin/${{ github.base_ref }}...${{ github.sha }} -- '*.ipynb')
          echo "files=$files" >> "$GITHUB_OUTPUT"

      - name: wald
        id: wald
        if: steps.changed.outputs.files != ''
        uses: Wake360/wald@v0.2.1
        with:
          paths: ${{ steps.changed.outputs.files }}
          fail-on: high

      - name: Upload SARIF
        if: always() && steps.changed.outputs.files != ''
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: ${{ steps.wald.outputs.sarif-file }}

      - name: Gate on wald
        if: steps.changed.outputs.files != ''
        run: exit ${{ steps.wald.outputs.gate-exit-code }}
```

## Architecture

```
notebook ──> ingest (nbformat) ──> layer A: static detectors ──> report
                                    - libcst def-use dataflow      md / json
                                    - deterministic, key-free      exit code
                                    - exact cell:line + evidence

             layer B (--llm): LLM narrative detector
             claims <-> computation consistency, closed taxonomy,
             span-grounded, verified by a second provider; fuses with
             layer A candidates (filter + population claim = flag)
```

The hybrid split is the point: leakage is an AST pattern — an LLM has no
business there. The LLM layer handles only what statics cannot: whether
the prose claims match what the code computes. It is built and tested
key-free against replay fixtures; its quality gates (G2/G3) have not run
yet — they need Anthropic + OpenAI keys — so no narrative-layer numbers
are claimed. Static-only is the default; without `--llm`, Wald calls no
API.

Taxonomy is data (`flaws.yaml`): 8 flaw classes with definition, book
anchor, mutation recipe and severity. Adding a class = a record + a
detector + a mutation; prompts and reports generate from the same file, so
documentation and detector cannot drift.

## Install & use

```bash
pip install wald-lint
wald check notebook.ipynb            # exit 0 clean / 1 medium / 2 high / 3 input or usage error
```

Contributing to Wald itself:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[corpus,dev]"       # or: uv sync --all-extras

wald check examples/leaky.ipynb      # exit 0 clean / 1 medium / 2 high / 3 input or usage error
wald check examples/leaky.ipynb --format json   # one object; a JSON array for multiple notebooks
wald corpus build                    # build clean corpus + verified mutants
wald eval                            # confusion matrix -> evals/<date>-eval.md
pytest                               # unit tests + golden gates G0/G1
```

`examples/leaky.ipynb` ships in the repo and reproduces the report shown
at the top of this file.

Wald never executes *your* notebook (static analysis + stored outputs
only). Only self-authored corpus notebooks are executed, at corpus build
time, to verify mutations.

## Roadmap

- **M2** — LLM narrative layer + cross-provider verifier + fusion: built,
  tested key-free against replay fixtures (`plans/m2.md`). Remaining: the
  G2/G3 gate runs (F1 ≥ 0.7, verifier kills ≥ 80% of seeded false flags),
  blocked on Anthropic + OpenAI keys.
- **M3** — table consistency checks (cohort sums vs. declared population).
- **M4** — GitHub Action with PR annotations, severity calibration.
  (Dogfood on real licensed notebooks: done, `evals/2026-07-04-dogfood.md`.)
- Corpus contributions welcome: see `CONTRIBUTING.md` for the clean-corpus
  criteria, real-notebook licensing rules, and the flaw-class contribution
  shape (taxonomy record, detector, mutation with mechanical `verify()`).

## License

MIT.
