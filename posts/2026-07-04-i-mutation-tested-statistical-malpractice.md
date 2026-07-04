# I mutation-tested statistical malpractice

*Draft — publish when the repo goes public.*

Code linters check style. I wanted one that checks whether the analysis is
a lie. Train/test leakage, uncorrected multiple testing, survivorship
bias, accuracy brags on imbalanced classes — the finite, catalogued ways a
notebook produces a confident wrong answer.

The hard part is not detecting these things. The hard part is knowing
whether your detector works. A linter demo proves nothing: you wrote the
example, of course it fires. So I built the measurement before the tool.

## Corpus before detector

The idea is mutation testing, borrowed from compiler-land and pointed at
statistics. You start with clean notebooks — analyses that state their
splits, correct their tests, scope their claims. Then you inject a named
flaw programmatically: move the scaler fit before the split, add an
uncorrected t-test screen over eight columns, filter the cohort to
survivors and rewrite the conclusion as a population claim.

The trick that makes this evidence rather than vibes: **every mutant must
mechanically prove the flaw is present before it enters the corpus.** A
leakage mutant is executed, and the scaler's `n_samples_seen_` must equal
the full row count — the transformer really saw the test set. A
multiple-testing mutant must really run more than five uncorrected tests.
No proof, no corpus entry. 64 of 64 mutants passed verification.

Detection quality is then a confusion matrix per flaw class, not an
opinion:

| class (static layer) | mutants | TP | FN | FP | precision | recall |
|---|---|---|---|---|---|---|
| leakage-fit-before-split | 12 | 12 | 0 | 0 | 1.00 | 1.00 |
| testing-multiple-uncorrected | 40 | 40 | 0 | 0 | 1.00 | 1.00 |
| baserate-accuracy-imbalanced | 4 | 4 | 0 | 0 | 1.00 | 1.00 |
| survivorship (candidate) | 8 | 8 | 0 | — | — | 1.00 |

Perfect numbers. Which is exactly the problem.

## Perfect numbers mean your test is too easy

The corpus was synthetic and stereotypical by design — canonical
pandas/sklearn idioms, written by the same person who wrote the detectors.
Those numbers measure detector correctness on shapes it was built for.
They say nothing about the real world. I wrote that caveat into the eval
report and then did the only honest next step: ran the tool on 34 real
notebooks from permissively licensed GitHub repositories — textbook
material, mostly correct by construction.

It flagged 17 of them. 118 confident flags. Hand review of every single
flag: **3 real, 119 false.**

A 96% false-positive rate is what kills linters. Nobody reads flag 12 of a
tool that was wrong 11 times. This was the project's named top risk, and
the synthetic eval — precision 1.00, remember — had been structurally
unable to see it.

## What the false positives taught

Every false flag got a verdict and an idiom label. The distribution was
the design document for the fix:

- 44× *fit on the train split only* — textbook-correct usage, flagged
  because the dataflow analysis unioned dependencies across
  reassignments and treated every `.fit` alike.
- 45× *model fit on full data with no test evaluation* — a classifier
  fitted for a decision-boundary plot is not preprocessing leakage. The
  detector conflated estimators with transformers: `.fit` is not one
  thing.
- 19× *name reuse across unrelated sections* — reuse `X` for a second
  dataset and the tool linked two analyses that never touch.
- 4× *cross-validation manages its own splits* — `GridSearchCV` and
  pipelines-inside-CV are the correct pattern, and the tool punished
  them.

Each cause is mechanical, so each fix is too: flow-sensitive dependency
chains that die on reassignment; a transformer/estimator distinction
(`fit_transform` is definitionally a transformer; bare `.fit` counts only
for known transformer classes or receivers that later `.transform`);
cross-validation treated as an evaluation sink with the estimator argument
excluded, because CV clones and refits whatever you pass it.

The review also surfaced what the tool *missed* — real leaks in
"unflagged clean" notebooks: median imputation over the full dataset
before the split (pandas, no sklearn transformer in sight), and
`SelectKBest(chi2).fit(X, y)` on all rows feeding `cross_val_score` — the
classic from Elements of Statistical Learning. Both are now detected;
both have regression tests.

After the rebuild, on the same 34 notebooks: 3 confident flags, and they
are exactly the 3 reviewer-confirmed real leaks. Zero known false
positives. Parse failures went from 13 cells to 0. The synthetic corpus
gates stayed at 1.00 — the fix cost nothing on the mutants.

And the corpus got its payment: 27 of the reviewed notebooks entered it
as licensed real clean entries. The clean false-positive rate is now
measured over 47 files, and the next detector change gets caught by real
code, not just my synthetic idioms.

## The loop is the method

Nothing here is specific to statistics. The loop is: corpus before
detector, mechanical proof before corpus entry, confusion matrix instead
of demos, and a dogfood run whose failures are labeled one by one and fed
back as both fixes and corpus entries. Perfect numbers are a smell.
The 119 false positives were the most useful artifact this project has
produced so far.

The tool is called wald, after Abraham Wald, who armored the bullet holes
that weren't in the returning planes. Static layer only for now — no LLM,
no API key, runs in CI, exit code 2 on high-severity findings. The
narrative layer (claims vs. computation) is next, and it will be measured
the same way, because now there is something honest to measure it
against.
