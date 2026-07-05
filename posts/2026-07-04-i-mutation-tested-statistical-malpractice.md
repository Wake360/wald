# I mutation-tested statistical malpractice

I built a linter for statistical malpractice in notebooks: train/test
leakage, uncorrected multiple testing, survivorship bias, accuracy brags
on imbalanced classes. On its own eval suite it scored precision 1.00.
Then I ran it on 34 real notebooks from GitHub. It flagged 17 of them:
124 flags, 118 at full confidence. Review verdict: 3 real, 119 false,
2 borderline candidates.

A 96% false-positive rate is what kills linters. Nobody reads flag 12 of
a tool that was wrong 11 times. False-positive flood was the project's
named top risk, and the eval that said 1.00 was structurally unable to
see it coming. This post is about why, and about what the 119 wrong
flags were worth.

## The eval came first

Code linters check style. I wanted one that checks whether the analysis
is a lie. The failure modes are finite and catalogued: fit the scaler
before the split, screen eight columns with uncorrected t-tests, filter
the cohort to survivors and write the conclusion as a population claim.

Writing detectors for these is straightforward. Knowing whether they
work is not. A demo proves nothing; you wrote the example, of course it
fires. So the measurement came before the tool: mutation testing,
borrowed from compiler-land and pointed at statistics. Start with clean
notebooks. Inject a named flaw programmatically. Now you have labeled
data, and detection quality is a confusion matrix per flaw class instead
of an opinion.

One rule keeps the corpus from being vibes: every mutant must
mechanically prove its flaw before it enters. A leakage mutant is
executed, and the scaler's `n_samples_seen_` must equal the full row
count — the transformer really saw the test set. A multiple-testing
mutant must really run more than five uncorrected tests. No proof, no
entry. 64 of 64 mutants passed verification.

| class (static layer) | mutants | TP | FN | FP | precision | recall |
|---|---|---|---|---|---|---|
| leakage-fit-before-split | 12 | 12 | 0 | 0 | 1.00 | 1.00 |
| testing-multiple-uncorrected | 40 | 40 | 0 | 0 | 1.00 | 1.00 |
| baserate-accuracy-imbalanced | 4 | 4 | 0 | 0 | 1.00 | 1.00 |
| survivorship (candidate) | 8 | 8 | 0 | — | — | 1.00 |

Perfect numbers. That should have worried me more than it did.

## Perfect numbers mean the test is too easy

The corpus was synthetic and stereotypical by design: canonical
pandas/sklearn idioms, written by the same person who wrote the
detectors. The numbers measure detector correctness on shapes it was
built for, nothing else. That caveat was in the eval report from day
one. The only way past it was contact with real code, so I pulled 34
notebooks from five permissively licensed GitHub repos (teaching
material, mostly correct by construction) and ran the tool.

Every flag got a verdict and an idiom label: one reviewer per notebook,
miss-hunters over the unflagged files, and an adversarial pass that
tried to refute every claimed true positive. The reviewers were LLM
agents, not domain-expert humans, so weight the ground truth
accordingly. The 3-versus-119 split was not close.

## What the false positives taught

The idiom distribution was the design document for the fix:

- 44× *fit on the train split only* — textbook-correct usage, flagged
  because the dataflow analysis unioned dependencies across
  reassignments and treated every `.fit` alike.
- 45× *model fit on full data with no test evaluation* — a classifier
  fitted for a decision-boundary plot is not preprocessing leakage. The
  detector conflated estimators with transformers. `.fit` is not one
  thing.
- 19× *name reuse across unrelated sections* — reuse `X` for a second
  dataset and the tool linked two analyses that never touch.
- 4× *cross-validation manages its own splits* — `GridSearchCV` and
  pipelines-inside-CV are the correct pattern, and the tool punished
  them.

Each cause is mechanical, so each fix is too. Flow-sensitive dependency
chains that die on reassignment. A transformer/estimator distinction:
`fit_transform` is definitionally a transformer; bare `.fit` counts only
for known transformer classes or receivers that later `.transform`.
Cross-validation as an evaluation sink with the estimator argument
excluded, because CV clones and refits whatever you pass it.

The review also surfaced what the tool missed: real leaks in notebooks
it had passed as clean. Median imputation over the full dataset before
the split, in plain pandas with no sklearn transformer in sight.
`SelectKBest(chi2).fit(X, y)` on all rows feeding `cross_val_score`, the
classic from Elements of Statistical Learning. Both are now detected;
both have regression tests.

After the rebuild, on the same 34 notebooks: 3 confident flags — the two
pre-split median imputations and the SelectKBest case, all
reviewer-confirmed real leaks. Zero known false positives. Parse
failures went from 13 cells to 0. The synthetic gates stayed at 1.00;
the fix cost nothing on the mutants.

The run paid twice. 27 of the 34 notebooks reviewed clean and entered
the corpus as licensed real entries, so the clean false-positive rate
was, at that point, measured over 47 files, and every detector change
since gets caught by real code instead of my own synthetic idioms. The
honest limit: recall on real flaws rested on 7 confirmed instances at
the time, too few to quote as a number.

None of this is specific to statistics. The loop works for any detector
project, and its most valuable output to date is the labeled list of its
own failures.

The tool is called wald, after Abraham Wald, who armored the bullet
holes that weren't in the returning planes. At the time of this story it
was static layer only: no LLM, no API key, runs in CI, exit code 2 on
high-severity findings. The narrative layer — checking prose claims
against what the code computes — came next, measured the same way,
because by then there was something honest to measure it against.

The code is at github.com/filipvachek/wald.
