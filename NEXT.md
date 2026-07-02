# Next steps

State (2026-07-02): M0+M1 done. Static layer, CLI, mutation corpus
(20 clean + 64 verified mutants), gates G0+G1 green, 31 tests. Eval
numbers are perfect because the corpus is synthetic — that is a caveat,
not an achievement. Repo is local-only.

Order below is deliberate: the free, honest test comes before any spend.

## 1. Dogfood on real notebooks (no key, ~1 afternoon)

The plan's R1 risk: false-positive fatigue kills linters. Measure it.

- Source: Kaggle kernels (Apache-2.0 by default, license recorded per
  kernel). `kaggle kernels list --language python --sort-by voteCount`
  on classic tabular competitions (Titanic, House Prices, fraud, churn),
  then `kaggle kernels pull`. 30–50 notebooks.
- Run `wald check` over all of them; hand-review every flag.
- Write the FP/miss report to `evals/` (dated, same format).
- Expected gaps to fix afterwards:
  - leakage detector must NOT flag sklearn `Pipeline` /
    `ColumnTransformer` (the correct pattern) and must understand
    `KFold` / `cross_val_score` splits,
  - parse robustness beyond `_strip_magics` (shell lines, cell magics).
- Publishing rule: aggregates only, never named authors.

## 2. Fold reviewed real notebooks into the corpus

Passing notebooks become licensed clean-corpus entries (record license
per file in MANIFEST). Mutations apply via existing `applicable()`
checks; where they don't apply, generalize the mutation, not the
notebook. This shrinks the "synthetic-only" caveat in README.

## 3. M2 — LLM narrative layer (blocked on API key + ~$150 budget)

The signature move: fusion. Static survivorship candidate (conf 0.55)
+ narrative population claim = high-confidence flag.

- Layer B detector: claims vs. code consistency, closed taxonomy,
  prompt generated from `flaws.yaml`, structured output, two mandatory
  evidence spans per flag.
- Verifier on a different provider; negative corpus of seeded false
  flags; gates G2 (F1 ≥ 0.7, FP ≤ 10%) and G3 (verifier kills ≥ 80%).
- New narrative classes: survivorship decision, regression-to-mean,
  significant-but-meaningless. Mutations for each exist as recipes in
  the plan (LifeOS `outputs/wald-plan.md`, Příloha A).

## 4. Blogpost #1 draft (needs only what exists + step 1 numbers)

"I mutation-tested statistical malpractice" — methodology post:
corpus-before-detector, mechanical `verify()`, confusion matrix as hero
image, dogfood numbers for credibility. Write now, publish when repo
goes public. The writeup is half the artifact.

## 5. Parked until the repo is public

- GitHub Action with PR annotations (M4) — pointless while local-only.
- M3 table QA and severity calibration follow M2.

## Termination rule (from the plan, unchanged)

If after M2 the narrative layer cannot hold recall ≥ 0.6 with FP ≤ 10%
after two prompt/fusion iterations, v1 narrows to the static linter and
ships with the honest eval report. The static layer alone is a usable
tool; the corpus is a durable artifact either way.
