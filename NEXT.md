# Next steps

State (2026-07-04): M0+M1 done, dogfood done. First contact with 34 real
notebooks produced the predicted FP flood (119/124 flags false); detector
rebuilt from the review data — flow-sensitive dataflow, transformer vs.
estimator, CV sinks, imputation pattern. After: 3 confident flags on the
same set, all confirmed real, 0 known FPs, parse errors 0. 27 reviewed
clean notebooks folded into `corpus/real/` with licenses; clean corpus is
now 47 files at 0.0% FP. 47 tests. Full story: `evals/2026-07-04-dogfood.md`.
Blogpost #1 drafted (`posts/`). Repo still local-only.

## 1. Widen the dogfood set (no key, incremental)

Teaching repos are the easy case. Next: messier sources — Kaggle kernels
proper (needs `kaggle` CLI + credentials, licenses recorded per kernel),
or GitHub search for analysis notebooks in org repos. Target: first real
recall number (needs ≥ 30 confirmed real flaws, we have 7).

- Known miss classes to keep on the list: groupby-`apply` imputation,
  function-scoped target-correlation selection. Both need M2/M3, not more
  static heuristics — resist the urge.

## 2. M2 — LLM narrative layer (blocked on API key + ~$150 budget)

Unchanged plan, one addition from the dogfood data: the contested pre-CV
unsupervised fits (candidates at 0.75) are now a third fusion input
besides survivorship — narrative claims about CV scores + static pre-CV
fit candidate = confident flag.

- Layer B detector: claims vs. code consistency, closed taxonomy,
  prompt generated from `flaws.yaml`, structured output, two mandatory
  evidence spans per flag.
- Verifier on a different provider; negative corpus of seeded false
  flags; gates G2 (F1 ≥ 0.7, FP ≤ 10%) and G3 (verifier kills ≥ 80%).
- New narrative classes: survivorship decision, regression-to-mean,
  significant-but-meaningless. Mutations for each exist as recipes in
  the plan (LifeOS `outputs/wald-plan.md`, Příloha A).

## 3. Publish

- Blogpost #1 draft is in `posts/` — final numbers are in, publish when
  the repo goes public.
- Pre-publication checklist: LICENSE files in `corpus/real/LICENSES/`
  ship with the repo (Apache-2.0/MIT attribution), report stays
  aggregate-only.

## 4. Parked until the repo is public

- GitHub Action with PR annotations (M4) — pointless while local-only.
- M3 table QA and severity calibration follow M2.

## Termination rule (from the plan, unchanged)

If after M2 the narrative layer cannot hold recall ≥ 0.6 with FP ≤ 10%
after two prompt/fusion iterations, v1 narrows to the static linter and
ships with the honest eval report. The static layer alone is a usable
tool; the corpus is a durable artifact either way.
