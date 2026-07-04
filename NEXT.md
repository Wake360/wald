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

## 2. M2 — LLM narrative layer (planned; runs blocked on keys)

Detailed plan: `plans/m2.md` (2026-07-04, synthesized from a
3-designer + 2-critic workflow). Key points:

- Phase 1 is key-free and buildable now: narrative mutations, held-out
  split, negative corpus (one recipe mined from the 119 dogfood FPs),
  backend seam, detector, verifier, fusion, eval extension — all tested
  against replay fixtures.
- Phase 3 needs TWO keys (Anthropic detector + OpenAI verifier;
  cross-provider is a hard constraint). Projected spend ~$30–45 of the
  $150 budget.
- Only the three mutant-backed classes enter the prompt; pre-CV fusion
  ships FP-gated but recall-unclaimed; confidences are fixed constants,
  never the model's number; held-out contamination is structurally
  prevented (eval raises), not just labeled.

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
