# Changelog

## 0.2.0

- Fourth static class: `leakage-temporal-shuffle` — shuffled train/test
  splits on time-ordered data with lag/rolling-window features, evaluated
  per evaluation sink on that sink's dependency chain. Adversarially
  hardened: 4 confirmed false-positive shapes (sorted-frame neighbor
  features) fixed at the root; 4 evasions documented as residual risk.
- Corpus grew to 83 clean notebooks (56 synthetic + 27 real), 192 verified
  mutants. New class: 16/16 at precision 1.00 / recall 1.00. All four
  static classes still precision 1.00 / recall 1.00, clean FP rate 0.0%.
  (`evals/2026-07-05-eval.md`, `evals/2026-07-05-eval.json`.)

## 0.1.0

- Static layer: three flaw classes — `leakage-fit-before-split`,
  `testing-multiple-uncorrected`, `baserate-accuracy-imbalanced` — plus a
  below-floor candidate for `selection-survivorship-cohort`. Mutation-tested
  at precision 1.00 / recall 1.00 on all three; 0.0% FP on 75 clean
  notebooks (48 synthetic + 27 real). (`evals/2026-07-04-eval.md`.)
- Narrative layer (M2): LLM claims-vs-computation detector, cross-provider
  verifier, fusion — built and tested key-free against replay fixtures.
  G2/G3 quality gates not yet run (blocked on Anthropic + OpenAI keys).
- Dogfooded on 34 real notebooks: detector rebuilt after an initial 50%
  file flag rate (119 of 124 flags false positives); after the fix, 3
  confident flags, all confirmed real leaks, 0 known false positives.
  (`evals/2026-07-04-dogfood.md`.)
- SARIF 2.1.0 output (`wald check --format sarif`) and agent-gate docs
  (`docs/agent-gate.md`): Claude Code PostToolUse hook, pre-commit,
  GitHub Actions recipe with SARIF upload.
