# Wald LLM eval — 2026-07-09 (corpus built 2026-07-05, split dev)

Detector anthropic-agent/session (agent), verifier codex/session (agent). Confidence floor 0.8. Gate evidence: False.

## Narrative classes (layer B, full pipeline)

| class | TP | FN | FP | precision | recall | F1 |
|---|---|---|---|---|---|---|
| regression-to-mean-claim | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| selection-survivorship-cohort | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| significance-meaningless | 8 | 0 | 0 | 1.00 | 1.00 | 1.00 |

Clean FP rate (28 notebooks, 0 real): 0.0% (0 files)
Dropped ungrounded: 0/23 raw findings (0.0%)

## Backend errors (17 files — gate evidence void)

- mutated/cohort-s12__selection-survivorship-cohort__m1.ipynb: backend error: agent session failed (exit 1): 
- mutated/cohort-s13__selection-survivorship-cohort__m0.ipynb: backend error: agent session failed (exit 1): 
- mutated/cohort-s13__selection-survivorship-cohort__m1.ipynb: backend error: agent session failed (exit 1): 
- mutated/cohort-s14__selection-survivorship-cohort__m0.ipynb: backend error: agent session failed (exit 1): 
- clean/churn-s11.ipynb: backend error: agent session failed (exit 1): 
- clean/churn-s12.ipynb: backend error: agent session failed (exit 1): 
- clean/churn-s13.ipynb: backend error: agent session failed (exit 1): SessionEnd hook [node "${CLAUDE_PLUGIN_ROOT}/scripts/session-lifecycle-hook.mjs" SessionEnd] failed: EAGAIN: resource temporarily unavailable, read


- clean/churn-s14.ipynb: backend error: agent session failed (exit 1): 
- clean/abtest-s12.ipynb: backend error: agent session failed (exit 1): 
- clean/abtest-s13.ipynb: backend error: agent session failed (exit 1): 
- clean/abtest-s14.ipynb: backend error: agent session failed (exit 1): SessionEnd hook [node "${CLAUDE_PLUGIN_ROOT}/scripts/session-lifecycle-hook.mjs" SessionEnd] failed: EAGAIN: resource temporarily unavailable, read


- clean/housing-s13.ipynb: backend error: agent session failed (exit 1): 
- clean/cohort-s14.ipynb: backend error: agent session failed (exit 1): 
- clean/program-s12.ipynb: backend error: agent session failed (exit 1): 
- clean/program-s13.ipynb: backend error: agent session failed (exit 1): SessionEnd hook [node "${CLAUDE_PLUGIN_ROOT}/scripts/session-lifecycle-hook.mjs" SessionEnd] failed: EAGAIN: resource temporarily unavailable, read


- clean/program-s14.ipynb: backend error: agent session failed (exit 1): 
- clean/forecast-s11.ipynb: backend error: agent session failed (exit 1): 

## G3 — seeded false flags (verifier kill rate per recipe)

- control-group-present: killed 4/4 (100%)
- effect-size-present: killed 4/4 (100%)
- scoped-claim: killed 4/4 (100%)
- wrong-code-span: killed 8/8 (100%)

True-flag survival on the same run: 20/20 (100%)

## Honest caveats
- Narrative mutants are template prose from one author; the phrasing holdout detects memorization but cannot rule it out.
- Token/cost totals: detector None, verifier None (None = backend exposes no usage counters).