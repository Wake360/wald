# Next steps

State (2026-07-07): Milestones 0, 1, and 2 are done. `wald-lint 0.2.1`
is on PyPI (install-proven in a clean venv) and the repo is public
(github.com/Wake360/wald). The static layer is four classes at
precision/recall 1.00 on 192 mutants, 0.0% FP on 83 clean notebooks. The
GitHub Action (`action.yml`) ships and was demonstrated end-to-end (a real
PR annotation surfaced, then dismissed). First real-world numbers are on
main: **0.60 notebook-level recall, 0.89 flag precision** over 60 fresh
GitHub notebooks (`evals/2026-07-07-dogfood-wide.md`), leakage classes
only. Roadmap: `plans/v1-completion.md`.

The only remaining v1 work is the LLM narrative layer's gate runs. Below
that are optional static extensions and known papercuts — none block v1.

## 1. Milestone 3 — LLM narrative layer (run-path DONE; G2/G3 gates blocked on two API keys)

The narrative (`--llm`) layer is built and tested key-free against replay
fixtures. **Run-path: done.** It now runs end-to-end on subscriptions via
`--llm-subscription` (claude/codex CLIs, $0 API) — proven on a survivorship
mutant and dogfooded over `corpus/real/*`
(`evals/2026-07-08-llm-dogfood-subscription.md`: 27 notebooks, 0 backend
errors, 1 true-positive leakage catch, non-gate). The three narrative
definitions were also sharpened from the book-extraction sources and frozen.
**Still open:** it has never run its G2/G3 quality gates — that needs the two
API keys and is the last v1 milestone. Subscription runs are `gate_eligible=
False` by construction and cannot substitute (see the bullet below).

- Needs `ANTHROPIC_API_KEY` + `OPENAI_API_KEY` in the environment
  (cross-provider is a hard constraint, enforced in `fuse.py`). Projected
  spend ~$19–21 of the $150 budget; the binding limit is the 2-iteration /
  2-attempt count, not dollars.
- Sequence and exact gate thresholds: `plans/v1-completion.md` §3A and
  `plans/g2g3-runbook.md`. In short: ~$1 fixture smoke → ≤2 dev checkpoints
  (dev bar: recall ≥ 0.6, FP ≤ 10%, dropped ≤ 10% per class) → decide the
  `[output]`-quote grounding seam → ≤2 held-out attempts (gate: all three
  F1 ≥ 0.7, clean FP ≤ 0.10, pooled G3 kill ≥ 0.80, survival ≥ 0.75,
  `gate_evidence == true`, `backend_errors == []`) → LLM dogfood over
  `corpus/real/*`.
- Only the three mutant-backed narrative classes enter the prompt.
  Confidences are fixed constants, never the model's number. Held-out
  contamination is structurally prevented (the eval raises), not just
  labeled.
- Publish rule: no narrative recall/F1 reaches README until a held-out
  report shows `gate_evidence == true`. Numbers then ride a follow-up tag
  through the existing release pipeline.
- Key-free running (not gating) is now available: `--llm-subscription`
  routes the detector/verifier through the `claude`/`codex` CLIs
  (subscription billing, no API keys). These backends are `kind="agent"`,
  `gate_eligible=False` — usable for dev and dogfood only. The held-out
  eval refuses them (`eval.py` raises on non-api at `heldout`), so this
  does not touch the gate above: G2/G3 still need the two keys.

## 2. Optional static extensions (not blocking v1)

- **Dogfood batch 2.** Batch 1 already met the ≥30-confirmed target (33
  instances), so a second batch is optional. Its value is coverage, not
  count: batch 1 produced zero confirmed real instances of
  `testing-multiple-uncorrected` and `baserate-accuracy-imbalanced`, so the
  0.60 recall is leakage-only. A targeted batch aimed at those two classes
  would turn "unmeasured" into a real number. Sourcing mechanics and the
  confirmation-workflow spec are in `plans/v1-completion.md` (WS-A).
- **selection-survivorship-cohort** stays parked for narrative fusion
  (a scoped-claim notebook emits the identical below-floor candidate;
  static-alone cannot hold precision). Decided by the narrative layer's
  claim signal in M3.
- **Known miss classes** to leave to M2/M3, not more static heuristics:
  groupby-`apply` imputation, function-scoped target-correlation selection.

## 3. Known papercuts

- **RFECV scout-subsample false positive** — the one FP from the wide
  dogfood. Documented with disposition in `docs/fit-before-split-fps.md`
  (accept for 0.2.x; a precise fix needs reach analysis). Revisit if a
  wider dogfood shows the idiom recurring.
- **`wald corpus build` hangs locally** on macOS — it executes mutant
  notebooks through Jupyter kernels, and one deadlocks under concurrent
  builds. Per-cell (180s) and startup (60s) timeouts already exist, so the
  phase at fault is not yet isolated. Operational fix for now: run one
  build at a time. CI is unaffected (fresh runners, green every release);
  do not patch the executor until a standalone single-build hang is
  reproduced.

## Termination rule (from the plan, unchanged)

If after M3 the narrative layer cannot hold recall ≥ 0.6 at FP ≤ 10% after
two prompt/fusion iterations, v1 narrows to the static linter and ships
with the honest eval report. The static layer alone is a usable tool and
the corpus is a durable artifact either way. If the two keys never arrive,
that static-only outcome is the default v1 and Milestones 0–2 are the
shipped product.
