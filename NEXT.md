# Next steps

State (2026-07-09): v1 is closed. Per the pre-planned termination rule
(bottom of this file), the two API keys (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`) needed for the LLM narrative layer's G2/G3 quality
gates never arrived, so v1 ships as the static linter only — Milestones
0, 1, and 2 (static layer + corpus + mutation harness) are the shipped
product. `wald-lint 0.2.1` is on PyPI (install-proven in a clean venv)
and the repo is public (github.com/Wake360/wald). The static layer is
four classes at precision/recall 1.00 on 192 mutants, 0.0% FP on 83
clean notebooks. The GitHub Action (`action.yml`) ships and was
demonstrated end-to-end (a real PR annotation surfaced, then dismissed).
First real-world numbers are on main: **0.60 notebook-level recall, 0.89
flag precision** over 60 fresh GitHub notebooks
(`evals/2026-07-07-dogfood-wide.md`), leakage classes only. Roadmap:
`plans/v1-completion.md`.

0.3.0 adds, on top of that closed v1: `.py` input support (plain scripts
and percent-format cells), `--keep-going` on `wald check` (continue past
unreadable files, exit 3 if any failed), `wald rules` (list the taxonomy's
flaw classes, `--format json` for machines), and pre-commit hooks
(`.pre-commit-hooks.yaml`). The G2/G3 gate harness stays intact and
unchanged for whoever runs it with keys later — nothing about the
termination rule requires deleting or degrading that code path.

## 1. Milestone 3 — LLM narrative layer (run-path done; G2/G3 gates never ran, keys never arrived)

The narrative (`--llm`) layer is built and tested key-free against replay
fixtures. **Run-path: done.** It runs end-to-end on subscriptions via
`--llm-subscription` (claude/codex CLIs, $0 API) — proven on a survivorship
mutant and dogfooded over `corpus/real/*`
(`evals/2026-07-08-llm-dogfood-subscription.md`: 27 notebooks, 0 backend
errors, 1 notebook flagged with 2 confirmed true-positive leaks on hand
review, non-gate). An indicative dev-split run over the same subscription
backend is at `evals/2026-07-09-llm-eval-dev-subscription-indicative.md`
— unpinned models, not reproducible, 17 of its files hit backend errors
and are excluded, so it covers only the completed subset and is not gate
evidence. The three narrative definitions were also sharpened from the
book-extraction sources and frozen.

**Closed, not open:** the G2/G3 quality gates never ran — that needed the
two API keys, and per the termination rule below, v1 does not wait on
them any longer. Subscription runs are `gate_eligible=False` by
construction and cannot substitute for the gate (see the bullet below).
If the two keys arrive later, the sequence below is still the way to
run the gate; nothing here needs re-deriving.

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

## 2. Optional work (v1 is shipped; none of this blocks anything)

- **Dogfood batch 2.** Batch 1 already met the ≥30-confirmed target (33
  instances), so a second batch is optional. Its value is coverage, not
  count: batch 1 produced zero confirmed real instances of
  `testing-multiple-uncorrected` and `baserate-accuracy-imbalanced`, so the
  0.60 recall is leakage-only. A targeted batch aimed at those two classes
  would turn "unmeasured" into a real number. Sourcing mechanics and the
  confirmation-workflow spec are in `plans/v1-completion.md` (WS-A).
- **Quiet re-run of the indicative subscription eval.** The 2026-07-09
  dev-split run (`evals/2026-07-09-llm-eval-dev-subscription-indicative.md`)
  had 17 files hit backend errors (agent session failures, several citing
  a `SessionEnd` hook `EAGAIN`). A re-run during a quieter window would
  tell whether that was environmental or a real backend limitation, and
  would let the indicative numbers cover the full corpus instead of the
  completed subset. Still not gate evidence either way — subscription
  backends stay `gate_eligible=False`.
- **selection-survivorship-cohort** stays parked for narrative fusion
  (a scoped-claim notebook emits the identical below-floor candidate;
  static-alone cannot hold precision). Decided by the narrative layer's
  claim signal if the gate ever runs.
- **Known miss classes** left to a future narrative pass, not more static
  heuristics: groupby-`apply` imputation, function-scoped
  target-correlation selection.
- **Editor integration** (LSP or extension surfacing flags inline) is a
  v2 candidate, not scoped or started.

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
- **`SURVIVOR_QUERY_ASSIGN_RE` in `wald/detect.py` is O(n^2)** on long
  single-line inputs just under the 200KB cell cap (found in the 0.3.0
  review; the over-cap path is already fixed via `skipped_cells`).
  Follow-up: bound the leading `\w+` or pre-filter to lines containing
  `.query(`.
- **The 0.3.0 depth-guard fix over-counts scientific-notation literals**
  — `_structural_depth` now keys off the character preceding a `+`/`-`
  rather than the one following it, so a cell containing 500+ e-notation
  floats (`1e-5`, `2e+3`, ...) gets depth-skipped even though libcst
  would parse it fine. Safe direction (over-skip, not a crash); noted by
  the reviewer 2026-07-09.

## Termination rule (from the plan, unchanged)

If after M3 the narrative layer cannot hold recall ≥ 0.6 at FP ≤ 10% after
two prompt/fusion iterations, v1 narrows to the static linter and ships
with the honest eval report. The static layer alone is a usable tool and
the corpus is a durable artifact either way. If the two keys never arrive,
that static-only outcome is the default v1 and Milestones 0–2 are the
shipped product.
