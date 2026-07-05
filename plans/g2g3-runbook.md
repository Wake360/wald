# G2/G3 gate runbook

For the day both API keys arrive. Read once, then execute top to bottom.
Do not improvise a different order — the iteration/gate-attempt counts
are the actual budget constraint, not the dollars.

Sources: `plans/m2.md` (gates, cost ledger, iteration budget, R3),
`wald/llm.py`, `wald/cli.py`, `wald/eval.py`, `NEXT.md`.

## 0. Prerequisites

- `ANTHROPIC_API_KEY` set — detector.
- `OPENAI_API_KEY` set — verifier.
- Cross-provider is enforced in code, not a convention: `fuse.py` raises
  `detector and verifier must use distinct providers` if
  `det_backend.provider == ver_backend.provider`. Using the default pairing
  (`AnthropicBackend` + `OpenAIBackend`, wired by `cli._llm_backends`)
  satisfies this automatically. Do not swap in two backends from the same
  provider.
- Pinned model ids (verbatim from `wald/llm.py`):
  - `PINNED_DETECTOR_MODEL = "claude-sonnet-4-6"`
  - `PINNED_VERIFIER_MODEL = "gpt-4.1-2025-04-14"`
  `gate_evidence` in every report is computed, not asserted: it is `True`
  only if both backends are `kind == "api"`, both `.model` fields equal
  the two pinned ids above, and there were no backend errors. If you pass
  a different model, the report will silently show `gate_evidence: false`
  — check it after every run, don't assume the pinned ids stuck.
- Confidence floor for all narrative gating is `0.8`
  (`DEFAULT_CONFIDENCE_FLOOR` in `wald/detect.py`), pinned in every report.
- Gotcha: `--replay-dir` makes `ReplayBackend.kind == "replay"`
  immediately, even before the first call. `cli._missing_llm_keys` only
  checks backends with `kind == "api"`, so **with `--replay-dir` set, the
  CLI's missing-key check is silently skipped.** If a key is unset, the
  first cache-miss crashes with an uncaught `RuntimeError` instead of the
  clean `wald: --llm needs ... set` message. Confirm both env vars
  yourself before any command below that uses `--replay-dir`.
- Gotcha: `--replay-dir` also flips `kind` away from `"api"`, so
  `wald check --llm --replay-dir ...` against a held-out or `corpus/real`
  file will hit `_heldout_refusal` and refuse to run. Point
  `--replay-dir` runs at `corpus/clean/*.ipynb` or `corpus/mutated/*`
  dev-split notebooks only.
- Gotcha: `wald eval --llm` has **no missing-key guard at all** — unlike
  `wald check --llm`, `cmd_eval` never calls `_missing_llm_keys`. If a key
  is unset, the eval starts, spends detector tokens on however many
  notebooks precede the first cache-miss/verifier call, then dies with an
  uncaught `RuntimeError` (not a clean message) and writes no report. So
  before every `wald eval --llm` (steps 2 and 4) confirm both env vars
  yourself: `test -n "$ANTHROPIC_API_KEY" && test -n "$OPENAI_API_KEY" &&
  echo keys-ok`.
- Gotcha: **report filenames collide.** `run_llm_eval` writes
  `evals/<date>-llm-eval.json`/`.md` with the date only — no split, no
  attempt number. A second `wald eval --llm` on the same day silently
  overwrites the first. On the day the keys arrive you will run several
  evals in one day, so the dev checkpoint report gets clobbered by the
  held-out report, and you lose the dev recall you need for the R3 gap
  (STOP 2). After every eval run, immediately rename both files to a
  split/attempt-tagged name before running the next one (commands inline
  in steps 2 and 4).

## 1. Command sequence

Run in this order. Do not skip ahead if an earlier step fails or looks
wrong — fix it there.

### Step 1 — fixture-recording smoke (~$1)

```
wald check --llm corpus/mutated/cohort-s11__selection-survivorship-cohort__m0.ipynb --replay-dir evals/llm-fixtures/smoke
```

One dev-split **survivorship mutant** (not a clean notebook), both keys
live. The mutant is chosen on purpose: it triggers a static survivorship
candidate + a narrative population claim → the fusion rule fires → the
verifier is called. A clean notebook can produce zero narrative findings,
in which case `verify_finding` is never reached and the OpenAI verifier
integration goes **untested** — defeating half the point of the smoke.
This mutant exercises the Anthropic detector *and* the OpenAI verifier
(auth headers, JSON parsing, schema, grounding) in one notebook before any
eval-sized spend, and leaves a recorded fixture under
`evals/llm-fixtures/smoke/` for `$0` regression tests later. (Dev-split
mutant, so `_heldout_refusal` does not block the `--replay-dir` run.)

Record: exit code, whether a flag or candidate was produced, and — the
actual pass condition — that fixture files landed in **both** the
`detector/` and `verifier/` subdirs. An empty `verifier/` subdir means the
verifier was never called: pick another survivorship mutant seed and retry
before trusting the OpenAI path.

### Step 2 — dev checkpoint run(s) (~$2–3 each, max 2)

```
test -n "$ANTHROPIC_API_KEY" && test -n "$OPENAI_API_KEY" && echo keys-ok || echo MISSING-KEY-STOP
wald eval --llm --split dev --corpus corpus --out evals
# rename immediately so the held-out run (same day) cannot overwrite it,
# and bump ckpt1 -> ckpt2 on the second checkpoint:
mv "evals/$(date +%F)-llm-eval.json" "evals/$(date +%F)-llm-eval-dev-ckpt1.json"
mv "evals/$(date +%F)-llm-eval.md"   "evals/$(date +%F)-llm-eval-dev-ckpt1.md"
```

Run the key check first — `wald eval --llm` will not stop for a missing
key on its own (see Prerequisites). No `--replay-dir` — this is the real
API dev checkpoint the plan requires before further spend. The eval writes
the date-only `evals/<date>-llm-eval.json`/`.md`; rename it at once
(commands above) because the held-out attempt writes the same filename and
would otherwise clobber this dev evidence.

This single command produces both the G2 numbers (per-class F1, clean FP
rate, `dropped_ungrounded`) **and** the G3 numbers (`g3_per_recipe`,
`true_flag_survival`) for the dev split — `evaluate_narrative` computes
both in one pass. There is no separate CLI flag for G3-only; step 3 below
is read from this same report, not a new command.

This run is one of the two allowed iteration checkpoints (see STOP box).
If a prompt/fusion change is needed, make it, then use this same command
for the next checkpoint — max 2 total.

Record per run: per-class recall (esp. against the ≥0.6 bar), clean FP
rate (against ≤10%), `dropped_ungrounded.rate`, `gate_evidence` value
(note: `gate_evidence` can read `true` on a dev run — see STOP box, this
is not a gate pass), token usage / cost from the `usage` block.

### Step 3 — read the G3 dev numbers (bundled into step 2, ~$0 extra)

From the same dev report: `g3_per_recipe` (kill rate per recipe) and
`true_flag_survival.rate`. Note the dev G3 read covers only the four
within-recipe-split recipes (`control-group-present`,
`effect-size-present`, `scoped-claim`, `wrong-code-span`).
`legit-cv-generalization` is **reserved entirely for held-out** (all its
negative flags carry `split: heldout`), so it will NOT appear in the dev
`g3_per_recipe` — do not look for it here or treat its absence as a bug;
it shows up only in step 4's report. No standalone command exists for the
dev G3 read; if you need a fresh one (e.g. after a verifier-only prompt
tweak), rerun step 2's command — budget it as another dev-run checkpoint,
not a separate line.

### Step 4 — held-out gate attempt (~$4 each, max 2)

**PREFLIGHT — run every line before you spend a held-out attempt. If any
line does not print what is shown, STOP; do not run the eval.** Each
held-out run is one of only two attempts and cannot be taken back.

```
# 1. Dev checkpoint actually happened and its report still exists (not
#    clobbered). You need it for the R3 gap (STOP 2).
ls evals/$(date +%F)-llm-eval-dev-ckpt*.json   # must list >=1 file

# 2. Dev bar was met (recall >= 0.6 at FP <= 10% on all three classes).
#    If not, STOP 1 fires: ship static-only, do NOT spend on held-out.

# 3. Attempt budget: count tagged held-out reports already written.
ls evals/*-llm-eval-heldout-attempt*.json 2>/dev/null | wc -l  # must be 0 or 1

# 4. Both keys present (eval has no key guard of its own).
test -n "$ANTHROPIC_API_KEY" && test -n "$OPENAI_API_KEY" && echo keys-ok || echo MISSING-KEY-STOP

# 5. Pinned models unchanged (a swapped model silently sets gate_evidence=false).
grep -E 'PINNED_(DETECTOR|VERIFIER)_MODEL' wald/llm.py
#   expect: claude-sonnet-4-6  and  gpt-4.1-2025-04-14
```

```
wald eval --llm --split heldout --corpus corpus --out evals
# rename immediately (date-only filename collides with the dev report and
# with the other attempt); bump attempt1 -> attempt2 on the second run:
mv "evals/$(date +%F)-llm-eval.json" "evals/$(date +%F)-llm-eval-heldout-attempt1.json"
mv "evals/$(date +%F)-llm-eval.md"   "evals/$(date +%F)-llm-eval-heldout-attempt1.md"
```

No `--replay-dir`, no agent backend — `evaluate_narrative` raises if
either backend's `kind != "api"` on `split=="heldout"`. This is the only
command that can produce real G2/G3 gate evidence. It also folds
`corpus/real/` clean notebooks into the held-out clean set automatically
(`n_clean_real` in the report), so the FP count here already includes the
real-notebook caveat (`clean_fp_caveat`).

Pass bars for this report (from `plans/m2.md` Gates): **G2** per-class
`f1` ≥ 0.7 on all three narrative classes, `clean_fp_rate` ≤ 0.10,
`dropped_ungrounded.rate` ≤ 0.10. **G3** overall kill ≥ 0.80 AND every
recipe's `kill_rate` reported (including `legit-cv-generalization`, which
appears only here), with `true_flag_survival.rate` ≥ 0.75. A report is
gate evidence only if `gate_evidence == true` and `backend_errors` is
empty — otherwise the numbers stand but do not count, and fixing the cause
and rerunning burns your second attempt.

Record: everything from step 2's list, plus `n_clean_real`,
`clean_fp_files`, `missed_mutants`, `backend_errors` (must be empty), and
the exact `gate_evidence` boolean — this time it's the real thing if
`true`.

Only two attempts total, ever. See STOP box.

### Step 5 — LLM dogfood on the real notebooks (~$4)

```
mkdir -p evals/llm-dogfood
for nb in corpus/real/*.ipynb; do
  wald check --llm --format json "$nb" > "evals/llm-dogfood/$(basename "$nb").json"
done
```

The `mkdir -p` is required — the redirect fails on the first notebook if
`evals/llm-dogfood/` does not exist. Live keys with no `--replay-dir`, so
`_heldout_refusal` is skipped (both backends `kind == "api"`) and the
`corpus/real/` notebooks run normally.

`corpus/real/` currently holds 27 of the original 34 dogfood notebooks
(the rest are the confirmed-flaw set referenced in
`evals/2026-07-04-dogfood.md`, not re-run here). This produces real-world
narrative recall/FP evidence, reviewed by hand exactly like the static
dogfood pass. Report it as its own number, next to G2, never folded into
G2 — the plan is explicit that real-notebook recall is not gate evidence.

## 2. Spend ledger

Fill in `actual` after each step. Compare running total against the
$30–45 projection from `plans/m2.md`.

| step | projected | actual | notes |
|---|---|---|---|
| smoke (step 1) | ~$1 | | |
| dev checkpoint 1 (step 2) | ~$2–3 | | iteration 1 of 2 |
| dev checkpoint 2 (step 2, if needed) | ~$2–3 | | iteration 2 of 2 |
| G3 dev read (step 3) | $0 (bundled) | | |
| held-out gate attempt 1 (step 4) | ~$4 | | attempt 1 of 2 |
| held-out gate attempt 2 (step 4, if needed) | ~$4 | | attempt 2 of 2 |
| LLM dogfood (step 5) | ~$4 | | |
| **total** | **~$30, contingency $45** | | |

The binding constraint is the iteration/attempt counts below, not this
dollar figure — the ledger says so explicitly.

## 3. STOP conditions

```
+----------------------------------------------------------------------+
| STOP 1 — two-iteration termination rule                              |
| Max 2 dev-run checkpoints (step 2). After checkpoint 2, if dev recall |
| is not >= 0.6 at FP <= 10% on all three narrative classes: STOP.      |
| Do not spend on held-out. Ship static-only v1. Keep the corpus and    |
| harness — they survive either outcome (NEXT.md termination rule).     |
|                                                                        |
| STOP 2 — R3 tripwire                                                  |
| If dev-vs-heldout recall gap > 0.15 on any class: corpus-overfit      |
| alarm. The held-out number is the honest one — report it as-is, do   |
| not retune the prompt to close the gap after seeing held-out (that IS |
| the overfit).                                                        |
|                                                                        |
| STOP 3 — replay/agent runs are never gate evidence                    |
| gate_evidence is computed from kind==api (both backends) + pinned     |
| models + zero backend_errors. A dev-split run, a replay run, or an    |
| agent run can never count toward G2/G3 regardless of what the        |
| numbers show. Only step 4 (split=heldout, live keys, no replay-dir)   |
| can produce gate evidence, and even then only if gate_evidence==true. |
|                                                                        |
| STOP 4 — max 2 held-out gate attempts, period                         |
| After attempt 2, whatever the numbers are, that is the final G2/G3    |
| result. Do not run a third time hoping for a better sample.           |
+----------------------------------------------------------------------+
```

## 4. What to record after each step

- Step 1: `evals/llm-fixtures/smoke/detector/*.json`,
  `evals/llm-fixtures/smoke/verifier/*.json` exist; smoke notebook's
  check output (flag/candidate/none).
- Step 2 (each checkpoint): `evals/<date>-llm-eval.json` +
  `evals/<date>-llm-eval.md`; `narrative_classes` per-class recall/F1;
  `clean_fp_rate`; `dropped_ungrounded.rate`; `gate_evidence` (expect
  `true` here too — flag it as dev, not gate, per STOP 3); `usage`
  totals for cost tracking.
- Step 3: same dev report's `g3_per_recipe` (per-recipe kill rate for the
  four dev recipes only — `legit-cv-generalization` is held-out-only and
  will not appear) and `true_flag_survival.rate`.
- Step 4 (each attempt): same fields as step 2, plus `n_clean_real`,
  `clean_fp_files`, `missed_mutants`, `backend_errors` (must be empty),
  and the final `gate_evidence` boolean — this is the number that goes
  in the M2 writeup.
- Step 5: one JSON per notebook under `evals/llm-dogfood/`; after hand
  review, a real-recall count reported next to G2, labeled separately.
