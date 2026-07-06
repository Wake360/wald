# Wald v1 completion roadmap

Date: 2026-07-06
Owner: maintainer
Scope: integrate three finalized track plans (LLM path, Static layer, CLI UX) into one sequenced roadmap. Save as `plans/v1-completion.md`.

## 1. What "done" means for v1

v1 ships the wald statistical-integrity linter as a public, installable tool whose static layer carries an evidence-backed real-world recall number (or an honest bound) and whose narrative (`--llm`) layer has either passed the G2/G3 quality gates or been cleanly terminated to static-only with an honest eval report. The static layer keeps precision/recall 1.00 on 192 mutants and 0.0% FP on 83 clean notebooks; pyright becomes a hard CI gate at zero errors; the next release is tagged, published to PyPI, and proven installable. The CLI behaves like ruff/eslint in a terminal (severity color, directory recursion, roll-up, `--llm` progress) while the piped md/json/sarif bytes and the 0/1/2/3 exit-code contract stay byte-identical to 0.2.0. A GitHub Action surfaces findings as PR annotations. No narrative recall/F1 number reaches README until a held-out report shows `gate_evidence == true`; if the gates do not clear in two iterations, v1 narrows to the static linter and says so plainly.

Verified repo state adopted for this roadmap (supersedes the older "local-only, unpublished" framing): the repo is already public (`gh repo view Wake360/wald --json visibility -q .visibility` → `PUBLIC`), v0.1.0 and v0.2.0 are both released, and PyPI serves `wald-lint` 0.2.0. Consequences: "publish 0.2.0 for the first time" is done; the "until public" gate on the M4 Action and blogpost is already satisfied; further releases ride the existing trusted-publisher path.

## 2. Sequencing

### Conflicts resolved across tracks (explicit)

1. Repo public/published status. The planning preamble said "local-only, unpublished"; verification shows `PUBLIC` + PyPI 0.2.0. Resolution: adopt the verified state everywhere; M4 and the blogpost are unblocked; the CLI-UX track ships in the next release like any other change.

2. `dist/` cleanup and the release. Both the LLM and Static tracks claimed dist/version work. Resolution: releases are owned once, by Static WS-E. The LLM track's dist task is dropped; narrative numbers ride a later tag through the same pipeline.

3. `--llm` progress indication. Resolution: `wald check --llm` progress is owned by CLI-UX WS4 (format `checking i/N <name>`, stderr only, `\r`-overwrite, early-return newline). LLM WS-A2 is narrowed to the `wald eval --llm` loop inside `evaluate_narrative` only. One progress convention, no duplicate edit.

4. README edits. Static WS-A4a/A4b edits the real-recall caveat (README lines 80–81); LLM WS-F1a edits the narrative-layer paragraph (README line 144). Different lines, no conflict; coordinated in one commit per release.

5. Shared non-goals, stated once: no static heuristics for groupby-`apply` imputation or function-scoped target-correlation selection (M2/M3-owned); no CLI polish beyond the named items; API keys + $150 budget belong to the LLM track.

### Parallel vs gating

Key-free and immediately parallel: LLM WS-A, Static WS-B, WS-C, WS-D, the entire CLI-UX track, and Static WS-A (its adversarial confirmation is a review method, not a keyed backend run).

Gating relationships:
- Both keys gate LLM WS-B, WS-C(fix path), WS-D, WS-E. Cross-provider (Anthropic detector + OpenAI verifier) is enforced in `fuse.py`.
- LLM WS-A gates WS-B/WS-D. WS-B's dev bar gates WS-D held-out spend (STOP 1). WS-C decision completes before WS-D.
- LLM WS-D `gate_evidence == true` gates any published narrative number (WS-F1a); otherwise the termination rule fires (WS-F1b).
- Static WS-B/WS-C/WS-D land before Static WS-E2. Static WS-A does not block WS-E — the release ships with the current honest caveat; WS-A updates README afterward.
- The repo being public already satisfies the gate on Static WS-F (M4) and the blogpost.
- CLI-UX is independent of keys, budget, gates, and publishing; it rides the next release.

### Milestones

1. **Key-free hardening (parallel, $0).** LLM WS-A; Static WS-B, WS-C, WS-D; full CLI-UX track (WS0 golden first commit → WS1–WS5).
2. **Static evidence + release (no keys).** Static WS-A (≥ 30 confirmed real flaws or honest bound) → Static WS-E (dist cleanup, version bump, tag, PyPI, install proof) → Static WS-F (M4 Action + PR-annotation demo). A shippable static-only v1 on its own.
3. **LLM gate run (needs both keys).** LLM WS-B → WS-C → WS-D (≤ 2 held-out attempts) → WS-E (LLM dogfood over `corpus/real/*`).
4. **Publish narrative outcome or terminate.** `gate_evidence == true` → WS-F1a publishes numbers on a follow-up tag. Gates missed twice → WS-F1b static-only. Keys never arrive → Milestones 1+2 are v1, termination branch is the default.

---

## 3. Track plans

### 3A. Track: LLM path to done

#### Objective

Take the narrative (`--llm`) layer from built-but-ungated to gate-passed (or cleanly terminated), published, and runnable by a stranger holding an `ANTHROPIC_API_KEY` and an `OPENAI_API_KEY`. The gate machinery, eval, backend seam, and runbook (`plans/g2g3-runbook.md`) already exist; this track spends the two keys against the held-out split under the fixed iteration/attempt budget, decides the `[output]`-quote grounding seam before spending, records cost and latency, and writes the resulting numbers into README/CHANGELOG only after `gate_evidence == true`.

Definition of done (measurable):
- A held-out report `evals/<date>-llm-eval-heldout-attempt{1,2}.json` exists with `gate_evidence == true` and `backend_errors == []`, and either (a) passes G2 (per-class `f1` ≥ 0.7 on all three narrative classes, `clean_fp_rate` ≤ 0.10, `dropped_ungrounded.rate` ≤ 0.10) and G3 (pooled kill ≥ 0.80 computed as `sum(killed)/sum(total)` over `g3_per_recipe`, every recipe's `kill_rate` reported including `legit-cv-generalization`, `true_flag_survival.rate` ≥ 0.75), OR (b) the termination rule fired and v1 ships static-only with the honest report.
- Total spend recorded in the runbook ledger is ≤ $45 (of $150).
- Cost per notebook (detector + verifier tokens → dollars) and wall-clock latency per notebook are written down from real runs, not projections.
- Each failure mode has a green pytest test: backend outage → `test_retry_exhausted_on_persistent_5xx_raises_backend_error`; rate limit → `test_retry_on_429_honors_retry_after_then_succeeds`; partial batch → `test_evaluate_narrative_continues_past_backend_error` (`tests/test_narrative_eval.py`); budget → `test_anthropic_accumulates_usage` / `test_openai_accumulates_usage` (`tests/test_llm.py`) proving usage counters are recorded so spend can be extrapolated. Non-goals forbid building overrun-detection code; the usage tests are the DoD evidence.
- The `[output]`-quote grounding seam is either fixed (dev-validated, FP not regressed, line numbers proven in-bounds) or documented as residual with its measured recall cost.
- README and CHANGELOG carry held-out numbers only in the branch where gates passed; on termination they carry the static-only outcome.
- Distribution is owned by Static WS-E; this track no longer rebuilds `dist/` itself.

#### Workstreams

**WS-A — Pre-flight hardening (close budget/attempt leaks before spending)**

A1. Add the missing test for the `cmd_eval` key guard (the guard already ships at `wald/cli.py:145-150` — do not re-add it). Add `tests/test_cli.py::test_eval_llm_missing_keys_exits_3`.
- Verify: `env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY uv run wald eval --llm --split dev; echo $?` → stderr `wald: --llm needs ANTHROPIC_API_KEY and OPENAI_API_KEY set in the environment`, prints `3`. Then `uv run pytest -q tests/test_cli.py -k eval_llm_missing` → `1 passed`.

A2. Per-notebook progress in the `--llm` eval loop only (`wald check --llm` progress is CLI-UX WS4's). Emit one stderr line per notebook in `evaluate_narrative`'s loops, convention `checking i/N <name>`.
- Verify: keep it $0 — do NOT use `--replay-dir` against a not-yet-created fixture dir; `ReplayBackend.complete` (`wald/llm.py:238-261`) falls through to the real API on a cache miss. Monkeypatch stub backends (pattern: `tests/test_cli.py::test_cli_llm_run_reports_narrative_provenance`) in `tests/test_narrative_eval.py::test_eval_progress_lines`; `uv run pytest -q tests/test_narrative_eval.py -k progress` → `1 passed`.

A3. Audit and repoint the existing partial-batch test (`test_evaluate_narrative_continues_past_backend_error` already asserts run-finishes, failed-file-in-`backend_errors`, `gate_evidence == False`); correct the stale `-k partial_batch` filter references.
- Verify: `uv run pytest -q tests/test_narrative_eval.py -k continues_past_backend_error` → `1 passed`.

**WS-B — Fixture smoke + dev checkpoints (runbook steps 1–3, ≤ 2 iterations)**

B1. Fixture-recording smoke (~$1): with both keys live, `uv run wald check --llm corpus/mutated/cohort-s11__selection-survivorship-cohort__m0.ipynb --replay-dir evals/llm-fixtures/smoke`.
- Verify: `ls evals/llm-fixtures/smoke/detector/*.json evals/llm-fixtures/smoke/verifier/*.json` → both non-empty. Empty `verifier/` → verifier never ran, retry another survivorship seed. HTTP 404 model-not-found → apply the pinned-model-retired mitigation before spending further.

B2. Dev checkpoint run(s) (~$2–3 each, max 2): `test -n "$ANTHROPIC_API_KEY" && test -n "$OPENAI_API_KEY" && echo keys-ok`, then `uv run wald eval --llm --split dev --corpus corpus --out evals`; rename to `evals/<date>-llm-eval-dev-ckpt1.json/.md`.
- Verify: every `narrative_classes.*.recall` ≥ 0.6, `clean_fp_rate` ≤ 0.10, `dropped_ungrounded.rate` ≤ 0.10, `gate_evidence == true` (dev — not a gate pass), `usage` populated. Record per-notebook cost and wall-clock via `time`.

B3. Read the four dev G3 recipes from the same report ($0).
- Verify: `python -c "import json;d=json.load(open('evals/<date>-llm-eval-dev-ckpt1.json'));print(sorted(d['g3_per_recipe']),d['true_flag_survival']['rate'])"` → four recipes (`legit-cv-generalization` is held-out-only, correctly absent), survival ≥ 0.75.

**WS-C — `[output]`-quote grounding seam decision (before held-out)**

C1. Decide fix vs document. The seam: `_ground_finding` grounds `code_quote` against `cc.source` (`narrative.py:306`) while the model sees `_render_code(cc)` = source + `\n[output]\n<capped>` (`narrative.py:153-160`); an output-region quote is never a substring of `cc.source`, so it drops. Cost: 2/8 dev recall on significance, sampling-dependent.

C2a. Fix path (surgical): ground against `_render_code(cc)`, recompute the line span over the same rendered text, keep the ≤ 400-char cap. Line-number hazard: the span flows through `Flag.line` (`fuse.py:45,96`) into markdown (`report.py:94`) and SARIF `region.startLine` (`report.py:144,152`) — a quote grounded in `[output]` would produce a span past the real source. Either report a sentinel (`cell N, output`) or clamp `code_line_end` to `len(cc.source.splitlines())`; pick one explicitly.
- Verify: `tests/test_narrative.py::test_output_region_code_quote_grounds` asserts sentinel OR `code_line_end <= len(cc.source.splitlines())`; `uv run pytest -q tests/test_narrative.py -k output_region` → `1 passed`. Then one dev checkpoint (counts against WS-B's two): significance recall rises, `clean_fp_rate` stays ≤ 0.10. FP regresses > 0.10 → revert to C2b.

C2b. Document path: record the seam as residual (cost 2/8 dev significance recall, deferred to M3) in `render_llm_report`'s "Honest caveats" block (`eval.py:405-416`); assert the string is present via a small test or grep on the rendered dev .md. No code change, no iteration spent.

**WS-D — Held-out gate attempts (runbook step 4, ~$4 each, max 2)**

D1. Preflight: dev ckpt exists and meets the bar, ≤ 1 held-out report already, both keys present, `grep -E 'PINNED_(DETECTOR|VERIFIER)_MODEL' wald/llm.py` shows `claude-sonnet-4-6` and `gpt-4.1-2025-04-14`.
- Verify: every preflight line prints its expected value; dev bar unmet → STOP 1, do not spend, go to WS-F termination branch. The grep proves the source string, not servability — that check is B1's first live call.

D2. Run the held-out gate: `uv run wald eval --llm --split heldout --corpus corpus --out evals`; immediately `mv` to `evals/<date>-llm-eval-heldout-attempt1.json/.md`.
- Verify (exact — no "overall kill" field exists; eval computes only per-recipe):

  ```
  python -c "import json;d=json.load(open('evals/<date>-llm-eval-heldout-attempt1.json'));\
  g=d['g3_per_recipe'];\
  kill=sum(r['killed'] for r in g.values())/sum(r['total'] for r in g.values());\
  nc=d['narrative_classes'];\
  print('gate',d['gate_evidence'],'errs',d['backend_errors']);\
  print('f1',{c:nc[c]['f1'] for c in nc});\
  print('fp',d['clean_fp_rate'],'dropped',d['dropped_ungrounded']['rate']);\
  print('recipes',sorted(g),'kill',round(kill,3),'survival',d['true_flag_survival']['rate']);\
  print('fp_caveat',d['clean_fp_caveat'])"
  ```
  Require: `gate_evidence == true`, `backend_errors == []`, all three `f1` ≥ 0.7, `clean_fp_rate` ≤ 0.10, `dropped_ungrounded.rate` ≤ 0.10, all five recipes present, pooled kill ≥ 0.80, survival ≥ 0.75. `clean_fp_caveat` must be non-null (`eval.py:328` folds `corpus/real/*` into the held-out clean set, so the fused-FP number leans optimistic); read it now.

D3. Apply STOP 2 (R3) and STOP 4: compute per-class dev-vs-heldout recall gap by script; any gap > 0.15 → report held-out as-is, do not retune after seeing held-out. After attempt 2, the report is final regardless of numbers.
- Verify:

  ```
  python -c "import json;\
  a=json.load(open('evals/<date>-llm-eval-dev-ckpt1.json'))['narrative_classes'];\
  b=json.load(open('evals/<date>-llm-eval-heldout-attempt1.json'))['narrative_classes'];\
  print({c:round(a[c]['recall']-b[c]['recall'],3) for c in a if c in b})"
  ```
  Gaps written into the report notes; `ls evals/*-llm-eval-heldout-attempt*.json | wc -l` ≤ 2.

**WS-E — LLM dogfood real evidence (runbook step 5, ~$4)**

E1. `mkdir -p evals/llm-dogfood; for nb in corpus/real/*.ipynb; do uv run wald check --llm --format json "$nb" > "evals/llm-dogfood/$(basename "$nb").json"; done`. Real-notebook narrative recall/FP evidence, reported next to G2, never folded into it. Not blocked by `_heldout_refusal` — the guard early-returns `None` when both backends are `kind=="api"` (`cli.py:41-42`).
- Verify: `ls evals/llm-dogfood/*.json | wc -l` == count of `corpus/real/*.ipynb`; hand-review each flag; write a labeled real-recall/FP count separate from G2.

**WS-F — Publish or terminate**

F1a. Gates passed → write held-out G2/G3 numbers (incl. pooled kill from D2's formula) and the WS-E real-dogfood number into README (replace the "no narrative-layer numbers are claimed" paragraph, `README.md:144`) + CHANGELOG; carry the `clean_fp_caveat` string; ship on a follow-up tag through Static WS-E.
- Verify: `grep -n 'gate_evidence' evals/<date>-llm-eval-heldout-attempt*.json` shows `true`; README figures match that JSON exactly.

F1b. Gates failed / termination fired → narrow v1 to the static linter, keep corpus and harness, publish the honest eval report; README states the narrative layer did not clear G2/G3 in two iterations.
- Verify: README/CHANGELOG contain no claimed narrative recall/F1; `grep -i 'static-only\|did not clear' README.md` returns the outcome sentence.

F2. Full test + gate rebuild before the release tag (release owned by Static WS-E).
- Verify: `uv run pytest -q` → all pass (baseline 264 + new tests); `uv run wald eval --corpus corpus --out evals` still shows four static classes precision/recall 1.00, clean FP 0.0%.

#### Dependencies

WS-B, WS-C(fix), WS-D, WS-E need both keys; cross-provider enforced in `fuse.py`. WS-A before WS-B/WS-D. WS-C before WS-D. WS-B dev bar gates WS-D (STOP 1). WS-F publish branch depends on `gate_evidence == true`. Budget worst-case ≈ $1 + 2×$2–3 + 2×$4 + $4 ≈ $19–21; the binding limit is the 2-iteration / 2-attempt count, not dollars.

#### Risks

Termination fires (dev recall < 0.6 at FP ≤ 10% after ckpt2) → WS-F1b pre-planned. — R3 corpus overfit (dev-vs-heldout gap > 0.15) → report held-out honestly, no post-hoc retuning. — Wasted attempt (`gate_evidence == false` from swapped model/backend error) → D1 grep + B1 smoke. — Pinned model retired → on B1 404, re-run key-free dev numbers against the replacement id, re-pin, then spend. — Budget overrun: no auto-stop by design (m2 decision 5); usage tests prove token totals; B2's per-notebook cost lets you extrapolate; trigger = ledger > $45. — Backend outage / rate limit: `_post_json` retries once on 429 (honoring `Retry-After`) and once on transport failure, then fails closed; detector outage tags `dropped` and `cmd_check` exits 3; eval records the file in `backend_errors` and continues. — Seam fix regresses FP or emits a bogus line number → validate on dev, assert span bounds, revert to document path if FP > 0.10. — Same-day held-out filename collision → `mv` to `-attempt1`/`-attempt2` immediately. — G3 aggregation ambiguity (uneven totals: wrong-code-span 16, effect-size-present 8, scoped-claim 8, control-group-present 8, legit-cv-generalization 4) → gate pinned to pooled `sum(killed)/sum(total)`, every per-recipe rate reported.

#### Non-goals

Shared non-goals (§2). LLM-specific: no widening dogfood to ≥ 30 real flaws on the gate path (Static WS-A owns that); no new narrative classes (`linearity-extrapolation`, `leakage-future-feature` stay `narrative_enabled: false`); no new corpus families / `PreCVFitMutation`; no budget-enforcement machinery, confirmation channel, grounding-repair heuristics, or `pipeline.py`; no model-generated confidences (fixed constants 0.91/0.88/0.80); no calibration/severity work (M3).

#### Effort

WS-A: S. WS-B: M. WS-C: S–M. WS-D: M. WS-E: S. WS-F: S–M.

---

### 3B. Track: Static layer to done

#### Objective

Move the static layer from perfect-on-corpus to evidence-backed. This track produces a real-world recall number (or an honest bound), closes the two open static-quality items (temporal-shuffle evasions, survivorship candidate), makes pyright a hard gate, ships the next release, and ships the PR-annotation Action. This track owns all release/dist/publishing work for the roadmap.

Definition of done:
- Real-world recall backed by ≥ 30 confirmed real flaws with sources and licenses; OR an honest bound ("N confirmed of M reviewed, too few to report as a rate") in README and the dogfood report.
- The 4 temporal-shuffle evasions each triaged: fixed at root (mutant added, gates still 1.00/0.0%) or accepted with a one-paragraph rationale.
- selection-survivorship-cohort decided: promoted with mutants and gate coverage, or parked with a written reason and owner.
- `uvx pyright wald/` a hard CI gate at 0 errors.
- Next release (0.2.1 or 0.3.0 per E2's rule) tagged, released, installable from PyPI; stale `dist/` 0.1.0 artifacts gone.
- GitHub Action with PR annotations shipped and demonstrated on a test PR.

#### Workstreams

**A. Real-world recall evidence (widen the dogfood set)**

A1. Pull the next source batch from GitHub, not Kaggle (`which kaggle` returns nothing, no `~/.kaggle/`). Use `gh` (authed as Wake360) to search/clone analysis notebooks from org repos; stage under a scratch dir, not the corpus; record per-notebook source URL + license.
- Verify: `ls scratch/dogfood-wide/*.ipynb | wc -l` ≥ 60; `test -f scratch/dogfood-wide/SOURCES.tsv` with one row per notebook.

A2. Run the static detector per file — `cmd_check` serializes JSON only after the whole loop and returns 3 on any single-file exception, so a glob invocation aborts before printing: `for f in scratch/dogfood-wide/*.ipynb; do uv run wald check "$f" --format json >> scratch/dogfood-wide/flags.jsonl 2>> scratch/dogfood-wide/errors.log || echo "FAIL $f" >> scratch/dogfood-wide/errors.log; done`.
- Verify: `wc -l < scratch/dogfood-wide/flags.jsonl` = one line per non-crashing notebook; `grep -c FAIL scratch/dogfood-wide/errors.log` = crash count.

A3. Confirm flags with the same adversarial-verification review method that produced the 7-flaw baseline (`evals/2026-07-04-dogfood.md`) — not bare eyeballing; ledger confirmed flaws to `evals/real-flaws.tsv` (class, source repo, license, one-line why-real); aggregate-only, no per-notebook attribution in published artifacts.
- Verify: `awk -F'\t' 'NR>1' evals/real-flaws.tsv | wc -l` ≥ 30; if under 30, go to A4b.

A4a. ≥ 30 branch: write `evals/2026-07-DD-dogfood-wide.md` (confirmed count, files reviewed, per-class breakdown, license roster); update README lines 80–81 to the new N.
- Verify: `grep -n "confirmed" README.md` shows the new N and no "7 confirmed".

A4b. < 30 branch: state exactly "N confirmed real flaws of M notebooks reviewed; too few to report a recall rate".
- Verify: `grep -n "too few to report" README.md` matches; no unearned recall percentage.

**B. Temporal-shuffle evasion triage**

B1. Enumerate the 4 evasions (`git show 096606f -- wald/detect.py` + the FP-discipline comments: non-literal `shuffle=` treated clean; A1-only datetime with non-date sort suppressed; namespaced free-function lags excluded via `MODULE_ALIASES`; strong-identity-only `train_test_split` no-lag path) as named shapes with minimal trigger sketches in `docs/temporal-shuffle-evasions.md`.
- Verify: `grep -c "^## " docs/temporal-shuffle-evasions.md` → 4.

B2. Decide each: root fix (only if it adds no new clean-corpus FP; extend detector + add mutant) or accept (one-paragraph rationale with a `Decision:` line).
- Verify: `grep -c "^Decision:" docs/temporal-shuffle-evasions.md` → 4.

B3. Prove no regression: `uv run wald corpus build --root corpus && uv run pytest tests/test_gates.py -q && uv run wald eval --corpus corpus --out evals`.
- Verify: gates pass; eval shows `leakage-temporal-shuffle: precision 1.00 recall 1.00`, `clean FP rate: 0.0%`; `python3 -c "import json;d=json.load(open('corpus/MANIFEST.json'));print(sum(1 for m in d['mutants'] if m['flaw_id']=='leakage-temporal-shuffle'))"` ≥ 16.

**C. selection-survivorship-cohort decision**

C1. Test whether static-only promotion can hold precision: a scoped-claim survivorship notebook ("among active users…") is clean, and the static half at confidence 0.55 cannot read claim text.
- Verify: `uv run wald check <scoped-claim-example>.ipynb --floor 0.0` emits the candidate — static-alone cannot distinguish scoped from unscoped.

C2. Park it with reason and owner (narrative-layer fusion) in NEXT.md + a comment on the taxonomy entry in `wald/taxonomy/flaws.yaml`; keep the 16 mutants and candidate recall 1.00. (If C1 unexpectedly shows a precision-safe promotion, take the promote branch instead.)
- Verify: `grep -n "park" NEXT.md` matches; `grep -n "selection-survivorship-cohort: candidate recall 1.00" "$(ls -t evals/*-eval.md | head -1)"` matches.

**D. pyright hard CI gate**

D1. Confirm clean: `uvx pyright wald/` → `0 errors, 0 warnings, 0 informations`.
D2. Delete `continue-on-error: true` under the pyright step in `.github/workflows/ci.yml`.
- Verify: `grep -n "continue-on-error" .github/workflows/ci.yml` returns nothing; a pushed CI run shows the step gating.

**E. Ship the next release (owns dist/version/PyPI for the whole roadmap)**

E1. `rm dist/wald_lint-0.1.0*`. Verify: `ls dist/ | grep -c 0.1.0` → 0.
E2. After B/C/D and CLI-UX land, pick the version by rule: docs+CI+CLI-chrome only → `0.2.1`; any detector-behavior change or new mutant → `0.3.0`. Set `pyproject.toml`, write CHANGELOG (pyright gate, evasion dispositions, survivorship park, any detector change, CLI-UX changes). E does NOT block on workstream A.
- Verify: `grep -m1 version pyproject.toml` prints the chosen version; `git tag -l v<newversion>` empty (re-tagging 0.2.0 fails — already released); `uv run pytest tests/ -q` full pass.
E3. Confirm the trusted-publisher path: `release.yml` still uses OIDC (`id-token: write`, `pypa/gh-action-pypi-publish`), last run succeeded.
- Verify: `curl -s https://pypi.org/pypi/wald-lint/json -o /dev/null -w "%{http_code}"` → 200; `gh run list --workflow release.yml --limit 1` → `completed success`.
E4. Licensing/attribution: `corpus/real/LICENSES/` ships; no report attributes an individual notebook; A's ledger stays out of `corpus/real`.
- Verify: `ls corpus/real/LICENSES/ | wc -l` → 5; no published report names an individual source notebook.
E5. `git tag v<newversion> && git push origin v<newversion>`, then `gh release create v<newversion> --title "<newversion>" --notes-from-tag`.
- Verify: release run `completed success`; PyPI JSON reports the new version.
E6. Clean-venv install proof: `pip install wald-lint==<newversion> && wald --version && wald check examples/leaky.ipynb`.
- Verify: version prints; the known-flawed example exits 1 or 2 (not 3).

(If the LLM gates later pass, WS-F1a's README/CHANGELOG edit rides a subsequent tag through E2→E5 unchanged.)

**F. GitHub Action with PR annotations (M4 — unblocked, repo is public)**

F1. Composite `action.yml` (repo root or `.github/actions/wald/`): install wald-lint, run `wald check --format sarif`, upload SARIF; plus an example consumer workflow; reuse the `security-events: write` recipe in `docs/agent-gate.md`.
- Verify: `test -f action.yml`; `python3 -c "import yaml;yaml.safe_load(open('action.yml'))"` exits 0.
F2. Throwaway PR with a known-flawed notebook; confirm the finding surfaces as a PR annotation via SARIF code-scanning upload.
- Verify: `gh pr checks <n>` shows the wald job as expected; `gh api repos/Wake360/wald/code-scanning/alerts --jq 'length'` ≥ 1.

#### Dependencies

A2←A1, A3←A2, A4←A3. B2←B1, B3 gates B2. C2←C1. E2 ← B, C, D, CLI-UX. E5 ← E2, E3, E4. E6 ← E5. F ← E5. A does NOT block E. No task needs keys or the $150 budget.

#### Risks

Root fix in B breaks 1.00/1.00 or 0.0% → B3 rebuilds and runs gates pre-merge; a fix reintroducing a value-sorted FP is downgraded to accept. — Batch yields < 30 confirmed → A4b honest bound is first-class; pull one more GitHub batch before switching. — Release job fails at publish → E3 pre-confirms publisher; E2 verifies the tag is free. — Licensing/attribution exposure → E4 gates on LICENSES + aggregate-only. — Hard pyright breaks CI on a pyright bump → 0 errors now, matrix pinned, `uv.lock` pins deps. — M4 action green but no annotations → F2 verifies via the code-scanning alerts API.

#### Non-goals

Shared non-goals (§2). Static-specific: no G2/G3 gate runs (LLM track); no promoting survivorship by weakening precision (parking is the expected outcome); no M3 work or fifth flaw class.

#### Effort

A: L (the long pole — GitHub pull + adversarial hand-confirmation to ≥ 30). B: M. C: S. D: S. E: M. F: M.

---

### 3C. Track: CLI UX

#### Objective

Make `wald` feel like ruff/eslint in a terminal without touching the output contract automation depends on. Interactive md gains severity color, a dimmed CLEAN line, directory recursion, a roll-up line, and `--llm` progress — every one gated on `isatty` (color also on absence of `NO_COLOR`). The moment stdout is a pipe or file, emitted md/json/sarif bytes are exactly what 0.2.0 emits. Exit-code contract unchanged. This track owns `wald check --llm` progress.

Definition of done (every bullet a re-runnable boolean):
- `wald check examples/leaky.ipynb | cat` and `> f` produce bytes identical to a golden captured from unmodified HEAD before this track; byte-for-byte test.
- Same command on a TTY (isatty forced True) emits ANSI on severity headers and a dim CLEAN line; ANSI-present test.
- `wald check <dir>` lints every `*.ipynb` under `<dir>` (recursive, sorted, skipping `.ipynb_checkpoints`); empty dir exits 3 with exact stderr `wald: <dir>: no .ipynb files found`.
- Multi-file md run prints one roll-up line to stderr on a TTY, nothing extra to stdout ever; json/sarif never roll up.
- `--llm` prints per-file progress to stderr on a TTY, nothing to stdout — stdout cleanliness asserted with the progress path forced on.
- Held-out refusal (exit 3) still fires when the held-out notebook is reached via directory expansion.
- Every string edited by WS5 is pinned by an exact-match test; the AI-slop pass is a review gate.
- Each behavior tested in both TTY and piped variants; `uv run pytest -q` stays green (264 + new).

#### Workstreams

**WS0 — Byte-identity golden from unmodified HEAD (first commit)**

0.1. On clean HEAD (`git diff --quiet` exits 0, `git stash list` empty): `mkdir -p tests/golden && uv run python -c "import sys; from wald.cli import main; main(['check','examples/leaky.ipynb'])" > tests/golden/leaky.md`, commit it alone.
- Verify: `git show --stat HEAD` lists `tests/golden/leaky.md` and no other file.

**WS1 — TTY-aware markdown rendering**

Design: piped path stays `print(to_markdown(...))` verbatim. TTY decoration is a single post-pass `_colorize(text) -> str` in `report.py` wrapping `## HIGH:`/`## MEDIUM:` in severity color and `## CLEAN (checked):` in dim, raw ANSI (`\033[31m`/`\033[33m`/`\033[2m`/`\033[0m`), no new dependency. `cmd_check` computes `color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")` and routes only md through `_colorize`. json/sarif never colorized.

1.1. Add `_colorize`. Verify: `uv run python -c "from wald.report import _colorize; s='## HIGH: x\n## CLEAN (checked): a\nplain'; print(repr(_colorize(s)))"` → `\x1b[31m` before `## HIGH`, `\x1b[2m` before `## CLEAN`, `plain` unchanged.
1.2. Gate the md branch in `cmd_check` on `color`; json/sarif untouched.
1.3. Test all three variants: piped (no `\x1b[`), TTY (`\x1b[31m` present, CLEAN dim), NO_COLOR (isatty True + `NO_COLOR=1` → no `\x1b[`). Verify: `uv run pytest tests/test_cli.py -q -k color` → 3 passed.
1.4. Byte-identity vs the WS0 golden (piped run under capsys equals `tests/golden/leaky.md`). Verify: `uv run pytest tests/test_cli.py -q -k golden` → passed.

**WS2 — Directory recursion**

Design: `_expand_notebooks(paths)` at the top of `cmd_check`: each `Path.is_dir()` entry → `sorted(str(p) for p in path.rglob("*.ipynb"))` excluding any `.ipynb_checkpoints` component; non-dirs pass through; dir yielding zero → stderr `wald: <dir>: no .ipynb files found`, return 3.

2.1. Add and wire `_expand_notebooks`. Verify: `uv run python -c "from wald.cli import _expand_notebooks; print(_expand_notebooks(['examples']))"` → `['examples/leaky.ipynb']`, deterministic, no checkpoints.
2.2. Empty-dir guard exits 3 with the exact message. Verify: `-k empty_dir` asserts rc 3 and exact stderr string.
2.3. Held-out refusal survives directory expansion: tmp dir with a manifested held-out notebook + `MANIFEST.json` (mirror `test_heldout_refusal_has_prefix_and_exits_3`, tests/test_cli.py:125), stub non-api backends, `main(["check","--llm", str(tmp_dir)])`. Verify: `-k heldout_dir_refusal` asserts rc 3, stderr contains `held-out corpus notebook is gate-only`.
2.4. Pin (do not change) mid-run abort: one unreadable notebook aborts the invocation (`return 3` in the per-path handler, cli.py:122-124). Verify: `-k dir_abort` — good notebook sorts first, invalid-JSON second; rc 3, exactly one `# Wald report` on stdout.
2.5. Both piped variants of a valid dir run: two readable notebooks → both headers in sorted order, worst exit code; single explicit file byte-unchanged (WS1 golden). Verify: `-k dir_recurse` passed.

**WS3 — Multi-file roll-up**

Design: after the loop, when `args.format == "md" and len(notebooks) > 1 and sys.stdout.isatty()`, print to stderr: `checked N notebooks: H high, M medium, C clean`. Buckets from each file's worst confident `Flag.severity` (`confidence >= floor`), never from exit code — `exit_code` returns 2 for a genuinely-medium file under `--severity-gate medium` (report.py:27).

3.1. Accumulate per-file worst confident severity; emit under the gate. Verify: `-k rollup` — TTY 2-notebook dir (one high, one clean): stderr matches `^checked 2 notebooks: 1 high, 0 medium, 1 clean$`; piped: no `checked ` line, stdout byte-identical to concatenated reports.
3.2. Correct under `--severity-gate medium`: a worst-medium file lands in the medium bucket despite exit 2. Verify: `-k rollup_severity_gate` matches `^checked \d+ notebooks: 0 high, 1 medium`.
3.3. json/sarif never roll up, even on a TTY. Verify: same test, `--format json` with isatty True → no `checked ` on stderr.

**WS4 — `--llm` progress (owns `wald check --llm`)**

Design: when `--llm` and `sys.stderr.isatty()`: `checking i/N <name>` to stderr, leading `\r`, no trailing newline; single `\n` after the loop; on any early return (`_heldout_refusal` cli.py:104-105, `backend_error` cli.py:117-119) print `\n` first so the pending line is closed before the `wald: ...` error. No stdout writes; per-file only.

4.1. Add the gated print + loop-end `\n` + pre-error `\n` on both early-return paths. Verify: `-k progress` — TTY (monkeypatched, stub backends): stderr contains `checking 1/`; piped: no `checking`.
4.2. No stdout leak; overwrite not stack: progress region of stderr contains exactly one `\n`. Verify: `-k progress_stdout_clean` passed.

**WS5 — ux-writing / AI-slop pass**

Scope: `help=` strings, subparser help/epilog, `_input_error` messages, the `wald: --llm needs ...` / refusal / backend-failure / empty-dir messages, md report labels (`Where`, `Evidence`, `Flaw`, `Failure scenario`, `Fix`, `Confidence`, `verdict`, `Candidates`, `CLEAN (checked)`). Rule: cut filler, no marketing words, each word testable. json/sarif field names frozen.

5.1. Review help/error strings; the exit-code epilog wording docs/agent-gate.md quotes (`exit codes: 0 clean, 1 medium, 2 high, 3 input or usage error`, cli.py:194) stays intact. Verify (judgment gate): reviewer confirms `wald --help` / `wald check --help` carry no filler; every edited string enters 5.3's pins.
5.2. Review md report labels; change only if a shorter word is clearer and machine formats untouched. If a label changes, regenerate the WS0 golden in the same commit and review the diff. Verify: `uv run pytest tests/test_report.py tests/test_cli.py -q` green.
5.3. Pin wording with exact-match tests: `--help` contains the exact exit-code line; `no such file`, `is a directory, not a notebook`, the `wald: --llm needs ...` message shape (cli.py:94), the empty-dir line, plus everything 5.1/5.2 touched. Verify: `uv run pytest tests/test_cli.py -q -k "help or message"` passed.

#### Dependencies

WS0 is the first commit on unmodified HEAD; 1.4 and 2.5 assert against it; a WS5 label change regenerates the golden in its own commit. WS3/WS4 reuse WS1's isatty gating pattern. WS2 feeds WS3. No dependency on keys, budget, gates, or publishing. No new dependency (raw ANSI, stdlib only).

#### Risks

Color/roll-up bytes leak into piped output → color only on the md branch behind `isatty and not NO_COLOR`; roll-up/progress stderr-only, roll-up also gated on `--format md`; trigger = golden test fails or a json/sarif test sees `\x1b[` or `checked `. — Roll-up mislabels medium as high under `--severity-gate medium` → bucket from `Flag.severity`. — Glob order platform-dependent → `sorted(...)`. — Recursion descends into `.ipynb_checkpoints` → filtered. — Empty dir silently exits 0 → exit 3 pinned. — Directory expansion slips a held-out notebook past the `--llm` gate → Task 2.3. — One unreadable notebook aborts a large scan → pinned as current contract (2.4), continue-on-error out of scope. — Progress line collides with an error message → pre-error `\n`. — WS5 edit changes a machine-read label → md prose and CLI chrome only, every edit pinned. — `NO_COLOR` still gets ANSI → honored and tested.

#### Non-goals

Shared non-goals (§2). CLI-specific: no exit-code contract change; no change to single-file-abort; no colorized json/sarif; no progress bar/spinner/curses, no colorama/rich; no sub-notebook `--llm` progress; no Windows color layer.

#### Effort

WS0: S. WS1: M. WS2: M. WS3: S. WS4: S. WS5: M.

---

## 4. Done checklist

Flat, mechanically checkable.

CLI UX:
- [ ] WS0 golden committed alone on clean HEAD: `git show --stat <WS0-commit>` lists only `tests/golden/leaky.md`.
- [ ] Piped output byte-identical to golden: `uv run pytest tests/test_cli.py -q -k golden` passed.
- [ ] TTY color present, NO_COLOR honored: `-k color` → 3 passed.
- [ ] `_colorize` decorates only the header shapes (repl check above).
- [ ] Directory recursion sorted, checkpoint-skipping: `-k dir_recurse` passed; `_expand_notebooks(['examples'])` → `['examples/leaky.ipynb']`.
- [ ] Empty dir exits 3 with exact line: `-k empty_dir` passed.
- [ ] Held-out refusal via directory expansion: `-k heldout_dir_refusal` passed.
- [ ] Mid-run abort pinned: `-k dir_abort` passed.
- [ ] Roll-up on TTY, none piped: `-k rollup` passed.
- [ ] Roll-up correct under `--severity-gate medium`: `-k rollup_severity_gate` passed.
- [ ] json/sarif never roll up on a TTY (same test).
- [ ] `--llm` progress stderr-only with `\r`-overwrite: `-k "progress or progress_stdout_clean"` passed.
- [ ] WS5 strings pinned: `-k "help or message"` passed; `wald --help` exit-code line unchanged.
- [ ] Machine-format labels unchanged: `uv run pytest tests/test_report.py tests/test_cli.py -q` green.

Static layer:
- [ ] pyright clean: `uvx pyright wald/` → `0 errors, 0 warnings, 0 informations`.
- [ ] pyright hard gate: `grep -n "continue-on-error" .github/workflows/ci.yml` → nothing.
- [ ] 4 evasions enumerated: `grep -c "^## " docs/temporal-shuffle-evasions.md` → 4.
- [ ] 4 dispositions: `grep -c "^Decision:" docs/temporal-shuffle-evasions.md` → 4.
- [ ] No regression from root fixes: gates pass; eval shows temporal-shuffle 1.00/1.00, clean FP 0.0%; manifest mutant count ≥ 16.
- [ ] Survivorship promotion shown precision-unsafe (`--floor 0.0` on a scoped-claim example emits the candidate).
- [ ] Survivorship parked with owner: `grep -n "park" NEXT.md` matches; latest eval keeps `candidate recall 1.00`.
- [ ] Dogfood batch staged: ≥ 60 notebooks + SOURCES.tsv.
- [ ] Per-file run captured: flags.jsonl line count + FAIL count consistent.
- [ ] Recall number OR honest bound: `evals/real-flaws.tsv` ≥ 30 rows AND README shows new N; OR `grep "too few to report" README.md` matches, no unearned percentage.
- [ ] Stale dist gone: `ls dist/ | grep -c 0.1.0` → 0.
- [ ] Version chosen, tag free: `grep -m1 version pyproject.toml` → 0.2.1 or 0.3.0; `git tag -l v<new>` empty.
- [ ] Full suite green before tag: `uv run pytest tests/ -q` → ≥ 264 passed, 0 failed.
- [ ] Trusted publisher live: PyPI 200; release.yml OIDC lines present; last release run `completed success`.
- [ ] Licensing gate: repo PUBLIC; `ls corpus/real/LICENSES/ | wc -l` → 5; aggregate-only reporting.
- [ ] Release published: release run success; PyPI serves the new version.
- [ ] Wheel installs and runs: clean-venv `pip install wald-lint==<new>`; `wald --version` correct; flawed example exits 1 or 2.
- [ ] M4 Action valid YAML: `action.yml` exists and parses.
- [ ] PR annotations demonstrated: `gh api repos/Wake360/wald/code-scanning/alerts --jq 'length'` ≥ 1.

LLM layer:
- [ ] Key-guard test: `-k eval_llm_missing` → 1 passed; keyless invocation prints the exact message and exit 3.
- [ ] Eval-loop progress test: `-k progress` in test_narrative_eval → 1 passed.
- [ ] Partial-batch coverage: `-k continues_past_backend_error` → 1 passed.
- [ ] Failure-mode tests green: 5xx-retry, 429-Retry-After, both usage-accumulation tests.
- [ ] Fixture smoke records both providers: smoke detector/ and verifier/ dirs non-empty.
- [ ] Dev checkpoint meets the dev bar with usage + cost/latency recorded.
- [ ] Dev G3 read: four recipes, survival ≥ 0.75.
- [ ] Seam fixed (`-k output_region` passed, FP held, significance recall risen) OR documented (caveat present in rendered report, 2/8 cost recorded).
- [ ] Held-out preflight passes (dev bar, ≤ 1 prior report, keys, pinned-model grep).
- [ ] Held-out gate recorded ≤ 2 attempts: D2 script prints gate true, errs [], f1 ≥ 0.7 ×3, fp ≤ 0.10, dropped ≤ 0.10, 5 recipes, pooled kill ≥ 0.80, survival ≥ 0.75, non-null fp_caveat. (OR termination fired — §5.)
- [ ] Dev-vs-heldout gaps computed and honored; no post-held-out retuning.
- [ ] LLM dogfood: one JSON per `corpus/real/*.ipynb`; labeled real-recall/FP separate from G2.
- [ ] Spend ledger ≤ $45 in the runbook.
- [ ] Publish branch correct: numbers match the held-out JSON exactly; OR README carries the static-only outcome and no narrative claims.
- [ ] Pre-tag full test + static gate rebuild green.

## 5. Termination rule

If after M2 the narrative layer cannot hold recall ≥ 0.6 with FP ≤ 10% after two prompt/fusion iterations, v1 narrows to the static linter and ships with the honest eval report.

In practice: the two iterations are the ≤ 2 dev checkpoints (one of which a WS-C2a seam fix may consume) plus the ≤ 2 held-out attempts. STOP 1 blocks held-out spend if the dev bar is unmet; STOP 2/STOP 4 forbid retuning after seeing held-out and make attempt 2 final. When the rule fires, WS-F1b executes: v1 is the static linter, the corpus and harness survive, and README states plainly that the narrative layer did not clear G2/G3 in two iterations. If the two keys never arrive, the static-only outcome is the default v1 and Milestones 1+2 constitute the shipped product.
