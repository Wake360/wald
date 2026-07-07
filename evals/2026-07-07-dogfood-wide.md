# Dogfood report — 2026-07-07 (wide batch, first recall rate)

Second real-world contact, batch 1 of the widened dogfood. 60 fresh
notebooks pulled from GitHub (not the 5 teaching repos of the
2026-07-04 run), spread across five sourcing buckets. This run produces
wald's first defensible recall *rate* and its first out-of-sample
precision number. Aggregate only; no notebook is attributed to an author.
Ground truth is multi-agent adversarial review (one TP/FP reviewer per
flagged notebook, opus miss-hunters over a seeded sample, opus verifiers
that tried to refute every surviving claim), not expert human labeling —
treat it accordingly.

## Setup

- 60 notebooks, deduped against the 5 prior repos and the 27 in
  `corpus/real/` by code-cell sha256 (0 duplicates). Buckets: B1
  supervised-selection code (15), B2 Kaggle-style competition work (20),
  B3 temporal/time-series (10), B4 imbalanced-accuracy (5), B5 broad ML
  tutorials (10). Licenses: 23 permissive, 37 none/other; the recall
  number counts all of them (local run, aggregate report — no
  redistribution), corpus folding is a separate permissive-only step and
  is not done here.
- `wald check` over all 60 (per-file loop, JSON compacted): 0 crashes,
  0 parse failures.
- Confirmation: every flag reviewed TP/FP; a stratified random sample of
  30 of the 47 unflagged notebooks (seed 42, allocated across buckets)
  miss-hunted; every claimed TP and every claimed miss put through an
  adversarial verifier that defaulted to refuted unless the leak was
  airtight.

## Result

wald flagged 13 of 60 notebooks (28 flags). After review + adversarial
verification:

- **Precision on fresh data: 25 of 28 flags confirmed real (0.89);
  12 of 13 flagged notebooks contained a real leak (0.92).** Three flags
  did not survive: one full false-positive notebook (an RFECV
  scout-subsample pattern — `fit_transform` on an already-split
  `X_train` whose downstream `train_test_split` only carves a discarded
  feature-selection scout set, not the evaluation split), one target-side
  `y.fillna` mistaken for a feature transform, and one duplicate/borderline
  scaler flag. This is the honest counterpart to the corpus's 0.0% FP:
  on curated clean notebooks wald is clean, on messy real code it is not
  perfect. The RFECV scout-split is a new FP idiom, logged for the
  evasions doc; not fixed in 0.2.1 (no detector change this release).

- **Recall: 12 of 20 leaky notebooks caught over the 43-notebook audited
  subset (0.60).** The audited subset is the 13 flagged plus the 30-notebook
  miss-hunt sample. Miss-hunting found 8 real leaks wald missed, in 8
  distinct notebooks. This 0.60 is over the audited subset at seed 42;
  the 17 unflagged notebooks not sampled may hide further misses, so true
  recall over the full 60 is likely lower, not higher.

- **33 confirmed leak instances across 20 notebooks** (instance count, the
  same unit as the prior "7 confirmed"). Distribution is concentrated:

  | class | instances | notebooks |
  |---|---|---|
  | leakage-fit-before-split | 28 | 17 |
  | leakage-temporal-shuffle | 5 | 4 |
  | testing-multiple-uncorrected | 0 | 0 |
  | baserate-accuracy-imbalanced | 0 | 0 |

  One notebook contributes 8 of the 28 fit-before-split instances (eight
  per-column scaler fits on full data in a single cell — one idiom, eight
  sites). The recall evidence is entirely about the two leakage classes;
  this sample produced zero confirmed real instances of the two
  testing/base-rate classes, so no real-world recall is claimed for them.

## The 8 confirmed misses (recall gaps)

All are leaks with no name-level syntactic anchor the static layer keys
on, matching the known blind spots:

- groupby-median / per-group `fillna` imputation before the split
- `scaler.fit` / `fit_transform` on full data feeding `cross_val_score`
- `fillna(df[col].mean())` column imputation before the split
- `TfidfVectorizer` / `TruncatedSVD` fit on the full text corpus
- `get_dummies` one-hot key leakage (contested, survived refutation)
- two temporal-shuffle misses: `StratifiedKFold`/`cross_val_score` on
  time-ordered features, and a demand-forecast train helper

These are M2/M3 territory (narrative fusion or new dataflow), not more
static heuristics — consistent with the prior report's stance.

## Decision gate

d1 (instance density) = 33/60 = 0.55; notebook density 20/60 = 0.33.
Target of ≥30 confirmed instances met in batch 1, and — more useful — a
real recall rate exists, so the honest-bound branch does not fire and no
batch 2 is pulled. Cap was 3 batches / 180 notebooks; used 1 / 60.

## Ledger

`evals/real-flaws.tsv`: 33 rows, one per confirmed instance
(notebook_id, class, license, kind, why_real). Notebook ids are
anonymized; the id→repo map is retained locally only, not committed, so
the published artifact stays aggregate and names no author.

## Honest caveats

- Ground truth is LLM adversarial review, not expert human labeling. The
  clear cases (full-data fit → split → metric) were unambiguous; the
  contested one (`get_dummies`) is flagged as such.
- Recall 0.60 is a subset estimate at one seed, an upper-leaning one.
- Evidence covers leakage only; the testing and base-rate classes are
  unmeasured on real data.
- Sources skew toward small personal/competition repos; still not
  production pipelines.
