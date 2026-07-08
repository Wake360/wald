# LLM narrative dogfood — subscription backends (2026-07-08)

**NOT GATE EVIDENCE. NOT REPRODUCIBLE.** Produced by the claude/codex
subscription backends (`--llm-subscription`), which route to unpinned
subscription models. These numbers cannot be reproduced and must never be
folded into G2/G3 or published to README (see `plans/g2g3-runbook.md` STOP 3).
Each flag requires hand review before it means anything; a zero counts only
if the quote grounded, so weigh backend errors (excluded, not clean).

- Notebooks: 27 (manifest-free copy of `corpus/real/*`)
- Notebooks with >=1 flag: 1
- Narrative-derived flags: 2
- Backend errors (excluded, not clean): 0 none

## Flags by class

- `leakage-fit-before-split`: 2

## Per-notebook

| notebook | secs | flags | narrative | backend_err |
|---|---|---|---|---|
| 00__ageron_handson-ml3__03_classification.ipynb | 209.7 | 2 | 2 |  |
| 01__rasbt_python-machine-learning-book-3rd-edition__ch07.ipynb | 0.0 | 0 | 0 |  |
| 02__ageron_handson-ml3__02_end_to_end_machine_learning_project.ipynb | 0.0 | 0 | 0 |  |
| 03__rasbt_python-machine-learning-book-3rd-edition__ch06.ipynb | 0.0 | 0 | 0 |  |
| 04__ageron_handson-ml3__07_ensemble_learning_and_random_forests.ipynb | 0.0 | 0 | 0 |  |
| 05__rasbt_python-machine-learning-book-3rd-edition__ch08.ipynb | 0.0 | 0 | 0 |  |
| 06__rasbt_python-machine-learning-book-3rd-edition__ch04.ipynb | 0.0 | 0 | 0 |  |
| 09__ageron_handson-ml3__05_support_vector_machines.ipynb | 160.4 | 0 | 0 |  |
| 11__rasbt_python-machine-learning-book-3rd-edition__ch03.ipynb | 64.1 | 0 | 0 |  |
| 13__ageron_handson-ml3__04_training_linear_models.ipynb | 232.2 | 0 | 0 |  |
| 14__ageron_handson-ml3__08_dimensionality_reduction.ipynb | 168.9 | 0 | 0 |  |
| 15__ageron_handson-ml3__06_decision_trees.ipynb | 90.7 | 0 | 0 |  |
| 16__dipanjanS_practical-machine-learning-with-python__Building, Tuning and Deploying Models.ipynb | 71.9 | 0 | 0 |  |
| 17__dipanjanS_practical-machine-learning-with-python__Predictive Analytics.ipynb | 121.2 | 0 | 0 |  |
| 18__rasbt_python-machine-learning-book-3rd-edition__ch05.ipynb | 61.6 | 0 | 0 |  |
| 19__rasbt_python-machine-learning-book-3rd-edition__ch10.ipynb | 89.1 | 0 | 0 |  |
| 20__ageron_handson-ml3__09_unsupervised_learning.ipynb | 191.7 | 0 | 0 |  |
| 21__dipanjanS_practical-machine-learning-with-python__decision_tree_regression.ipynb | 121.7 | 0 | 0 |  |
| 22__jakevdp_PythonDataScienceHandbook__05.03-Hyperparameters-and-Model-Validation.ipynb | 70.1 | 0 | 0 |  |
| 24__WillKoehrsen_Data-Analysis__logistic-regression-basics.ipynb | 122.7 | 0 | 0 |  |
| 25__WillKoehrsen_Data-Analysis__Improving Random Forest Part 2.ipynb | 115.2 | 0 | 0 |  |
| 26__ageron_handson-ml3__13_loading_and_preprocessing_data.ipynb | 77.6 | 0 | 0 |  |
| 27__ageron_handson-ml3__12_custom_models_and_training_with_tensorflow.ipynb | 149.7 | 0 | 0 |  |
| 28__dipanjanS_practical-machine-learning-with-python__linear_regression.ipynb | 75.9 | 0 | 0 |  |
| 29__dipanjanS_practical-machine-learning-with-python__Predicting Student Recommendation Machine Learning Pipeline.ipynb | 62.1 | 0 | 0 |  |
| 31__jakevdp_PythonDataScienceHandbook__05.13-Kernel-Density-Estimation.ipynb | 94.9 | 0 | 0 |  |
| 32__jakevdp_PythonDataScienceHandbook__05.14-Image-Features.ipynb | 93.2 | 0 | 0 |  |

## Flag detail (hand-review these)

- **00__ageron_handson-ml3__03_classification.ipynb** — `leakage-fit-before-split` (high, conf 0.88, narrative=True)
  - verifier: The code fits the preprocessing pipeline on all of train_data before cross_val_score creates CV folds, so each fold’s held-out rows were included when the transformers were fitted.
- **00__ageron_handson-ml3__03_classification.ipynb** — `leakage-fit-before-split` (high, conf 0.88, narrative=True)
  - verifier: The evidence says the preprocessing pipeline was fit on all of X_train before cross_val_score, so each CV fold's held-out test rows were included in the transformer fit.

## Adjudication (hand review, 2026-07-08)

Both flags on `00__ageron_handson-ml3__03_classification.ipynb` are **true
positives** — real fit-before-split leakage, not the known fit-before-split FP
pattern (`docs/fit-before-split-fps.md`).

- **Titanic (cell 186 → 194):** `X_train = preprocess_pipeline.fit_transform(train_data)`
  then `cross_val_score(forest_clf, X_train, y_train, cv=10)`. The pipeline is
  `SimpleImputer(strategy="median")` + scaler + `OneHotEncoder` — all stateful.
  Medians, scaling stats, and one-hot categories are learned from the full
  training set, so every CV fold's held-out rows were in the fit. TP; magnitude
  mild (median imputation / scaling leak little).
- **Spam (cell 258 → 259):** `preprocess_pipeline.fit_transform(X_train)` where
  the pipeline ends in `WordCounterToVectorTransformer`, which learns a
  **vocabulary** in `fit` (a vectorizer/feature-selection step). Built on the
  full training set before `cross_val_score`, so held-out emails shaped the
  feature space. TP, and more material than the Titanic case.

Not the FP idiom: Géron calls `.fit_transform(...)` and passes the materialized
transformed array into CV (transformers never refit per fold), rather than
handing an unfitted `Pipeline` to `cross_val_score`. Result stands: 1/27 real
notebooks flagged, and the catch holds up on review. Non-gate, non-reproducible
as labeled above.
