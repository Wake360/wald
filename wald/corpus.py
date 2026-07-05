"""Corpus builder: 6 clean analysis families x dev + held-out seeds, plus
verified mutants and seeded false flags (G3 substrate).

Clean notebooks satisfy the clean-corpus criteria (see README): scoped
claims, split-before-fit, corrected or <=3 tests, >=2 classification
metrics, imbalance stated, no extrapolation. They are executed at build
time so stored outputs (value_counts etc.) are available as detector
evidence. Synthetic by design in v1 — licensed real notebooks are a
planned corpus extension, and the eval report says so.

Every manifest entry carries a split: dev (base seeds 11-14, mutation
seeds 0-1, phrasing variants 0-1) or heldout (fresh base seeds 21-24,
mutation seeds 2-3, phrasing variants 2-4). Nothing held-out may be seen
during prompt iteration (risk R3).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import nbformat

from . import execute as ex
from .mutate import MUTATIONS


def _nb(cells, wald_meta):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata["wald"] = wald_meta
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def _md(text):
    return nbformat.v4.new_markdown_cell(text)


def _code(text):
    return nbformat.v4.new_code_cell(text)


def _normalize(nb, stem: str):
    """Make serialization byte-deterministic across rebuilds (T15): derive
    cell ids from the notebook stem + position (nbformat otherwise mints a
    fresh random id per cell every build) and drop nbclient's per-cell
    execution timestamps, which are transient and regenerated on each run."""
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"{stem}-{i}"
        cell.get("metadata", {}).pop("execution", None)
    return nb


CHURN_COLS = [
    "tenure_months", "monthly_spend", "support_tickets", "logins_per_week",
    "discount_share", "emails_opened", "pages_per_session",
    "days_since_last_order", "referrals", "cart_abandons",
]


def churn_notebook(seed: int):
    cells = [
        _md("# Churn model — activity features\n\n"
            "Predict churn from customer activity. Balanced classes; both "
            "accuracy and AUC reported."),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import accuracy_score, roc_auc_score"
        ),
        _code(
            f"rng = np.random.default_rng({seed})\n"
            "n = 1500\n"
            "df = pd.DataFrame({\n"
            '    "tenure_months": rng.gamma(4, 6, n).round(1),\n'
            '    "monthly_spend": rng.normal(52, 15, n).round(2),\n'
            '    "support_tickets": rng.poisson(1.4, n),\n'
            '    "logins_per_week": rng.gamma(2.5, 1.8, n).round(1),\n'
            '    "discount_share": rng.beta(2, 6, n).round(3),\n'
            '    "emails_opened": rng.poisson(5, n),\n'
            '    "pages_per_session": rng.gamma(3, 1.2, n).round(1),\n'
            '    "days_since_last_order": rng.gamma(3, 9, n).round(0),\n'
            '    "referrals": rng.poisson(0.6, n),\n'
            '    "cart_abandons": rng.poisson(2.1, n),\n'
            "})\n"
            'df["monthly_spend_q2"] = (0.5 * df["monthly_spend"] + 26 + rng.normal(0, 13, n)).round(2)\n'
            'logit = (-0.05 * df["tenure_months"] - 0.4 * df["logins_per_week"]\n'
            '         + 0.03 * df["days_since_last_order"] + 0.35 * df["support_tickets"]\n'
            "         + rng.normal(0, 1.5, n))\n"
            'df["churned"] = (logit > np.median(logit)).astype(int)'
        ),
        _code('df["churned"].value_counts(normalize=True)'),
        _code(
            f"num_cols = {CHURN_COLS!r}\n"
            "X = df[num_cols]\n"
            'y = df["churned"]'
        ),
        _code(
            "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, "
            f"random_state={seed}, stratify=y)\n"
            "scaler = StandardScaler()\n"
            "X_tr = scaler.fit_transform(X_tr)\n"
            "X_te = scaler.transform(X_te)"
        ),
        _code(
            "model = LogisticRegression(max_iter=1000)\n"
            "model.fit(X_tr, y_tr)\n"
            "pred = model.predict(X_te)\n"
            "proba = model.predict_proba(X_te)[:, 1]\n"
            "acc = accuracy_score(y_te, pred)\n"
            "auc = roc_auc_score(y_te, proba)\n"
            'print(f"accuracy={acc:.3f}  roc_auc={auc:.3f}")'
        ),
        _md("On the held-out 25% test set the model reaches the accuracy and "
            "AUC printed above. Classes are close to balanced (see the "
            "value_counts cell), so accuracy is meaningful alongside AUC. "
            "These results describe customers in this dataset only."),
    ]
    return _nb(cells, {
        "family": "churn", "seed": seed, "num_cols": CHURN_COLS,
        "binary_col": "churned", "binary_values": [1, 0],
        "period_cols": ["monthly_spend", "monthly_spend_q2"],
        "imports_cell": 1, "split_cell": 5, "metrics_cell": 6,
        "conclusion_cell": 7, "imbalanced": False,
    })


AB_COLS = [
    "session_minutes", "pages_viewed", "scroll_depth", "clicks",
    "search_uses", "cart_adds", "support_chats", "video_plays",
    "filters_used", "wishlist_adds",
]


def abtest_notebook(seed: int):
    cells = [
        _md("# A/B test — checkout redesign\n\n"
            "One pre-registered primary metric (session_minutes); secondary "
            "metrics are listed descriptively, not tested."),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "from scipy.stats import ttest_ind"
        ),
        _code(
            f"rng = np.random.default_rng({seed})\n"
            "n = 2400\n"
            'variant = np.where(rng.random(n) < 0.5, "A", "B")\n'
            "df = pd.DataFrame({\n"
            '    "variant": variant,\n'
            '    "session_minutes": rng.gamma(3, 2.2, n) + (variant == "B") * 0.5,\n'
            '    "pages_viewed": rng.poisson(6, n),\n'
            '    "scroll_depth": rng.beta(3, 2, n).round(3),\n'
            '    "clicks": rng.poisson(11, n),\n'
            '    "search_uses": rng.poisson(1.5, n),\n'
            '    "cart_adds": rng.poisson(1.1, n),\n'
            '    "support_chats": rng.poisson(0.2, n),\n'
            '    "video_plays": rng.poisson(0.8, n),\n'
            '    "filters_used": rng.poisson(2.3, n),\n'
            '    "wishlist_adds": rng.poisson(0.5, n),\n'
            "})"
        ),
        _code(
            'a = df[df["variant"] == "B"]["session_minutes"]\n'
            'b = df[df["variant"] == "A"]["session_minutes"]\n'
            "stat, p = ttest_ind(a, b)\n"
            'pooled_sd = df["session_minutes"].std()\n'
            "d = (a.mean() - b.mean()) / pooled_sd\n"
            'print(f"primary metric session_minutes: p={p:.4f}, '
            'effect size d={d:.2f}, lift={a.mean() - b.mean():.2f} min")'
        ),
        _md("The primary metric was chosen before the experiment; we report "
            "the effect size alongside the p-value. Secondary metrics were "
            "not tested — any pattern in them is exploratory and would need "
            "a corrected follow-up."),
    ]
    return _nb(cells, {
        "family": "abtest", "seed": seed, "num_cols": AB_COLS,
        "binary_col": "variant", "binary_values": ["B", "A"],
        "imports_cell": 1, "datagen_cell": 2, "conclusion_cell": 4,
        "imbalanced": False,
    })


FRAUD_COLS = [
    "amount", "merchant_risk", "hour_of_day", "card_age_days",
    "tx_last_24h", "distance_from_home", "failed_auths",
    "new_device", "amount_vs_avg", "country_risk",
]


def fraud_notebook(seed: int):
    cells = [
        _md("# Fraud screening model\n\n"
            "Heavily imbalanced target (~10% fraud) — class balance is "
            "reported and AUC accompanies accuracy."),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import accuracy_score, roc_auc_score"
        ),
        _code(
            f"rng = np.random.default_rng({seed})\n"
            "n = 3000\n"
            "df = pd.DataFrame({\n"
            '    "amount": rng.gamma(2, 40, n).round(2),\n'
            '    "merchant_risk": rng.beta(2, 5, n).round(3),\n'
            '    "hour_of_day": rng.integers(0, 24, n),\n'
            '    "card_age_days": rng.gamma(4, 200, n).round(0),\n'
            '    "tx_last_24h": rng.poisson(2.2, n),\n'
            '    "distance_from_home": rng.gamma(1.5, 30, n).round(1),\n'
            '    "failed_auths": rng.poisson(0.3, n),\n'
            '    "new_device": rng.integers(0, 2, n),\n'
            '    "amount_vs_avg": rng.normal(1, 0.6, n).round(2),\n'
            '    "country_risk": rng.beta(1.5, 8, n).round(3),\n'
            "})\n"
            'score = (0.004 * df["amount"] + 0.8 * df["merchant_risk"]\n'
            '         + 0.5 * df["failed_auths"] + rng.normal(0, 1.6, n))\n'
            'df["fraud"] = (score > np.quantile(score, 0.9)).astype(int)'
        ),
        _code('df["fraud"].value_counts(normalize=True)'),
        _code(
            f"num_cols = {FRAUD_COLS!r}\n"
            "X = df[num_cols]\n"
            'y = df["fraud"]'
        ),
        _code(
            "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, "
            f"random_state={seed}, stratify=y)\n"
            "scaler = StandardScaler()\n"
            "X_tr = scaler.fit_transform(X_tr)\n"
            "X_te = scaler.transform(X_te)"
        ),
        _code(
            "model = LogisticRegression(max_iter=1000)\n"
            "model.fit(X_tr, y_tr)\n"
            "pred = model.predict(X_te)\n"
            "proba = model.predict_proba(X_te)[:, 1]\n"
            "acc = accuracy_score(y_te, pred)\n"
            "auc = roc_auc_score(y_te, proba)\n"
            'print(f"accuracy={acc:.3f}  roc_auc={auc:.3f}")'
        ),
        _md("Fraud is ~10% of transactions, so accuracy alone would be "
            "misleading (a majority-class predictor scores ~90%). Model "
            "quality is judged by AUC; accuracy is shown for context only."),
    ]
    return _nb(cells, {
        "family": "fraud", "seed": seed, "num_cols": FRAUD_COLS,
        "binary_col": "fraud", "binary_values": [1, 0],
        "imports_cell": 1, "split_cell": 5, "metrics_cell": 6,
        "conclusion_cell": 7, "imbalanced": True,
    })


HOUSING_COLS = [
    "sqm", "rooms", "age_years", "distance_center_km", "floor",
    "balcony_sqm", "energy_rating", "renovation_score",
    "noise_index", "green_share",
]


def housing_notebook(seed: int):
    cells = [
        _md("# Apartment price model\n\n"
            "Ridge regression on listing features; R^2 and MAE on a held-out "
            "split. Predictions are only made inside the observed feature "
            "range."),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.metrics import r2_score, mean_absolute_error"
        ),
        _code(
            f"rng = np.random.default_rng({seed})\n"
            "n = 1800\n"
            "df = pd.DataFrame({\n"
            '    "sqm": rng.gamma(6, 12, n).round(0),\n'
            '    "rooms": rng.integers(1, 6, n),\n'
            '    "age_years": rng.gamma(3, 12, n).round(0),\n'
            '    "distance_center_km": rng.gamma(2, 4, n).round(1),\n'
            '    "floor": rng.integers(0, 12, n),\n'
            '    "balcony_sqm": rng.gamma(1.2, 3, n).round(1),\n'
            '    "energy_rating": rng.integers(1, 8, n),\n'
            '    "renovation_score": rng.beta(2, 2, n).round(2),\n'
            '    "noise_index": rng.beta(2, 3, n).round(2),\n'
            '    "green_share": rng.beta(2, 2, n).round(2),\n'
            "})\n"
            'df["price"] = (df["sqm"] * 2100 - df["age_years"] * 800\n'
            '               - df["distance_center_km"] * 3500\n'
            '               + df["renovation_score"] * 40000\n'
            "               + rng.normal(0, 25000, n)).round(0)"
        ),
        _code(
            f"num_cols = {HOUSING_COLS!r}\n"
            "X = df[num_cols]\n"
            'y = df["price"]'
        ),
        _code(
            "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, "
            f"random_state={seed})\n"
            "scaler = StandardScaler()\n"
            "X_tr = scaler.fit_transform(X_tr)\n"
            "X_te = scaler.transform(X_te)"
        ),
        _code(
            "model = Ridge(alpha=1.0)\n"
            "model.fit(X_tr, y_tr)\n"
            "pred = model.predict(X_te)\n"
            "r2 = r2_score(y_te, pred)\n"
            "mae = mean_absolute_error(y_te, pred)\n"
            'print(f"r2={r2:.3f}  mae={mae:,.0f}")'
        ),
        _md("Model fit is evaluated on the held-out quarter of listings. The "
            "model describes prices within the observed range of features; "
            "we make no claims about segments outside it."),
    ]
    return _nb(cells, {
        "family": "housing", "seed": seed, "num_cols": HOUSING_COLS,
        "target_col": "price",
        "imports_cell": 1, "split_cell": 4, "metrics_cell": 5,
        "conclusion_cell": 6, "imbalanced": False,
    })


COHORT_COLS = [
    "ltv", "orders", "tenure_days", "aov", "sessions",
    "tickets", "emails_opened", "referrals", "returns", "coupon_uses",
]


def cohort_notebook(seed: int):
    cells = [
        _md("# Cohort LTV\n\n"
            "Realized lifetime value by signup cohort, computed on the full "
            "cohort including churned customers."),
        _code(
            "import numpy as np\n"
            "import pandas as pd"
        ),
        _code(
            f"rng = np.random.default_rng({seed})\n"
            "n = 4000\n"
            'cohorts = rng.choice(["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4",'
            ' "2025-Q1", "2025-Q2"], n)\n'
            "churn_p = 0.35\n"
            "df = pd.DataFrame({\n"
            '    "signup_cohort": cohorts,\n'
            '    "status": np.where(rng.random(n) < churn_p, "churned", "active"),\n'
            '    "ltv": rng.gamma(2, 900, n).round(0),\n'
            '    "orders": rng.poisson(6, n),\n'
            '    "tenure_days": rng.gamma(3, 90, n).round(0),\n'
            '    "aov": rng.gamma(4, 30, n).round(2),\n'
            '    "sessions": rng.poisson(30, n),\n'
            '    "tickets": rng.poisson(1.1, n),\n'
            '    "emails_opened": rng.poisson(9, n),\n'
            '    "referrals": rng.poisson(0.4, n),\n'
            '    "returns": rng.poisson(0.9, n),\n'
            '    "coupon_uses": rng.poisson(2.5, n),\n'
            "})\n"
            'df.loc[df["status"] == "churned", "ltv"] *= 0.6'
        ),
        _code(
            'cohort_ltv = df.groupby("signup_cohort")["ltv"].mean().round(0)\n'
            "cohort_ltv"
        ),
        _code(
            'retained = df[df["status"] == "active"]\n'
            'retained_ltv = retained.groupby("signup_cohort")["ltv"].mean().round(0)\n'
            "retained_ltv"
        ),
        _md("Realized average LTV per cohort (all customers, including "
            "churned) is shown first. The second table is scoped to "
            "currently retained customers only — it is higher by "
            "construction, because it excludes churned customers' lower "
            "realized LTV, and must not be read as a population number."),
    ]
    return _nb(cells, {
        "family": "cohort", "seed": seed, "num_cols": COHORT_COLS,
        "binary_col": "status", "binary_values": ["active", "churned"],
        "status_col": "status",
        "imports_cell": 1, "agg_cell": 3, "conclusion_cell": 5,
        "imbalanced": False,
    })


PROGRAM_COLS = [
    "spend_q1", "logins_per_week", "support_tickets", "emails_opened",
    "tenure_months", "pages_per_session", "referrals", "cart_abandons",
]


def program_notebook(seed: int):
    cells = [
        _md("# Retention program evaluation\n\n"
            "Effect of the outreach program on extreme-spend accounts, "
            "measured against an equally extreme control group; the churn "
            "model is cross-validated with preprocessing inside the "
            "pipeline."),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.model_selection import cross_val_score\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.linear_model import LogisticRegression"
        ),
        _code(
            f"rng = np.random.default_rng({seed})\n"
            "n = 2000\n"
            "df = pd.DataFrame({\n"
            '    "spend_q1": rng.normal(48, 14, n).round(2),\n'
            '    "logins_per_week": rng.gamma(2.5, 1.8, n).round(1),\n'
            '    "support_tickets": rng.poisson(1.4, n),\n'
            '    "emails_opened": rng.poisson(5, n),\n'
            '    "tenure_months": rng.gamma(4, 6, n).round(1),\n'
            '    "pages_per_session": rng.gamma(3, 1.2, n).round(1),\n'
            '    "referrals": rng.poisson(0.6, n),\n'
            '    "cart_abandons": rng.poisson(2.1, n),\n'
            "})\n"
            'df["spend_q2"] = (0.5 * df["spend_q1"] + 24 + rng.normal(0, 12, n)).round(2)\n'
            'logit = (-0.06 * df["tenure_months"] - 0.35 * df["logins_per_week"]\n'
            '         + 0.3 * df["support_tickets"] + rng.normal(0, 1.5, n))\n'
            'df["churned"] = (logit > np.median(logit)).astype(int)'
        ),
        _code(
            'low = df.nsmallest(300, "spend_q1")\n'
            "enrolled = low.sample(150, random_state=0)\n"
            "control = low.drop(enrolled.index)\n"
            'print("enrolled q1->q2:", enrolled["spend_q1"].mean().round(1),\n'
            '      "->", enrolled["spend_q2"].mean().round(1))\n'
            'print("control  q1->q2:", control["spend_q1"].mean().round(1),\n'
            '      "->", control["spend_q2"].mean().round(1))'
        ),
        _md("The enrolled accounts' spend recovered strongly in the second "
            "quarter. So did the control group's — both groups were selected "
            "for extreme values, and regression to the mean moves both back "
            "toward the average. The program effect is the enrolled-minus-"
            "control difference in that movement, which is close to zero "
            "here."),
        _code(
            f"num_cols = {PROGRAM_COLS!r}\n"
            "X = df[num_cols]\n"
            'y = df["churned"]\n'
            'pipe = Pipeline([("scaler", StandardScaler()),\n'
            '                 ("model", LogisticRegression(max_iter=1000))])\n'
            "scores = cross_val_score(pipe, X, y, cv=5)\n"
            'print(f"cv accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")'
        ),
        _md("Cross-validated accuracy estimates how the model will perform "
            "on unseen customers. The scaler sits inside the pipeline, so it "
            "is refit on each training fold and never sees that fold's test "
            "rows."),
    ]
    return _nb(cells, {
        "family": "program", "seed": seed, "num_cols": PROGRAM_COLS,
        "binary_col": "churned", "binary_values": [1, 0],
        "imports_cell": 1, "conclusion_cell": 6, "imbalanced": False,
    })


def forecast_notebook(seed: int):
    cells = [
        _md("# Demand forecast — temporal validation\n\n"
            "Predicting daily demand from lagged history. Validation here is "
            "temporal: metrics come from forecasting later days using only "
            "information from earlier days, never the reverse."),
        _code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score\n"
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.metrics import mean_absolute_error, r2_score"
        ),
        _code(
            f"rng = np.random.default_rng({seed})\n"
            "dates = pd.date_range('2024-01-01', periods=730, freq='D')\n"
            "n = len(dates)\n"
            "trend = np.linspace(100, 160, n)\n"
            "weekly = 15 * np.sin(2 * np.pi * np.arange(n) / 7)\n"
            "noise = rng.normal(0, 5, n)\n"
            "ar = np.zeros(n)\n"
            "for t in range(1, n):\n"
            "    ar[t] = 0.6 * ar[t - 1] + noise[t]\n"
            "demand = trend + weekly + ar\n"
            "df = pd.DataFrame({'date': dates, 'demand': demand.round(1)})"
        ),
        _code(
            "df['demand_lag1'] = df['demand'].shift(1)\n"
            "df['demand_lag7'] = df['demand'].shift(7)\n"
            "df['demand_ma14'] = df['demand'].rolling(14).mean()\n"
            "df['dow'] = df['date'].dt.dayofweek\n"
            "df = df.dropna().reset_index(drop=True)"
        ),
        _code(
            "feature_cols = ['demand_lag1', 'demand_lag7', 'demand_ma14', 'dow']\n"
            "X = df[feature_cols]\n"
            "y = df['demand']\n"
            "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, shuffle=False)\n"
            "model = Ridge(alpha=1.0)\n"
            "model.fit(X_tr, y_tr)\n"
            "pred = model.predict(X_te)\n"
            "mae = mean_absolute_error(y_te, pred)\n"
            "r2 = r2_score(y_te, pred)\n"
            'print(f"mae={mae:.3f}  r2={r2:.3f}")'
        ),
        _code(
            "cv = TimeSeriesSplit(n_splits=5)\n"
            "scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=cv, "
            "scoring='neg_mean_absolute_error')\n"
            'print(f"cv neg_mae: {scores.mean():.3f} +/- {scores.std():.3f}")'
        ),
        _md("Both the held-out quarter and the walk-forward cross-validation "
            "scores describe the model's ability to forecast future demand "
            "from past observations only; because the split respects time "
            "order, these numbers are not inflated by peeking at the "
            "future."),
    ]
    return _nb(cells, {
        "family": "forecast", "seed": seed, "date_col": "date",
        "temporal_split_cell": 4, "temporal_cv_cell": 5,
        "imports_cell": 1, "conclusion_cell": 6, "imbalanced": False,
    })


FAMILIES = {
    "churn": churn_notebook,
    "abtest": abtest_notebook,
    "fraud": fraud_notebook,
    "housing": housing_notebook,
    "cohort": cohort_notebook,
    "program": program_notebook,
    "forecast": forecast_notebook,
}

DEV_SEEDS = (11, 12, 13, 14)
HELDOUT_SEEDS = (21, 22, 23, 24)

# which mutation seeds to run per flaw and split; seeds 0-1 select dev
# phrasing variants, 2-3 select held-out ones (see mutate.phrasing_variant)
MUTATION_SEEDS = {
    "leakage-fit-before-split": {"dev": (0,), "heldout": (2,)},
    "testing-multiple-uncorrected": {"dev": (0, 1), "heldout": (2, 3)},
    "baserate-accuracy-imbalanced": {"dev": (0,), "heldout": (2,)},
    "selection-survivorship-cohort": {"dev": (0, 1), "heldout": (2, 3)},
    "significance-meaningless": {"dev": (0, 1), "heldout": (2, 3)},
    "regression-to-mean-claim": {"dev": (0, 1), "heldout": (2, 3)},
    "leakage-temporal-shuffle": {"dev": (0, 1), "heldout": (2, 3)},
}


def build_clean(root: Path, seeds, split: str, log=print) -> list[dict]:
    clean_dir = root / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for family, builder in FAMILIES.items():
        for seed in seeds:
            name = f"{family}-s{seed}.ipynb"
            log(f"clean: {name}")
            executed = ex.execute(builder(seed))
            _normalize(executed, Path(name).stem)
            nbformat.write(executed, str(clean_dir / name))
            entries.append({"file": f"clean/{name}", "family": family,
                            "seed": seed, "split": split})
    return entries


def build_mutants(root: Path, clean_entries, log=print) -> tuple[list[dict], list[dict]]:
    mut_dir = root / "mutated"
    mut_dir.mkdir(parents=True, exist_ok=True)
    entries, discarded = [], []
    for clean_entry in clean_entries:
        path = root / clean_entry["file"]
        nb = nbformat.read(str(path), as_version=4)
        for mutation in MUTATIONS:
            if not mutation.applicable(nb):
                continue
            for seed in MUTATION_SEEDS[mutation.flaw_id][clean_entry["split"]]:
                name = f"{path.stem}__{mutation.flaw_id}__m{seed}.ipynb"
                log(f"mutant: {name}")
                mutated = mutation.apply(nb, seed)
                ok, evidence = mutation.verify(mutated)
                record = {
                    "file": f"mutated/{name}", "base": f"clean/{path.name}",
                    "flaw_id": mutation.flaw_id, "mutation_seed": seed,
                    "split": clean_entry["split"],
                    "verified": ok, "evidence": evidence,
                }
                concl = mutation.conclusion(nb, seed)
                if concl is not None:
                    record["conclusion"] = concl
                if ok:
                    # execute the mutant before writing so stored outputs
                    # reflect the mutated code; apply() only clears outputs of
                    # the edited cell, leaving downstream cells with stale
                    # clean outputs that can contradict the mutant's label
                    executed = ex.execute(mutated)
                    _normalize(executed, Path(name).stem)
                    nbformat.write(executed, str(mut_dir / name))
                    entries.append(record)
                else:
                    discarded.append(record)
                    log(f"  DISCARDED (verify failed): {evidence}")
    return entries, discarded


def build_negative_flags(root: str | Path, log=print) -> dict:
    """Seeded FALSE flags (G3 substrate): each cites real spans of corpus
    notebooks, and each recipe's falseness is mechanically checkable."""
    root = Path(root)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    neg_dir = root / "negative"
    neg_dir.mkdir(parents=True, exist_ok=True)

    def entry(flaw_id, recipe, split, file, claim, code, why):
        cells = nbformat.read(str(root / file), as_version=4).cells
        for cell_idx, quote in (claim, code):
            assert quote in cells[cell_idx]["source"], (file, cell_idx, quote)
        return {
            "flaw_id": flaw_id,
            "claim_span": {"cell": claim[0], "quote": claim[1]},
            "code_span": {"cell": code[0], "quote": code[1]},
            "source_file": file, "recipe": recipe, "split": split,
            "why_false": why,
        }

    flags = []
    for c in manifest["clean"]:
        if c["family"] == "cohort":
            flags.append(entry(
                "selection-survivorship-cohort", "scoped-claim", c["split"], c["file"],
                (5, "it is higher by construction, because it excludes "
                    "churned customers' lower realized LTV"),
                (4, 'retained = df[df["status"] == "active"]'),
                "the claim cell explicitly scopes the second table to "
                "currently retained customers and reports the population "
                "number separately",
            ))
        elif c["family"] == "abtest":
            flags.append(entry(
                "significance-meaningless", "effect-size-present", c["split"], c["file"],
                (4, "we report the effect size alongside the p-value"),
                (3, "stat, p = ttest_ind(a, b)"),
                "the cited analysis computes Cohen's d and the conclusion "
                "cell reports effect size alongside p",
            ))
        elif c["family"] == "program":
            flags.append(entry(
                "regression-to-mean-claim", "control-group-present", c["split"], c["file"],
                (4, "The enrolled accounts' spend recovered strongly in the "
                    "second quarter"),
                (3, 'low = df.nsmallest(300, "spend_q1")'),
                "an equally extreme control group is constructed in the "
                "cited cell and the notebook attributes the movement to "
                "regression to the mean",
            ))
            # reserved entirely for held-out G3: never tuned on (m2 plan §6).
            # only cite held-out program notebooks — a dev-split source here
            # would let a prompt overfit on a tuned-on notebook pass the
            # "held-out" leakage-generalization check and inflate G3.
            if c["split"] == "heldout":
                flags.append(entry(
                    "leakage-fit-before-split", "legit-cv-generalization", "heldout", c["file"],
                    (6, "Cross-validated accuracy estimates how the model will "
                        "perform on unseen customers"),
                    (5, "scores = cross_val_score(pipe, X, y, cv=5)"),
                    "the estimator is a Pipeline: the scaler is refit inside "
                    "each training fold and never sees that fold's test rows",
                ))
    for m in manifest["mutants"]:
        if m["flaw_id"] != "selection-survivorship-cohort":
            continue
        cells = nbformat.read(str(root / m["file"]), as_version=4).cells
        claim_cell = next(
            i for i, c in enumerate(cells)
            if c["cell_type"] == "markdown" and m["conclusion"] in c["source"]
        )
        flags.append(entry(
            "selection-survivorship-cohort", "wrong-code-span", m["split"], m["file"],
            (claim_cell, m["conclusion"]),
            (1, "import numpy as np"),
            "the cited code span is the imports cell and contains no cohort "
            "filter; the evidence does not support the flag",
        ))

    neg_manifest = {"built": date.today().isoformat(), "flags": flags}
    (neg_dir / "MANIFEST.json").write_text(json.dumps(neg_manifest, indent=2))
    log(f"negative: {len(flags)} false flags across "
        f"{len({f['recipe'] for f in flags})} recipes")
    return neg_manifest


def build_corpus(root: str | Path, seeds=DEV_SEEDS, heldout_seeds=HELDOUT_SEEDS, log=print) -> dict:
    root = Path(root)
    clean = (build_clean(root, seeds, "dev", log=log)
             + build_clean(root, heldout_seeds, "heldout", log=log))
    mutants, discarded = build_mutants(root, clean, log=log)
    manifest = {
        "built": date.today().isoformat(),
        "clean": clean,
        "mutants": mutants,
        "discarded": discarded,
        "provenance": "synthetic, self-authored (wald.corpus templates); MIT",
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    log(f"corpus: {len(clean)} clean, {len(mutants)} verified mutants, "
        f"{len(discarded)} discarded")
    build_negative_flags(root, log=log)
    return manifest
