"""Corpus builder: 5 clean analysis families x N seeds, plus verified mutants.

Clean notebooks satisfy the clean-corpus criteria (see README): scoped
claims, split-before-fit, corrected or <=3 tests, >=2 classification
metrics, imbalance stated, no extrapolation. They are executed at build
time so stored outputs (value_counts etc.) are available as detector
evidence. Synthetic by design in v1 — licensed real notebooks are a
planned corpus extension, and the eval report says so.
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
        "family": "churn", "num_cols": CHURN_COLS,
        "binary_col": "churned", "binary_values": [1, 0],
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
        "family": "abtest", "num_cols": AB_COLS,
        "binary_col": "variant", "binary_values": ["B", "A"],
        "imports_cell": 1, "conclusion_cell": 4, "imbalanced": False,
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
        "family": "fraud", "num_cols": FRAUD_COLS,
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
        "family": "housing", "num_cols": HOUSING_COLS,
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
        "family": "cohort", "num_cols": COHORT_COLS,
        "binary_col": "status", "binary_values": ["active", "churned"],
        "status_col": "status",
        "imports_cell": 1, "agg_cell": 3, "conclusion_cell": 5,
        "imbalanced": False,
    })


FAMILIES = {
    "churn": churn_notebook,
    "abtest": abtest_notebook,
    "fraud": fraud_notebook,
    "housing": housing_notebook,
    "cohort": cohort_notebook,
}

# which mutation seeds to run per flaw (variants with real randomness get 2)
MUTATION_SEEDS = {
    "leakage-fit-before-split": (0,),
    "testing-multiple-uncorrected": (0, 1),
    "baserate-accuracy-imbalanced": (0,),
    "selection-survivorship-cohort": (0, 1),
}


def build_clean(root: Path, seeds=(11, 12, 13, 14), log=print) -> list[dict]:
    clean_dir = root / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for family, builder in FAMILIES.items():
        for seed in seeds:
            name = f"{family}-s{seed}.ipynb"
            log(f"clean: {name}")
            executed = ex.execute(builder(seed))
            nbformat.write(executed, str(clean_dir / name))
            entries.append({"file": f"clean/{name}", "family": family, "seed": seed})
    return entries


def build_mutants(root: Path, log=print) -> tuple[list[dict], list[dict]]:
    clean_dir, mut_dir = root / "clean", root / "mutated"
    mut_dir.mkdir(parents=True, exist_ok=True)
    entries, discarded = [], []
    for path in sorted(clean_dir.glob("*.ipynb")):
        nb = nbformat.read(str(path), as_version=4)
        for mutation in MUTATIONS:
            if not mutation.applicable(nb):
                continue
            for seed in MUTATION_SEEDS[mutation.flaw_id]:
                name = f"{path.stem}__{mutation.flaw_id}__m{seed}.ipynb"
                log(f"mutant: {name}")
                mutated = mutation.apply(nb, seed)
                ok, evidence = mutation.verify(mutated)
                record = {
                    "file": f"mutated/{name}", "base": f"clean/{path.name}",
                    "flaw_id": mutation.flaw_id, "mutation_seed": seed,
                    "verified": ok, "evidence": evidence,
                }
                if ok:
                    nbformat.write(mutated, str(mut_dir / name))
                    entries.append(record)
                else:
                    discarded.append(record)
                    log(f"  DISCARDED (verify failed): {evidence}")
    return entries, discarded


def build_corpus(root: str | Path, seeds=(11, 12, 13, 14), log=print) -> dict:
    root = Path(root)
    clean = build_clean(root, seeds, log=log)
    mutants, discarded = build_mutants(root, log=log)
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
    return manifest
