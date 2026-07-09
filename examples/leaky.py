# %% [markdown]
# Churn model — activity features
#
# Predict churn from customer activity. Balanced classes; both accuracy and AUC reported.

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# %%
rng = np.random.default_rng(11)
n = 1500
df = pd.DataFrame({
    "tenure_months": rng.gamma(4, 6, n).round(1),
    "monthly_spend": rng.normal(52, 15, n).round(2),
    "support_tickets": rng.poisson(1.4, n),
    "logins_per_week": rng.gamma(2.5, 1.8, n).round(1),
    "discount_share": rng.beta(2, 6, n).round(3),
    "emails_opened": rng.poisson(5, n),
    "pages_per_session": rng.gamma(3, 1.2, n).round(1),
    "days_since_last_order": rng.gamma(3, 9, n).round(0),
    "referrals": rng.poisson(0.6, n),
    "cart_abandons": rng.poisson(2.1, n),
})
logit = (-0.05 * df["tenure_months"] - 0.4 * df["logins_per_week"]
         + 0.03 * df["days_since_last_order"] + 0.35 * df["support_tickets"]
         + rng.normal(0, 1.5, n))
df["churned"] = (logit > np.median(logit)).astype(int)

# %%
df["churned"].value_counts(normalize=True)

# %%
num_cols = ['tenure_months', 'monthly_spend', 'support_tickets', 'logins_per_week', 'discount_share', 'emails_opened', 'pages_per_session', 'days_since_last_order', 'referrals', 'cart_abandons']
X = df[num_cols]
y = df["churned"]

# %%
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=11, stratify=y)

# %%
model = LogisticRegression(max_iter=1000)
model.fit(X_tr, y_tr)
pred = model.predict(X_te)
proba = model.predict_proba(X_te)[:, 1]
acc = accuracy_score(y_te, pred)
auc = roc_auc_score(y_te, proba)
print(f"accuracy={acc:.3f}  roc_auc={auc:.3f}")

# %% [markdown]
# On the held-out 25% test set the model reaches the accuracy and AUC printed above.
# Classes are close to balanced (see the value_counts cell), so accuracy is meaningful
# alongside AUC. These results describe customers in this dataset only.
