"""
Grid search CV for the Logistic Regression churn model.

Assumes you already have your stratified 80/20 split as:
    X_train, X_test, y_train, y_test

If not, the commented block at the top shows how to recreate that split
from the raw Telco_Churn data. Otherwise skip straight to "GRID SEARCH SETUP".
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay,
    make_scorer
)

# ---------------------------------------------------------------------------
# OPTIONAL: recreate data + split (skip if you already have X_train/X_test/y_train/y_test)
# ---------------------------------------------------------------------------
# df = pd.read_excel("1__Project_Data.xlsx", sheet_name="Telco_Churn")
#
# # Total Charges has 11 blank rows where Tenure Months == 0 (brand new sign-ups)
# df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
# df["Total Charges"] = df["Total Charges"].fillna(0)
#
# leakage_cols = ["CustomerID", "Count", "Churn Label", "Churn Reason", "Churn Value",
#                 "Lat Long", "Latitude", "Longitude", "Zip Code", "Country"]
# X = df.drop(columns=leakage_cols)
# y = df["Churn Value"]
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# ---------------------------------------------------------------------------
# GRID SEARCH SETUP
# ---------------------------------------------------------------------------

# Adjust these two lists to match whatever's actually left in X_train's columns
# after your own cleaning/feature selection.
numeric_features = ["Tenure Months", "Monthly Charges", "Total Charges"]
categorical_features = [c for c in X_train.columns if c not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_features),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ]
)

# liblinear supports both l1 and l2 penalties, and handles this dataset size easily
param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["liblinear"],
    "clf__class_weight": [None, "balanced"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Custom precision@k% scorer -------------------------------------------
# 500 out of the full 7,043 customers is ~7.1%. Since each CV fold is only a
# fraction of the training data, we scale k proportionally to the fold size
# rather than hardcoding 500 (which wouldn't fit inside a single fold).
TOP_K_FRACTION = 500 / 7043  # ~0.071

def precision_at_k_fraction(y_true, y_proba, k_fraction=TOP_K_FRACTION):
    y_true = np.asarray(y_true)
    n_top = max(1, int(round(len(y_true) * k_fraction)))
    top_idx = np.argsort(y_proba)[::-1][:n_top]
    return y_true[top_idx].mean()

precision_at_k_scorer = make_scorer(
    precision_at_k_fraction, response_method="predict_proba", greater_is_better=True
)

# roc_auc / average_precision as the primary metrics since churn is imbalanced
# (~26.5% positive class); accuracy alone would reward a model that just
# predicts "no churn" for everyone. precision_at_k is tracked here for
# reference only — it's noisier per-fold, so it's not what we refit on.
scoring = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "average_precision": "average_precision",
    "precision_at_k": precision_at_k_scorer,
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scoring,
    refit="roc_auc",   # picks the best model by roc_auc, refits it on all of X_train
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
    verbose=1,
)

# Fit ONLY on training data — cv splits are drawn from X_train/y_train only,
# X_test is never touched until final evaluation below.
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print(f"Best CV ROC AUC: {grid_search.best_score_:.4f}")

# ---------------------------------------------------------------------------
# Inspect CV results across all parameter combos (sorted by mean test ROC AUC)
# ---------------------------------------------------------------------------
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results_sorted = cv_results.sort_values("mean_test_roc_auc", ascending=False)
print(
    cv_results_sorted[
        ["params", "mean_test_roc_auc", "std_test_roc_auc",
         "mean_test_f1", "mean_test_average_precision", "mean_test_precision_at_k"]
    ].head(10).to_string(index=False)
)

# ---------------------------------------------------------------------------
# Final, unbiased evaluation on the held-out 20% test set
# ---------------------------------------------------------------------------
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nTest set performance:")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))