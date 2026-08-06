"""
Swan Telco – Churn Analysis Pipeline
Outputs:
  charts/  – PNG charts for presentation deck
  mailer_list.csv   – top 500 at-risk customers for mailers
  churn_risk_all.csv – churn probability for every current customer
  findings.txt      – key text findings
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
CHARTS = os.path.join(BASE, "charts")
os.makedirs(CHARTS, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_STAY  = "#2196F3"   # blue  – retained
C_CHURN = "#F44336"   # red   – churned
C_ACC   = "#4CAF50"   # green – accent

TITLE_FONT  = dict(fontsize=15, fontweight="bold", color="#212121")
LABEL_FONT  = dict(fontsize=11, color="#424242")
TICK_FONT   = dict(labelsize=10)

def savefig(name):
    path = os.path.join(CHARTS, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {name}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. LOAD & CLEAN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Loading data …")
df = pd.read_excel(
    os.path.join(BASE, "1 - Project Data.xlsx"),
    sheet_name="Telco_Churn", engine="openpyxl"
)
df.columns = df.columns.str.strip()
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df["Total Charges"].fillna(df["Monthly Charges"], inplace=True)

# Binary flags
df["Churned"] = (df["Churn Label"] == "Yes").astype(int)
churners  = df[df["Churned"] == 1].copy()
stayers   = df[df["Churned"] == 0].copy()
n_total   = len(df)
n_churn   = df["Churned"].sum()
n_stay    = n_total - n_churn
churn_rate = n_churn / n_total * 100
print(f"  Total customers: {n_total}  |  Churned: {n_churn} ({churn_rate:.1f}%)  |  Retained: {n_stay}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. SLIDE 1 – OVERVIEW DONUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nSlide 1 – Overview …")
fig, ax = plt.subplots(figsize=(6, 6))
vals  = [n_stay, n_churn]
labs  = [f"Retained\n{n_stay:,}", f"Churned\n{n_churn:,}"]
cols  = [C_STAY, C_CHURN]
wedges, texts, autotexts = ax.pie(
    vals, labels=labs, colors=cols, autopct="%1.1f%%",
    startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=3)
)
for t in texts:   t.set_fontsize(13)
for a in autotexts: a.set_fontsize(12); a.set_color("white"); a.set_fontweight("bold")
ax.set_title("Overall Churn Rate", **TITLE_FONT, pad=18)
savefig("01_overview_donut.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. SLIDE 2 – DEMOGRAPHICS OF CHURNERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Slide 2 – Demographics …")

def churn_rate_by(col):
    g = df.groupby(col)["Churned"].agg(["sum","count"])
    g.columns = ["churned","total"]
    g["rate"] = g["churned"] / g["total"] * 100
    return g.reset_index()

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Demographics of Churners", **TITLE_FONT, y=1.02)

demo_cols = ["Gender", "Senior Citizen", "Partner", "Dependents"]
demo_titles = ["Gender", "Senior Citizen", "Has Partner", "Has Dependents"]

for ax, col, title in zip(axes, demo_cols, demo_titles):
    g = churn_rate_by(col)
    bars = ax.bar(g[col], g["rate"], color=[C_CHURN if r > churn_rate else C_STAY for r in g["rate"]], edgecolor="white", linewidth=1.5)
    ax.axhline(churn_rate, color="#9E9E9E", linestyle="--", linewidth=1.2, label=f"Overall {churn_rate:.1f}%")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#212121")
    ax.set_ylabel("Churn Rate (%)", **LABEL_FONT)
    ax.tick_params(**TICK_FONT)
    ax.set_ylim(0, 60)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
savefig("02_demographics.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. SLIDE 3 – TENURE DISTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Slide 3 – Tenure …")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(stayers["Tenure Months"],  bins=24, color=C_STAY,  alpha=0.7, label="Retained", edgecolor="white")
ax.hist(churners["Tenure Months"], bins=24, color=C_CHURN, alpha=0.7, label="Churned",  edgecolor="white")
ax.axvline(churners["Tenure Months"].median(), color=C_CHURN, linestyle="--", linewidth=1.5,
           label=f"Median churner: {churners['Tenure Months'].median():.0f} mo")
ax.axvline(stayers["Tenure Months"].median(),  color=C_STAY,  linestyle="--", linewidth=1.5,
           label=f"Median retained: {stayers['Tenure Months'].median():.0f} mo")
ax.set_title("Tenure Distribution: Churners vs Retained", **TITLE_FONT)
ax.set_xlabel("Tenure (months)", **LABEL_FONT)
ax.set_ylabel("Number of Customers", **LABEL_FONT)
ax.tick_params(**TICK_FONT)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
savefig("03_tenure_distribution.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. SLIDE 4 – WHY ARE THEY CHURNING?
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Slide 4 – Churn reasons …")
reasons = churners["Churn Reason"].value_counts().head(12)

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(reasons.index[::-1], reasons.values[::-1],
               color=C_CHURN, edgecolor="white", linewidth=1)
for bar in bars:
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            str(int(bar.get_width())), va="center", fontsize=10)
ax.set_title("Top Reasons Customers Churned", **TITLE_FONT)
ax.set_xlabel("Number of Customers", **LABEL_FONT)
ax.tick_params(**TICK_FONT)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
savefig("04_churn_reasons.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. SLIDE 5 – PRODUCTS & CONTRACT TYPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Slide 5 – Products …")
product_cols = ["Phone Service","Multiple Lines","Internet Service",
                "Online Security","Online Backup","Device Protection",
                "Tech Support","Streaming TV","Streaming Movies",
                "Contract","Paperless Billing","Payment Method"]

fig, axes = plt.subplots(3, 4, figsize=(22, 14))
fig.suptitle("Product & Contract Profile: Churn Rate by Category", **TITLE_FONT, y=1.01)

for ax, col in zip(axes.flatten(), product_cols):
    g = churn_rate_by(col).sort_values("rate", ascending=False)
    colors = [C_CHURN if r > churn_rate else C_STAY for r in g["rate"]]
    ax.bar(range(len(g)), g["rate"], color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(churn_rate, color="#9E9E9E", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g[col], rotation=25, ha="right", fontsize=9)
    ax.set_title(col, fontsize=11, fontweight="bold", color="#212121")
    ax.set_ylabel("Churn %", fontsize=9)
    ax.set_ylim(0, 85)
    ax.spines[["top","right"]].set_visible(False)
    for i, r in enumerate(g["rate"]):
        ax.text(i, r + 0.8, f"{r:.0f}%", ha="center", fontsize=8, fontweight="bold")

plt.tight_layout()
savefig("05_products_contract.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. SLIDE 6 – MONTHLY CHARGES COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Slide 6 – Monthly charges …")
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(stayers["Monthly Charges"],  bins=30, color=C_STAY,  alpha=0.7, label="Retained", edgecolor="white")
ax.hist(churners["Monthly Charges"], bins=30, color=C_CHURN, alpha=0.7, label="Churned",  edgecolor="white")
ax.axvline(churners["Monthly Charges"].mean(), color=C_CHURN, linestyle="--", linewidth=1.5,
           label=f"Avg churner: ${churners['Monthly Charges'].mean():.0f}")
ax.axvline(stayers["Monthly Charges"].mean(),  color=C_STAY,  linestyle="--", linewidth=1.5,
           label=f"Avg retained: ${stayers['Monthly Charges'].mean():.0f}")
ax.set_title("Monthly Charges: Churners vs Retained", **TITLE_FONT)
ax.set_xlabel("Monthly Charges ($)", **LABEL_FONT)
ax.set_ylabel("Number of Customers", **LABEL_FONT)
ax.tick_params(**TICK_FONT)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
savefig("06_monthly_charges.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. BUILD PREDICTIVE MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nBuilding model …")

FEATURE_COLS = [
    "Gender","Senior Citizen","Partner","Dependents","Tenure Months",
    "Phone Service","Multiple Lines","Internet Service",
    "Online Security","Online Backup","Device Protection","Tech Support",
    "Streaming TV","Streaming Movies","Contract","Paperless Billing",
    "Payment Method","Monthly Charges","Total Charges"
]

df_model = df[FEATURE_COLS + ["Churned"]].copy()

# Encode categoricals with OrdinalEncoder (preserves NaN as NaN – safe for HGBC)
cat_cols = df_model[FEATURE_COLS].select_dtypes("object").columns.tolist()
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                     encoded_missing_value=np.nan)
df_model[cat_cols] = enc.fit_transform(df_model[cat_cols])

X = df_model[FEATURE_COLS].astype(float)
y = df_model["Churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# HistGradientBoostingClassifier handles NaN natively and is fast
model = HistGradientBoostingClassifier(
    max_iter=300, max_depth=4, learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc  = roc_auc_score(y_test, y_prob)
cv   = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(f"  AUC (test): {auc:.4f}  |  CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")
print(classification_report(y_test, y_pred))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. SLIDE 7 – FEATURE IMPORTANCE (what factors drive churn?)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Slide 7 – Feature importance …")
# HistGBT doesn't have feature_importances_; use permutation importance on test set
perm = permutation_importance(model, X_test, y_test, n_repeats=10,
                              random_state=42, scoring="roc_auc")
fi = pd.Series(perm.importances_mean, index=FEATURE_COLS).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
colors = [C_CHURN if v >= fi.nlargest(5).min() else C_STAY for v in fi]
ax.barh(fi.index, fi.values, color=colors, edgecolor="white", linewidth=1)
ax.set_title("Factors That Most Influence Churn\n(Permutation Importance – AUC drop)",
             **TITLE_FONT)
ax.set_xlabel("Importance Score", **LABEL_FONT)
ax.tick_params(**TICK_FONT)
ax.spines[["top","right"]].set_visible(False)

red_patch  = mpatches.Patch(color=C_CHURN, label="Top 5 drivers")
blue_patch = mpatches.Patch(color=C_STAY,  label="Other factors")
ax.legend(handles=[red_patch, blue_patch], fontsize=10)
plt.tight_layout()
savefig("07_feature_importance.png")

# Store top features for findings
top5 = fi.nlargest(5)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. SLIDE 8 – SIGN-UP INCENTIVE ANALYSIS ($2.50 on one metric)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Slide 8 – Incentive analysis …")

# Binary product columns where Yes/No is meaningful
incentive_cols = [
    "Online Security","Online Backup","Device Protection",
    "Tech Support","Streaming TV","Streaming Movies",
    "Multiple Lines","Paperless Billing"
]

# For each, calculate churn rate for "Yes" vs "No" (excluding "No internet service" etc.)
rows = []
for col in incentive_cols:
    sub = df[df[col].isin(["Yes","No"])]
    rate_yes = sub[sub[col]=="Yes"]["Churned"].mean() * 100
    rate_no  = sub[sub[col]=="No" ]["Churned"].mean() * 100
    rows.append({"Feature": col, "Churn if Yes": rate_yes, "Churn if No": rate_no,
                 "Reduction": rate_no - rate_yes})

inc_df = pd.DataFrame(rows).sort_values("Reduction", ascending=False)
print(inc_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(inc_df))
w = 0.35
ax.bar(x - w/2, inc_df["Churn if No"],  w, label="Churn if NOT on product", color=C_CHURN, alpha=0.85)
ax.bar(x + w/2, inc_df["Churn if Yes"], w, label="Churn if ON product",      color=C_STAY,  alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(inc_df["Feature"], rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Churn Rate (%)", **LABEL_FONT)
ax.set_title("Churn Rate With vs Without Each Add-On\n(Which sign-up should we incentivise?)",
             **TITLE_FONT)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(**TICK_FONT)
# Annotate reduction on top of No-product bar
for i, row in inc_df.reset_index(drop=True).iterrows():
    ax.text(i - w/2, row["Churn if No"] + 0.5,
            f"↓{row['Reduction']:.0f}pp", ha="center", fontsize=9, color=C_CHURN, fontweight="bold")
plt.tight_layout()
savefig("08_incentive_analysis.png")

best_incentive = inc_df.iloc[0]["Feature"]
best_reduction = inc_df.iloc[0]["Reduction"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. SCORE ALL CURRENT CUSTOMERS (non-churners)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nScoring remaining customers …")

current = df[df["Churned"] == 0].copy()
X_current = current[FEATURE_COLS].copy()
# Re-use the same OrdinalEncoder fitted on training data
X_current[cat_cols] = enc.transform(X_current[cat_cols])
X_current = X_current.astype(float)

current["Churn_Probability"] = model.predict_proba(X_current)[:, 1]
current["Churn_Risk_Band"] = pd.cut(
    current["Churn_Probability"],
    bins=[0, 0.25, 0.50, 0.75, 1.0],
    labels=["Low", "Medium", "High", "Very High"]
)

# Full churn risk list (all current customers)
risk_cols = ["CustomerID","Gender","Senior Citizen","Partner","Dependents",
             "Tenure Months","Contract","Monthly Charges",
             "Churn_Probability","Churn_Risk_Band"]
churn_risk_all = current[risk_cols].sort_values("Churn_Probability", ascending=False)
churn_risk_all["Churn_Probability"] = churn_risk_all["Churn_Probability"].round(4)
churn_risk_all.to_csv(os.path.join(BASE, "churn_risk_all.csv"), index=False)
print(f"  churn_risk_all.csv → {len(churn_risk_all):,} customers")
print(churn_risk_all["Churn_Risk_Band"].value_counts())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. MAILER LIST – top 500 at-risk customers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nBuilding mailer list …")
mailer = churn_risk_all.head(500).copy()
mailer.to_csv(os.path.join(BASE, "mailer_list.csv"), index=False)
print(f"  mailer_list.csv → {len(mailer)} customers")
print(f"  Probability range: {mailer['Churn_Probability'].min():.3f} – {mailer['Churn_Probability'].max():.3f}")
print(f"  Expected uptake (20%): {int(500 * 0.2)} customers saved")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. SLIDE 9 – CHURN RISK DISTRIBUTION (current customers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Slide 9 – Risk distribution …")
band_counts = current["Churn_Risk_Band"].value_counts().reindex(["Low","Medium","High","Very High"])
band_colors = ["#4CAF50","#FFC107","#FF5722","#B71C1C"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(band_counts.index, band_counts.values, color=band_colors, edgecolor="white", linewidth=1.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_title("Current Customers by Churn Risk Band", **TITLE_FONT)
ax.set_xlabel("Risk Band", **LABEL_FONT)
ax.set_ylabel("Number of Customers", **LABEL_FONT)
ax.tick_params(**TICK_FONT)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
savefig("09_risk_bands.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. SLIDE 10 – MAILER LIST PROFILE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Slide 10 – Mailer profile …")
mailer_full = current[current["CustomerID"].isin(mailer["CustomerID"])].copy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Profile of 500 Mailer Recipients", **TITLE_FONT, y=1.02)

for ax, col, title in zip(axes,
    ["Contract","Internet Service","Payment Method"],
    ["Contract Type","Internet Service","Payment Method"]):
    vc = mailer_full[col].value_counts()
    ax.pie(vc.values, labels=vc.index, autopct="%1.0f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title(title, fontsize=12, fontweight="bold")

plt.tight_layout()
savefig("10_mailer_profile.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 15. SLIDE 11 – MODEL PERFORMANCE (ROC curve)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Slide 11 – ROC curve …")
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=C_CHURN, linewidth=2.5, label=f"Model AUC = {auc:.3f}")
ax.plot([0,1],[0,1], color="#9E9E9E", linestyle="--", linewidth=1.5, label="Random baseline")
ax.fill_between(fpr, tpr, alpha=0.08, color=C_CHURN)
ax.set_xlabel("False Positive Rate", **LABEL_FONT)
ax.set_ylabel("True Positive Rate", **LABEL_FONT)
ax.set_title("Model ROC Curve\n(Hist Gradient Boosting Classifier)", **TITLE_FONT)
ax.legend(fontsize=11)
ax.tick_params(**TICK_FONT)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
savefig("11_roc_curve.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 16. WRITE FINDINGS SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nWriting findings …")

senior_churn = df[df["Senior Citizen"]=="Yes"]["Churned"].mean()*100
no_partner   = df[df["Partner"]=="No"]["Churned"].mean()*100
no_dep       = df[df["Dependents"]=="No"]["Churned"].mean()*100
mtm_churn    = df[df["Contract"]=="Month-to-month"]["Churned"].mean()*100
two_yr_churn = df[df["Contract"]=="Two year"]["Churned"].mean()*100
fiber_churn  = df[df["Internet Service"]=="Fiber optic"]["Churned"].mean()*100
no_sec_churn = df[df["Online Security"]=="No"]["Churned"].mean()*100
sec_churn    = df[df["Online Security"]=="Yes"]["Churned"].mean()*100

findings = f"""
==============================================================
  SWAN TELCO – CHURN ANALYSIS KEY FINDINGS
==============================================================

DATASET
  Total customers : {n_total:,}
  Churned         : {n_churn:,} ({churn_rate:.1f}%)
  Retained        : {n_stay:,} ({100-churn_rate:.1f}%)

──────────────────────────────────────────────────────────────
DEMOGRAPHICS OF CHURNERS
──────────────────────────────────────────────────────────────
  • Gender: churn rate is similar for Male and Female (~{df[df["Gender"]=="Male"]["Churned"].mean()*100:.0f}% vs {df[df["Gender"]=="Female"]["Churned"].mean()*100:.0f}%) — not a differentiator.
  • Senior Citizens churn at {senior_churn:.0f}% vs {df[df["Senior Citizen"]=="No"]["Churned"].mean()*100:.0f}% for non-seniors.
  • Customers with no partner churn at {no_partner:.0f}% vs {df[df["Partner"]=="Yes"]["Churned"].mean()*100:.0f}% with partner.
  • Customers with no dependents churn at {no_dep:.0f}% vs {df[df["Dependents"]=="Yes"]["Churned"].mean()*100:.0f}% with dependents.
  • Churners have a median tenure of {churners["Tenure Months"].median():.0f} months vs {stayers["Tenure Months"].median():.0f} months for stayers — most churn happens early.
  • Average monthly charge of churners: ${churners["Monthly Charges"].mean():.0f} vs ${stayers["Monthly Charges"].mean():.0f} for retained.

TOP CHURN REASONS
  1. Attitude of support person     ({df[df["Churn Reason"]=="Attitude of support person"]["Churned"].sum()} customers)
  2. Competitor higher download speeds ({df[df["Churn Reason"]=="Competitor offered higher download speeds"]["Churned"].sum()} customers)
  3. Competitor offered more data    ({df[df["Churn Reason"]=="Competitor offered more data"]["Churned"].sum()} customers)
  4. Don't know                       ({df[df["Churn Reason"]=="Don't know"]["Churned"].sum()} customers)
  5. Competitor made better offer    ({df[df["Churn Reason"]=="Competitor made better offer"]["Churned"].sum()} customers)

──────────────────────────────────────────────────────────────
WHAT FACTORS MOST INFLUENCE CHURN? (Model-driven)
──────────────────────────────────────────────────────────────
Top 5 features from Gradient Boosting model (AUC = {auc:.3f}):
{chr(10).join(f"  {i+1}. {feat}  (importance: {imp:.4f})" for i,(feat,imp) in enumerate(top5.items()))}

  Key takeaways:
  • Month-to-month contract: {mtm_churn:.0f}% churn vs {two_yr_churn:.0f}% on two-year contracts.
  • Fiber optic customers churn at {fiber_churn:.0f}% — high charges + competitor pressure.
  • No Online Security: {no_sec_churn:.0f}% churn vs {sec_churn:.0f}% with Online Security.

──────────────────────────────────────────────────────────────
RECOMMENDED $2.50 SIGN-UP INCENTIVE
──────────────────────────────────────────────────────────────
  RECOMMEND: incentivise sign-up to "{best_incentive}"
  Churn rate drops by {best_reduction:.0f} percentage points for customers
  who have this product vs those who don't.

  Full ranking:
{inc_df[["Feature","Churn if No","Churn if Yes","Reduction"]].to_string(index=False)}

──────────────────────────────────────────────────────────────
MAILER LIST (500 customers)
──────────────────────────────────────────────────────────────
  File: mailer_list.csv
  These are the 500 current customers with the highest
  predicted churn probability.
  Min prob in list : {mailer['Churn_Probability'].min():.3f}
  Max prob in list : {mailer['Churn_Probability'].max():.3f}
  Expected saves (20% uptake): {int(500*0.2)} customers

──────────────────────────────────────────────────────────────
CHURN RISK – ALL CURRENT CUSTOMERS
──────────────────────────────────────────────────────────────
  File: churn_risk_all.csv
  {len(churn_risk_all):,} current customers scored.
  Risk bands:
{current["Churn_Risk_Band"].value_counts().reindex(["Very High","High","Medium","Low"]).to_string()}

  Share this file with the customer service team.
  Column "Churn_Probability" (0–1) is the churn likelihood.

==============================================================
"""

with open(os.path.join(BASE, "findings.txt"), "w", encoding="utf-8") as f:
    f.write(findings)
print(findings)
print("\nAll outputs complete.")
print(f"  Charts  : {CHARTS}")
print(f"  Mailer  : {os.path.join(BASE, 'mailer_list.csv')}")
print(f"  Risk    : {os.path.join(BASE, 'churn_risk_all.csv')}")
print(f"  Findings: {os.path.join(BASE, 'findings.txt')}")
