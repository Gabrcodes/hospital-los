"""
Regenerate IEEE-spec figures for the report from the artefacts produced by
models.py. Reads:
  saved_models/metrics.json
  saved_models/regression_model.pkl  (best regressor pipeline)
  saved_models/classification_model.pkl  (best classifier pipeline)
  saved_models/feature_importance.csv
  saved_models/y_test_regression.csv
  saved_models/y_test_classification.csv
  LengthOfStay.csv  (only for the EDA distribution + correlation panels)

Writes vector PDF + 300 DPI PNG to report_ieee/figures/.

IEEE column widths: single = 3.5 in, double = 7.16 in.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SAVED = PROJECT / "saved_models"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_COL = 3.5
DOUBLE_COL = 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

C = dict(
    primary="#1f4e79", accent="#2e75b6",
    good="#3a7d44", warn="#c87f0a", bad="#a33b3b", neutral="#7f7f7f",
)


def save(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  saved figures/{stem}.{{pdf,png}}")


# ── load
metrics = json.loads((SAVED / "metrics.json").read_text(encoding="utf-8"))
metadata = json.loads((SAVED / "metadata.json").read_text(encoding="utf-8"))
fi = pd.read_csv(SAVED / "feature_importance.csv")
y_test_reg = pd.read_csv(SAVED / "y_test_regression.csv").iloc[:, 0]
y_test_clf = pd.read_csv(SAVED / "y_test_classification.csv").iloc[:, 0]
reg_pipe = joblib.load(SAVED / "regression_model.pkl")
clf_pipe = joblib.load(SAVED / "classification_model.pkl")
df_raw = pd.read_csv(PROJECT / "LengthOfStay.csv")

best_reg = metrics["best_regression_model"]
best_clf = metrics["best_classification_model"]


# ── Fig 1: LOS distribution
def fig01():
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    ax.hist(df_raw["lengthofstay"], bins=17, color=C["primary"], edgecolor="white", linewidth=0.4)
    ax.axvline(3.5, color=C["good"], linestyle="--", linewidth=0.8, label="Short / Medium")
    ax.axvline(7.5, color=C["bad"],  linestyle="--", linewidth=0.8, label="Medium / Long")
    ax.set_xlabel("Length of stay (days)")
    ax.set_ylabel("Patients")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig01_los_distribution")


# ── Fig 2: class breakdown
def fig02():
    los = df_raw["lengthofstay"]
    counts = pd.Series(
        [(los <= 3).sum(), ((los > 3) & (los <= 7)).sum(), (los > 7).sum()],
        index=["Short\n(0–3 d)", "Medium\n(4–7 d)", "Long\n(8+ d)"],
    )
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))
    bars = ax.bar(counts.index, counts.values,
                  color=[C["good"], C["warn"], C["bad"]], edgecolor="white")
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + total*0.01,
                f"{v:,}\n({v/total*100:.1f}%)", ha="center", fontsize=6)
    ax.set_ylabel("Patients")
    ax.set_ylim(0, max(counts.values) * 1.18)
    save(fig, "fig02_category_breakdown")


# ── Fig 3: correlations (numeric features vs LOS)
def fig03():
    num = df_raw.select_dtypes(include=[np.number]).copy()
    if "eid" in num.columns:
        num.drop(columns=["eid"], inplace=True)
    corr = num.corr()["lengthofstay"].drop("lengthofstay").sort_values()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.4))
    colors = [C["bad"] if v < 0 else C["primary"] for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Pearson correlation with LOS")
    ax.tick_params(axis="y", labelsize=6)
    save(fig, "fig03_correlations")


# ── Fig 4: heatmap
def fig04():
    num = df_raw.select_dtypes(include=[np.number]).copy()
    if "eid" in num.columns:
        num.drop(columns=["eid"], inplace=True)
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 5.0))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
    ax.set_yticklabels(corr.columns, fontsize=5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Pearson r", fontsize=7)
    save(fig, "fig04_heatmap")


# ── Fig 5: regression compare (4 models)
def fig05():
    reg = metrics["regression"]
    names = list(reg.keys())
    short = [n.replace(" (BONUS)", "*") for n in names]
    r2s   = [reg[n]["R2"]   for n in names]
    rmses = [reg[n]["RMSE"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))
    bars1 = axes[0].bar(short, r2s, color=C["primary"], edgecolor="white")
    axes[0].set_ylabel(r"$R^2$"); axes[0].set_ylim(0, 1)
    axes[0].set_title(r"$R^2$ (higher better)")
    for b, v in zip(bars1, r2s):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{v:.3f}",
                     ha="center", fontsize=6)
    bars2 = axes[1].bar(short, rmses, color=C["accent"], edgecolor="white")
    axes[1].set_ylabel("RMSE (days)")
    axes[1].set_title("RMSE (lower better)")
    axes[1].set_ylim(0, max(rmses) * 1.18)
    for b, v in zip(bars2, rmses):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + max(rmses)*0.02,
                     f"{v:.3f}", ha="center", fontsize=6)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6, rotation=15)
    save(fig, "fig05_regression_compare")


# ── Fig 6: actual vs predicted for best regressor
def fig06():
    from sklearn.model_selection import train_test_split
    drop_cols = ["eid", "vdate", "discharged", "facid"]
    X_full = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns])
    y_reg_full = X_full["lengthofstay"]
    y_clf_full = y_reg_full.apply(lambda d: 0 if d <= 3 else (1 if d <= 7 else 2))
    X_full = X_full.drop(columns=["lengthofstay"])
    _, X_te, _, y_te, _, _ = train_test_split(
        X_full, y_reg_full, y_clf_full,
        test_size=0.2, random_state=42, stratify=y_clf_full,
    )
    pr = reg_pipe.predict(X_te)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))
    ax.scatter(y_te, pr, s=2, alpha=0.25, color=C["primary"], edgecolors="none")
    lo, hi = float(y_te.min()), float(y_te.max())
    ax.plot([lo, hi], [lo, hi], color=C["bad"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Actual LOS (days)")
    ax.set_ylabel("Predicted LOS (days)")
    ax.set_title(best_reg.replace(" (BONUS)", "*"))
    save(fig, "fig06_actual_vs_pred")


# ── Fig 7: classification compare (5 models)
def fig07():
    clf = metrics["classification"]
    names = list(clf.keys())
    short_map = {
        "Logistic Regression":              "LogReg",
        "KNN":                              "KNN",
        "Linear SVM (BONUS)":               "LinSVM*",
        "Random Forest Classifier (BONUS)": "RF*",
        "Neural Network (BONUS)":           "MLP*",
    }
    short = [short_map.get(n, n) for n in names]
    acc = [clf[n]["Accuracy"] for n in names]
    f1  = [clf[n]["F1"]       for n in names]
    x = np.arange(len(short)); w = 0.38
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    ax.bar(x - w/2, acc, w, label="Accuracy", color=C["primary"], edgecolor="white")
    ax.bar(x + w/2, f1,  w, label="F1 (weighted)", color=C["accent"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    save(fig, "fig07_classification_compare")


# ── Fig 8: confusion matrix for best classifier
def fig08():
    cm = np.array(metrics["classification"][best_clf]["confusion_matrix"])
    labels = ["Short", "Medium", "Long"]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(best_clf.replace(" (BONUS)", "*"))
    thresh = cm.max() * 0.6
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    save(fig, "fig08_confusion_matrix")


# ── Fig 9: feature importance from best regressor
def fig09():
    top = fi.head(12).copy().sort_values("Importance")
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))
    ax.barh(top["Feature"], top["Importance"], color=C["primary"], edgecolor="white")
    ax.set_xlabel(f"Importance ({best_reg.replace(' (BONUS)','*')})")
    ax.tick_params(axis="y", labelsize=6)
    save(fig, "fig09_feature_importance")


print("Regenerating IEEE-styled figures from team pipeline artefacts…")
fig01(); fig02(); fig03(); fig04()
fig05(); fig06(); fig09()
fig07(); fig08()
print("Done.")
