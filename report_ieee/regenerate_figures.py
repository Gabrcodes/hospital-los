"""
Regenerate all report figures at IEEE publication specs.

IEEE column widths:
  - Single column: 3.5 in (88.9 mm)
  - Double column: 7.16 in (181.9 mm)

Output: vector PDF (preferred for LaTeX) at 300 DPI fallback raster.
Run with the user's Python 3.13:
  C:/Users/bedok/scoop/apps/python313/current/python.exe regenerate_figures.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              mean_absolute_error, mean_squared_error,
                              precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# ─────────────────────────────────────────────
# IEEE figure styling
# ─────────────────────────────────────────────
SINGLE_COL = 3.5      # inches
DOUBLE_COL = 7.16     # inches

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

# Color palette — colourblind-friendly, print-safe
C = {
    "primary":  "#1f4e79",
    "accent":   "#2e75b6",
    "good":     "#3a7d44",
    "warn":     "#c87f0a",
    "bad":      "#a33b3b",
    "neutral":  "#7f7f7f",
}

PROJECT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_CSV = PROJECT / "LengthOfStay.csv"


def save(fig, stem: str):
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  saved figures/{stem}.{{pdf,png}}")


# ─────────────────────────────────────────────
# Data pipeline (mirrors models.py exactly)
# ─────────────────────────────────────────────
def load_and_preprocess():
    df_raw = pd.read_csv(DATA_CSV)
    df = df_raw.copy()

    drop_cols = ["eid", "vdate", "discharged", "facid"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    yn_cols = [c for c in df.columns if df[c].dtype == object
               and set(df[c].dropna().unique()).issubset({"Yes", "No"})]
    for col in yn_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    df.fillna(df.median(numeric_only=True), inplace=True)

    y_reg = df["lengthofstay"].copy()
    y_clf = y_reg.apply(lambda d: 0 if d <= 3 else (1 if d <= 7 else 2))
    X = df.drop(columns=["lengthofstay"])

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return df_raw, df, X_scaled, y_reg, y_clf, X.columns.tolist(), scaler


# ─────────────────────────────────────────────
# Train models
# ─────────────────────────────────────────────
def train_regression(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression":      (LinearRegression(),                                   "MUST"),
        "Decision Tree":          (DecisionTreeRegressor(max_depth=6, random_state=42), "MUST"),
        "Random Forest (BONUS)":  (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), "BONUS"),
        "XGBoost (BONUS)":        (xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1), "BONUS"),
    }
    out = {}
    for name, (m, kind) in models.items():
        m.fit(X_tr, y_tr)
        pr = m.predict(X_te)
        out[name] = {
            "model": m, "kind": kind, "preds": pr, "y_test": y_te,
            "R2":    r2_score(y_te, pr),
            "RMSE":  float(np.sqrt(mean_squared_error(y_te, pr))),
            "MAE":   mean_absolute_error(y_te, pr),
        }
        print(f"  {name:<28} R2={out[name]['R2']:.4f}  RMSE={out[name]['RMSE']:.4f}")
    return out


def train_classification(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Logistic Regression":     (LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1), "MUST"),
        "K-Nearest Neighbors":     (KNeighborsClassifier(n_neighbors=5, n_jobs=-1),                "MUST"),
        "SVM (BONUS)":             (SVC(kernel="rbf", probability=False, random_state=42),         "BONUS"),
        "Neural Network (BONUS)":  (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42), "BONUS"),
    }
    out = {}
    for name, (m, kind) in models.items():
        # SVM on 80k samples is slow — sub-sample training set for SVM only
        if "SVM" in name:
            idx = np.random.RandomState(42).choice(len(X_tr), size=20000, replace=False)
            m.fit(X_tr.iloc[idx], y_tr.iloc[idx])
        else:
            m.fit(X_tr, y_tr)
        pr = m.predict(X_te)
        out[name] = {
            "model": m, "kind": kind, "preds": pr, "y_test": y_te,
            "Accuracy":  accuracy_score(y_te, pr),
            "Precision": precision_score(y_te, pr, average="weighted", zero_division=0),
            "Recall":    recall_score(y_te, pr, average="weighted", zero_division=0),
            "F1":        f1_score(y_te, pr, average="weighted", zero_division=0),
            "CM":        confusion_matrix(y_te, pr),
        }
        print(f"  {name:<28} Acc={out[name]['Accuracy']:.4f}  F1={out[name]['F1']:.4f}")
    return out


# ─────────────────────────────────────────────
# Figure generators
# ─────────────────────────────────────────────
def fig_los_distribution(df_raw):
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    ax.hist(df_raw["lengthofstay"], bins=17, color=C["primary"], edgecolor="white", linewidth=0.4)
    ax.axvline(3.5, color=C["good"], linestyle="--", linewidth=0.8, label="Short / Medium")
    ax.axvline(7.5, color=C["bad"],  linestyle="--", linewidth=0.8, label="Medium / Long")
    ax.set_xlabel("Length of stay (days)")
    ax.set_ylabel("Patients")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig01_los_distribution")


def fig_category_breakdown(y_clf):
    counts = pd.Series(y_clf).value_counts().sort_index()
    labels = ["Short\n(0–3 d)", "Medium\n(4–7 d)", "Long\n(8+ d)"]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))
    bars = ax.bar(labels, counts.values, color=[C["good"], C["warn"], C["bad"]], edgecolor="white")
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + total*0.01,
                f"{v:,}\n({v/total*100:.1f}%)", ha="center", fontsize=6)
    ax.set_ylabel("Patients")
    ax.set_ylim(0, max(counts.values) * 1.18)
    save(fig, "fig02_category_breakdown")


def fig_correlations(df, target="lengthofstay"):
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()[target].drop(target).sort_values()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.4))
    colors = [C["bad"] if v < 0 else C["primary"] for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Pearson correlation with LOS")
    ax.tick_params(axis="y", labelsize=6)
    save(fig, "fig03_correlations")


def fig_heatmap(df):
    num = df.select_dtypes(include=[np.number])
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


def fig_regression_compare(reg_results):
    names = list(reg_results.keys())
    short = [n.replace(" (BONUS)", "*") for n in names]
    r2 = [reg_results[n]["R2"] for n in names]
    rmse = [reg_results[n]["RMSE"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))
    bars1 = axes[0].bar(short, r2, color=C["primary"], edgecolor="white")
    axes[0].set_ylabel(r"$R^2$")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(r"$R^2$ (higher better)")
    for b, v in zip(bars1, r2):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{v:.3f}",
                     ha="center", fontsize=6)
    bars2 = axes[1].bar(short, rmse, color=C["accent"], edgecolor="white")
    axes[1].set_ylabel("RMSE (days)")
    axes[1].set_title("RMSE (lower better)")
    axes[1].set_ylim(0, max(rmse) * 1.18)
    for b, v in zip(bars2, rmse):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + max(rmse)*0.02,
                     f"{v:.3f}", ha="center", fontsize=6)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6, rotation=15)
    save(fig, "fig05_regression_compare")


def fig_actual_vs_pred(reg_results):
    best = max((k for k in reg_results), key=lambda k: reg_results[k]["R2"])
    yt = reg_results[best]["y_test"]
    pr = reg_results[best]["preds"]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))
    ax.scatter(yt, pr, s=2, alpha=0.25, color=C["primary"], edgecolors="none")
    lo, hi = float(yt.min()), float(yt.max())
    ax.plot([lo, hi], [lo, hi], color=C["bad"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Actual LOS (days)")
    ax.set_ylabel("Predicted LOS (days)")
    ax.set_title(f"{best.replace(' (BONUS)', '*')}")
    save(fig, "fig06_actual_vs_pred")


def fig_classification_compare(clf_results):
    names = list(clf_results.keys())
    short = [n.replace(" (BONUS)", "*").replace("Neural Network", "MLP")
                .replace("Logistic Regression", "LogReg")
                .replace("K-Nearest Neighbors", "KNN")
                .replace("SVM", "SVM")
             for n in names]
    acc = [clf_results[n]["Accuracy"] for n in names]
    f1  = [clf_results[n]["F1"]       for n in names]
    x = np.arange(len(short))
    w = 0.38
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    ax.bar(x - w/2, acc, w, label="Accuracy", color=C["primary"], edgecolor="white")
    ax.bar(x + w/2, f1,  w, label="F1 (weighted)", color=C["accent"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    save(fig, "fig07_classification_compare")


def fig_confusion_matrix(clf_results):
    best = max(clf_results, key=lambda k: clf_results[k]["F1"])
    cm = clf_results[best]["CM"]
    labels = ["Short", "Medium", "Long"]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{best.replace(' (BONUS)', '*')}")
    thresh = cm.max() * 0.6
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=7)
    save(fig, "fig08_confusion_matrix")


def fig_feature_importance(reg_results, feature_names):
    rf = reg_results["Random Forest (BONUS)"]["model"]
    fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.0))
    ax.barh(fi.index[::-1], fi.values[::-1], color=C["primary"], edgecolor="white")
    ax.set_xlabel("Importance (Random Forest)")
    ax.tick_params(axis="y", labelsize=6)
    save(fig, "fig09_feature_importance")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("Loading and preprocessing…")
    df_raw, df_clean, X, y_reg, y_clf, feature_names, scaler = load_and_preprocess()
    print(f"  rows={len(df_raw):,}  features={X.shape[1]}")

    print("\nFigure 1 — LOS distribution")
    fig_los_distribution(df_raw)
    print("Figure 2 — Category breakdown")
    fig_category_breakdown(y_clf)
    print("Figure 3 — Correlations")
    fig_correlations(df_clean)
    print("Figure 4 — Correlation heatmap")
    fig_heatmap(df_clean)

    print("\nTraining regression models…")
    reg = train_regression(X, y_reg)
    print("Figure 5 — Regression comparison")
    fig_regression_compare(reg)
    print("Figure 6 — Actual vs predicted")
    fig_actual_vs_pred(reg)
    print("Figure 9 — Feature importance")
    fig_feature_importance(reg, feature_names)

    print("\nTraining classification models…")
    clf = train_classification(X, y_clf)
    print("Figure 7 — Classification comparison")
    fig_classification_compare(clf)
    print("Figure 8 — Confusion matrix")
    fig_confusion_matrix(clf)

    # Persist numeric results so the LaTeX numbers come from a real run
    metrics = {
        "regression": {k: {kk: float(vv) for kk, vv in v.items()
                           if kk in ("R2", "RMSE", "MAE")}
                       for k, v in reg.items()},
        "classification": {k: {kk: float(vv) for kk, vv in v.items()
                               if kk in ("Accuracy", "Precision", "Recall", "F1")}
                           for k, v in clf.items()},
        "confusion_matrix_best": {
            "model": max(clf, key=lambda k: clf[k]["F1"]),
            "matrix": clf[max(clf, key=lambda k: clf[k]["F1"])]["CM"].tolist(),
        },
        "class_distribution": {
            "short":  int((y_clf == 0).sum()),
            "medium": int((y_clf == 1).sum()),
            "long":   int((y_clf == 2).sum()),
        },
        "n_rows":     int(len(df_raw)),
        "n_features": int(X.shape[1]),
    }
    with open(FIG_DIR.parent / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {FIG_DIR.parent / 'metrics.json'}")

    # Persist trained models for the dashboard to load at inference time
    print("\nSaving trained models…")
    best_reg_key = max((k for k in reg), key=lambda k: reg[k]["R2"])
    best_clf_key = max(clf, key=lambda k: clf[k]["F1"])
    artefacts = {
        "regressor.pkl":     reg[best_reg_key]["model"],
        "classifier.pkl":    clf[best_clf_key]["model"],
        "scaler.pkl":        scaler,
        "rf_regressor.pkl":  reg["Random Forest (BONUS)"]["model"],
    }
    for fname, obj in artefacts.items():
        with open(MODELS_DIR / fname, "wb") as f:
            pickle.dump(obj, f)
        print(f"  saved models/{fname}")

    meta = {
        "feature_names":  feature_names,
        "best_regressor": best_reg_key,
        "best_classifier": best_clf_key,
        "feature_means":  {n: float(v) for n, v in zip(feature_names, df_clean[feature_names].mean().values)},
        "feature_stds":   {n: float(v) for n, v in zip(feature_names, df_clean[feature_names].std().values)},
    }
    with open(MODELS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved models/meta.json")


if __name__ == "__main__":
    main()
