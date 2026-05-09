"""
CSE271 - Hospital Length of Stay Prediction
Complete ML Pipeline: Regression + Classification (+ Bonus Models)
Dataset: Microsoft Hospital LOS Dataset (Kaggle - aayushchou)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & INSPECT DATA
# ─────────────────────────────────────────────
def load_data(filepath="LengthOfStay.csv"):
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()

    # Drop leakage columns (discharge date leaks the target)
    drop_cols = ['eid', 'vdate', 'discharged', 'facid']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Convert Yes/No flags to binary
    yn_cols = [c for c in df.columns if df[c].dtype == object and
               set(df[c].dropna().unique()).issubset({'Yes', 'No'})]
    for col in yn_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Encode remaining categoricals
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # ── Regression target: lengthofstay (continuous)
    y_reg = df['lengthofstay'].copy()

    # ── Classification target: LOS category
    def categorize_los(days):
        if days <= 3:   return 0   # Short
        elif days <= 7: return 1   # Medium
        else:           return 2   # Long

    y_clf = y_reg.apply(categorize_los)

    # Features (drop target)
    X = df.drop(columns=['lengthofstay'])

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Features: {X.shape[1]}")
    print(f"LOS distribution — Short(0-3d): {(y_clf==0).sum():,}  "
          f"Medium(4-7d): {(y_clf==1).sum():,}  Long(8+d): {(y_clf==2).sum():,}")

    return X_scaled, y_reg, y_clf, X.columns.tolist()


# ─────────────────────────────────────────────
# 3. REGRESSION MODELS
# ─────────────────────────────────────────────
def run_regression(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression":         LinearRegression(),
        "Decision Tree Regressor":   DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest (BONUS)":     RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost (BONUS)":           xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }

    results = {}
    print("\n" + "=" * 60)
    print("REGRESSION RESULTS")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2   = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        results[name] = {"model": model, "preds": preds,
                         "R2": r2, "RMSE": rmse, "MAE": mae,
                         "y_test": y_test}

        tag = " ★ BONUS" if "BONUS" in name else ""
        print(f"\n{name}{tag}")
        print(f"  R²   = {r2:.4f}  → Model explains {r2*100:.1f}% of LOS variance")
        print(f"  RMSE = {rmse:.4f} days → Avg prediction error")
        print(f"  MAE  = {mae:.4f} days → Avg absolute error")

        # SO WHAT interpretations
        if r2 > 0.7:
            print(f"  ✅ Strong fit. Clinically useful for bed scheduling.")
        elif r2 > 0.4:
            print(f"  ⚠️  Moderate fit. Useful for triage, not for individual beds.")
        else:
            print(f"  ❌ Weak fit. Social/environmental factors likely missing.")

    # Feature importance from RF
    rf = results["Random Forest (BONUS)"]["model"]
    fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    results["feature_importance"] = fi

    return results, X_test, y_test


# ─────────────────────────────────────────────
# 4. CLASSIFICATION MODELS
# ─────────────────────────────────────────────
def run_classification(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors":    KNeighborsClassifier(n_neighbors=5),
        "SVM (BONUS)":            SVC(kernel='rbf', probability=True, random_state=42),
        "Neural Network (BONUS)": MLPClassifier(hidden_layer_sizes=(64, 32),
                                                max_iter=300, random_state=42),
    }

    results = {}
    class_labels = ["Short (0-3d)", "Medium (4-7d)", "Long (8+d)"]

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec  = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1   = f1_score(y_test, preds, average='weighted', zero_division=0)
        cm   = confusion_matrix(y_test, preds)
        results[name] = {"model": model, "preds": preds, "cm": cm,
                         "Accuracy": acc, "Precision": prec,
                         "Recall": rec, "F1": f1, "y_test": y_test}

        tag = " ★ BONUS" if "BONUS" in name else ""
        print(f"\n{name}{tag}")
        print(f"  Accuracy  = {acc:.4f}")
        print(f"  Precision = {prec:.4f}")
        print(f"  Recall    = {rec:.4f}")
        print(f"  F1-Score  = {f1:.4f}")

        # SO WHAT
        fp_rate = cm[0, 2] / cm[0].sum() if cm[0].sum() > 0 else 0
        if fp_rate > 0.1:
            print(f"  ⚠️  {fp_rate*100:.1f}% short-stay patients predicted Long — "
                  f"wastes {int(fp_rate * cm[0].sum())} beds/batch")
        print(f"  📋 {classification_report(y_test, preds, target_names=class_labels, zero_division=0)}")

    return results, X_test, y_test


# ─────────────────────────────────────────────
# 5. VISUALIZATIONS (for report + dashboard)
# ─────────────────────────────────────────────
def generate_plots(df_raw, reg_results, clf_results, feature_names):
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#2563EB', '#16A34A', '#DC2626', '#D97706']

    os.makedirs("plots", exist_ok=True)

    # ── Plot 1: LOS Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_raw['lengthofstay'], bins=40, color='#2563EB', edgecolor='white', alpha=0.85)
    ax.axvline(3, color='#16A34A', linestyle='--', lw=2, label='Short/Medium threshold (3d)')
    ax.axvline(7, color='#DC2626', linestyle='--', lw=2, label='Medium/Long threshold (7d)')
    ax.set_xlabel("Length of Stay (days)", fontsize=13)
    ax.set_ylabel("Patient Count", fontsize=13)
    ax.set_title("Distribution of Hospital Length of Stay", fontsize=15, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/01_los_distribution.png", dpi=150)
    plt.close()

    # ── Plot 2: Regression model comparison
    names = list(reg_results.keys())
    r2s   = [reg_results[n]['R2']   for n in names if n != 'feature_importance']
    rmses = [reg_results[n]['RMSE'] for n in names if n != 'feature_importance']
    names_clean = [n for n in names if n != 'feature_importance']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = axes[0].bar(names_clean, r2s, color=colors[:len(names_clean)], edgecolor='white')
    axes[0].set_title("Regression — R² Score", fontweight='bold')
    axes[0].set_ylabel("R²")
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars, r2s):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha='center', fontsize=10)
    axes[0].tick_params(axis='x', rotation=15)

    bars2 = axes[1].bar(names_clean, rmses, color=colors[:len(names_clean)], edgecolor='white')
    axes[1].set_title("Regression — RMSE (lower = better)", fontweight='bold')
    axes[1].set_ylabel("RMSE (days)")
    for bar, val in zip(bars2, rmses):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.3f}", ha='center', fontsize=10)
    axes[1].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig("plots/02_regression_comparison.png", dpi=150)
    plt.close()

    # ── Plot 3: Classification comparison
    clf_names = list(clf_results.keys())
    accs = [clf_results[n]['Accuracy'] for n in clf_names]
    f1s  = [clf_results[n]['F1']       for n in clf_names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(clf_names, accs, color=colors[:len(clf_names)], edgecolor='white')
    axes[0].set_title("Classification — Accuracy", fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=15)

    axes[1].bar(clf_names, f1s, color=colors[:len(clf_names)], edgecolor='white')
    axes[1].set_title("Classification — F1 Score (Weighted)", fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig("plots/03_classification_comparison.png", dpi=150)
    plt.close()

    # ── Plot 4: Confusion matrix for best classifier
    best_clf = max(clf_results, key=lambda k: clf_results[k]['F1'])
    cm = clf_results[best_clf]['cm']
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=["Short", "Medium", "Long"],
                yticklabels=["Short", "Medium", "Long"])
    ax.set_title(f"Confusion Matrix — {best_clf}", fontweight='bold')
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("plots/04_confusion_matrix.png", dpi=150)
    plt.close()

    # ── Plot 5: Feature Importance
    fi = reg_results['feature_importance'].head(12)
    fig, ax = plt.subplots(figsize=(10, 6))
    fi.sort_values().plot(kind='barh', ax=ax, color='#2563EB', edgecolor='white')
    ax.set_title("Top 12 Features Driving LOS (Random Forest)", fontweight='bold')
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plots/05_feature_importance.png", dpi=150)
    plt.close()

    # ── Plot 6: Scatter — actual vs predicted (best regression)
    best_reg = max((k for k in reg_results if k != 'feature_importance'),
                   key=lambda k: reg_results[k]['R2'])
    y_test = reg_results[best_reg]['y_test']
    preds  = reg_results[best_reg]['preds']
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.3, s=10, color='#2563EB')
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual LOS (days)")
    ax.set_ylabel("Predicted LOS (days)")
    ax.set_title(f"Actual vs Predicted — {best_reg}", fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/06_actual_vs_predicted.png", dpi=150)
    plt.close()

    print("\n✅ All plots saved to /plots/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
import os

if __name__ == "__main__":
    # ── Load
    df = load_data("LengthOfStay.csv")

    # ── Preprocess
    X, y_reg, y_clf, feature_names = preprocess(df)

    # ── Regression
    reg_results, X_test_reg, y_test_reg = run_regression(X, y_reg, feature_names)

    # ── Classification
    clf_results, X_test_clf, y_test_clf = run_classification(X, y_clf, feature_names)

    # ── Plots
    generate_plots(df, reg_results, clf_results, feature_names)

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE — check /plots/ for all visualizations")
    print("=" * 60)
