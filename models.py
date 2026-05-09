"""
CSE271 - Hospital Length-of-Stay Prediction

This script is the source of truth for the project:
1. Load and clean LengthOfStay.csv
2. Train regression and classification models with leak-safe preprocessing
3. Save trained model pipelines, metrics, feature importance, and plots

Run:
    python models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("LengthOfStay.csv")
PLOTS_DIR = Path("plots")
MODELS_DIR = Path("saved_models")
TARGET = "lengthofstay"
DROP_COLUMNS = ["eid", "vdate", "discharged", "facid"]
CLASS_LABELS = {
    0: "Short (0-3d)",
    1: "Medium (4-7d)",
    2: "Long (8+d)",
}


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


def load_data(filepath: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print("=" * 72)
    print("DATASET OVERVIEW")
    print("=" * 72)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(df.dtypes)
    return df


def categorize_los(days: int | float) -> int:
    if days <= 3:
        return 0
    if days <= 7:
        return 1
    return 2


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data = df.copy()
    data = data.drop(columns=[c for c in DROP_COLUMNS if c in data.columns])

    y_reg = data[TARGET].copy()
    y_clf = y_reg.apply(categorize_los)
    X = data.drop(columns=[TARGET])

    return X, y_reg, y_clf


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def model_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", estimator),
        ]
    )


def regression_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "Linear Regression": model_pipeline(X, LinearRegression()),
        "Decision Tree": model_pipeline(
            X, DecisionTreeRegressor(max_depth=8, random_state=42)
        ),
        "Random Forest (BONUS)": model_pipeline(
            X,
            RandomForestRegressor(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=42,
            ),
        ),
        "XGBoost (BONUS)": model_pipeline(
            X,
            xgb.XGBRegressor(
                n_estimators=180,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                n_jobs=1,
                random_state=42,
                verbosity=0,
            ),
        ),
    }


def classification_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": model_pipeline(
            X, LogisticRegression(max_iter=1200, random_state=42)
        ),
        "KNN": model_pipeline(X, KNeighborsClassifier(n_neighbors=7)),
        "Linear SVM (BONUS)": model_pipeline(
            X, LinearSVC(class_weight="balanced", max_iter=4000, random_state=42)
        ),
        "Random Forest Classifier (BONUS)": model_pipeline(
            X,
            RandomForestClassifier(
                n_estimators=140,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=1,
                random_state=42,
            ),
        ),
        "Neural Network (BONUS)": model_pipeline(
            X,
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                max_iter=180,
                random_state=42,
            ),
        ),
    }


def evaluate_regression(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, dict], str]:
    results: dict[str, dict] = {}
    print("\n" + "=" * 72)
    print("REGRESSION RESULTS")
    print("=" * 72)

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = mean_absolute_error(y_test, preds)
        results[name] = {
            "R2": float(r2),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "predictions": preds.tolist(),
        }

        print(f"\n{name}")
        print(f"  R2   = {r2:.4f}")
        print(f"  RMSE = {rmse:.4f} days")
        print(f"  MAE  = {mae:.4f} days")
        print(
            "  So what? "
            + (
                "Strong enough for bed-capacity planning."
                if r2 >= 0.75
                else "Useful as a baseline, but not enough for individual planning."
            )
        )

    best_name = max(results, key=lambda n: results[n]["R2"])
    return results, best_name


def evaluate_classification(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, dict], str]:
    results: dict[str, dict] = {}
    print("\n" + "=" * 72)
    print("CLASSIFICATION RESULTS")
    print("=" * 72)

    target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)]

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        long_recall = recall_score(
            y_test == 2,
            preds == 2,
            zero_division=0,
        )

        results[name] = {
            "Accuracy": float(acc),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "LongStayRecall": float(long_recall),
            "confusion_matrix": cm.tolist(),
            "predictions": preds.tolist(),
            "classification_report": classification_report(
                y_test,
                preds,
                target_names=target_names,
                zero_division=0,
            ),
        }

        print(f"\n{name}")
        print(f"  Accuracy         = {acc:.4f}")
        print(f"  Precision        = {precision:.4f}")
        print(f"  Recall           = {recall:.4f}")
        print(f"  F1-score         = {f1:.4f}")
        print(f"  Long-stay recall = {long_recall:.4f}")
        print("  So what? Long-stay recall matters because missed long stays affect beds.")

    best_name = max(results, key=lambda n: results[n]["F1"])
    return results, best_name


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocess"]
    return preprocessor.get_feature_names_out().tolist()


def save_feature_importance(best_regression: Pipeline) -> pd.DataFrame:
    feature_names = get_feature_names(best_regression)
    estimator = best_regression.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).ravel()
    else:
        values = np.zeros(len(feature_names))

    importance = (
        pd.DataFrame({"Feature": feature_names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    return importance


def write_artifacts(
    X: pd.DataFrame,
    y_test_reg: pd.Series,
    y_test_clf: pd.Series,
    reg_results: dict[str, dict],
    clf_results: dict[str, dict],
    best_reg_name: str,
    best_clf_name: str,
    reg_models: dict[str, Pipeline],
    clf_models: dict[str, Pipeline],
    feature_importance: pd.DataFrame,
) -> None:
    joblib.dump(reg_models[best_reg_name], MODELS_DIR / "regression_model.pkl")
    joblib.dump(clf_models[best_clf_name], MODELS_DIR / "classification_model.pkl")

    metadata = {
        "feature_columns": X.columns.tolist(),
        "class_labels": CLASS_LABELS,
        "best_regression_model": best_reg_name,
        "best_classification_model": best_clf_name,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics = {
        "best_regression_model": best_reg_name,
        "best_classification_model": best_clf_name,
        "regression": {
            name: {k: v for k, v in values.items() if k != "predictions"}
            for name, values in reg_results.items()
        },
        "classification": {
            name: {k: v for k, v in values.items() if k != "predictions"}
            for name, values in clf_results.items()
        },
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    y_test_reg.to_csv(MODELS_DIR / "y_test_regression.csv", index=False)
    y_test_clf.to_csv(MODELS_DIR / "y_test_classification.csv", index=False)
    feature_importance.head(25).to_csv(MODELS_DIR / "top_features.csv", index=False)


def generate_plots(
    df_raw: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test_reg: pd.Series,
    y_test_clf: pd.Series,
    reg_results: dict[str, dict],
    clf_results: dict[str, dict],
    best_reg_name: str,
    best_clf_name: str,
    feature_importance: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    los_category = df_raw[TARGET].apply(lambda d: CLASS_LABELS[categorize_los(d)])

    plt.figure(figsize=(10, 5))
    sns.histplot(df_raw[TARGET], bins=30, color="#2563eb")
    plt.axvline(3, color="#16a34a", linestyle="--", label="Short/Medium threshold")
    plt.axvline(7, color="#dc2626", linestyle="--", label="Medium/Long threshold")
    plt.title("Distribution of Hospital Length of Stay")
    plt.xlabel("Length of Stay (days)")
    plt.ylabel("Patient Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_01_los_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    los_category.value_counts().reindex(CLASS_LABELS.values()).plot(
        kind="bar", color=["#16a34a", "#f59e0b", "#dc2626"]
    )
    plt.title("LOS Category Breakdown")
    plt.xlabel("LOS Category")
    plt.ylabel("Patient Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_02_category_breakdown.png", dpi=150)
    plt.close()

    numeric = df_raw.select_dtypes(include=np.number)
    corr_with_target = numeric.corr(numeric_only=True)[TARGET].sort_values()
    plt.figure(figsize=(9, 6))
    corr_with_target.drop(TARGET).plot(kind="barh", color="#2563eb")
    plt.title("Numeric Feature Correlation with LOS")
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_03_correlations.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric.corr(numeric_only=True), cmap="RdBu_r", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_04_heatmap.png", dpi=150)
    plt.close()

    reg_df = pd.DataFrame(
        [
            {"Model": name, "R2": vals["R2"], "RMSE": vals["RMSE"]}
            for name, vals in reg_results.items()
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=reg_df, x="Model", y="R2", ax=axes[0], color="#2563eb")
    sns.barplot(data=reg_df, x="Model", y="RMSE", ax=axes[1], color="#16a34a")
    axes[0].set_title("Regression R2")
    axes[1].set_title("Regression RMSE")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_05_regression_comparison.png", dpi=150)
    plt.close()

    best_reg_preds = np.array(reg_results[best_reg_name]["predictions"])
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_reg, best_reg_preds, alpha=0.25, s=10, color="#2563eb")
    plt.plot(
        [y_test_reg.min(), y_test_reg.max()],
        [y_test_reg.min(), y_test_reg.max()],
        color="#dc2626",
        linestyle="--",
    )
    plt.title(f"Actual vs Predicted LOS - {best_reg_name}")
    plt.xlabel("Actual LOS (days)")
    plt.ylabel("Predicted LOS (days)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_06_actual_vs_predicted.png", dpi=150)
    plt.close()

    clf_df = pd.DataFrame(
        [
            {"Model": name, "Accuracy": vals["Accuracy"], "F1": vals["F1"]}
            for name, vals in clf_results.items()
        ]
    )
    clf_df.plot(x="Model", y=["Accuracy", "F1"], kind="bar", figsize=(11, 5))
    plt.title("Classification Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_07_classification_comparison.png", dpi=150)
    plt.close()

    cm = np.array(clf_results[best_clf_name]["confusion_matrix"])
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[CLASS_LABELS[i] for i in sorted(CLASS_LABELS)],
        yticklabels=[CLASS_LABELS[i] for i in sorted(CLASS_LABELS)],
    )
    plt.title(f"Confusion Matrix - {best_clf_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_08_confusion_matrices.png", dpi=150)
    plt.close()

    top_features = feature_importance.head(15).sort_values("Importance")
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["Feature"], top_features["Importance"], color="#2563eb")
    plt.title("Top Features Driving LOS")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_09_feature_importance.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_dirs()
    df = load_data()
    X, y_reg, y_clf = prepare_features(df)

    print("\n" + "=" * 72)
    print("PREPROCESSING PLAN")
    print("=" * 72)
    print(f"Features used: {X.shape[1]}")
    print("Categorical columns are one-hot encoded; numeric columns are imputed and scaled.")
    print("Preprocessing is fitted inside each model pipeline after train/test split.")

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=0.2,
        random_state=42,
        stratify=y_clf,
    )

    reg_models = regression_models(X)
    clf_models = classification_models(X)

    reg_results, best_reg_name = evaluate_regression(
        reg_models, X_train, X_test, y_reg_train, y_reg_test
    )
    clf_results, best_clf_name = evaluate_classification(
        clf_models, X_train, X_test, y_clf_train, y_clf_test
    )

    feature_importance = save_feature_importance(reg_models[best_reg_name])
    write_artifacts(
        X,
        y_reg_test,
        y_clf_test,
        reg_results,
        clf_results,
        best_reg_name,
        best_clf_name,
        reg_models,
        clf_models,
        feature_importance,
    )
    generate_plots(
        df,
        X_test,
        y_reg_test,
        y_clf_test,
        reg_results,
        clf_results,
        best_reg_name,
        best_clf_name,
        feature_importance,
    )

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print(f"Best regression model: {best_reg_name}")
    print(f"Best classification model: {best_clf_name}")
    print(f"Saved trained models and metrics to: {MODELS_DIR.resolve()}")
    print(f"Saved plots to: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
