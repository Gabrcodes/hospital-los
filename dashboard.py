"""
CSE271 - Hospital LOS Dashboard

Run after training:
    python models.py
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


DATA_PATH = Path("LengthOfStay.csv")
MODELS_DIR = Path("saved_models")
PLOTS_DIR = Path("plots")
DROP_COLUMNS = ["eid", "vdate", "discharged", "facid"]
CLASS_LABELS = {
    0: "Short (0-3d)",
    1: "Medium (4-7d)",
    2: "Long (8+d)",
}


st.set_page_config(
    page_title="Hospital LOS Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    [data-testid="stAppViewContainer"] { background: #0f172a; color: #e2e8f0; }
    [data-testid="stSidebar"] { background: #1e293b; }
    h1, h2, h3 { color: #f8fafc !important; }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        min-height: 96px;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .insight-box {
        background: #172554;
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        color: #dbeafe;
    }
</style>
""",
    unsafe_allow_html=True,
)


def categorize_los(days: int | float) -> str:
    if days <= 3:
        return CLASS_LABELS[0]
    if days <= 7:
        return CLASS_LABELS[1]
    return CLASS_LABELS[2]


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["los_category"] = df["lengthofstay"].apply(categorize_los)
    return df


@st.cache_resource
def load_artifacts():
    required = [
        MODELS_DIR / "regression_model.pkl",
        MODELS_DIR / "classification_model.pkl",
        MODELS_DIR / "metrics.json",
        MODELS_DIR / "metadata.json",
        MODELS_DIR / "feature_importance.csv",
    ]
    if not all(path.exists() for path in required):
        return None

    return {
        "regression_model": joblib.load(MODELS_DIR / "regression_model.pkl"),
        "classification_model": joblib.load(MODELS_DIR / "classification_model.pkl"),
        "metrics": json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8")),
        "metadata": json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8")),
        "feature_importance": pd.read_csv(MODELS_DIR / "feature_importance.csv"),
    }


def default_patient(df: pd.DataFrame, feature_columns: list[str]) -> dict:
    source = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    source = source.drop(columns=["lengthofstay", "los_category"], errors="ignore")
    values = {}

    for col in feature_columns:
        if col not in source.columns:
            values[col] = 0
        elif pd.api.types.is_numeric_dtype(source[col]):
            values[col] = float(source[col].median())
        else:
            values[col] = str(source[col].mode().iloc[0])
    return values


def metric_card(value: str, label: str, color: str = "#38bdf8") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{color}">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    df = load_data()
except FileNotFoundError:
    st.error("LengthOfStay.csv was not found. Place it beside dashboard.py.")
    st.stop()

artifacts = load_artifacts()

st.sidebar.markdown("## Filters")
los_range = st.sidebar.slider(
    "Length of Stay (days)",
    int(df["lengthofstay"].min()),
    int(df["lengthofstay"].max()),
    (int(df["lengthofstay"].min()), int(df["lengthofstay"].max())),
)
category_filter = st.sidebar.multiselect(
    "LOS Category",
    options=list(CLASS_LABELS.values()),
    default=list(CLASS_LABELS.values()),
)

df_filtered = df[
    (df["lengthofstay"].between(los_range[0], los_range[1]))
    & (df["los_category"].isin(category_filter))
]

st.sidebar.markdown("---")
st.sidebar.markdown("**CSE271 - Spring 2026**")
st.sidebar.markdown("Hospital LOS Prediction Framework")

st.markdown("# 🏥 Hospital Length-of-Stay Prediction")
st.markdown("**Interactive dashboard using the trained artifacts from `models.py`**")

if artifacts is None:
    st.warning(
        "Saved models were not found. Run `python models.py` first, then reload this dashboard."
    )

tab_data, tab_predict, tab_models, tab_features = st.tabs(
    ["Data Explorer", "Patient Predictor", "Model Comparison", "Feature Importance"]
)

with tab_data:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(f"{len(df_filtered):,}", "Filtered Patients")
    with col2:
        metric_card(f"{df_filtered['lengthofstay'].mean():.1f}d", "Average LOS")
    with col3:
        long_pct = (df_filtered["los_category"] == CLASS_LABELS[2]).mean() * 100
        metric_card(f"{long_pct:.1f}%", "Long-Stay Share", "#f87171")
    with col4:
        metric_card(f"{df_filtered['lengthofstay'].median():.0f}d", "Median LOS")

    st.markdown("### LOS Distribution")
    fig = px.histogram(
        df_filtered,
        x="lengthofstay",
        color="los_category",
        nbins=35,
        labels={"lengthofstay": "Length of Stay (days)", "los_category": "LOS Category"},
        color_discrete_map={
            CLASS_LABELS[0]: "#22c55e",
            CLASS_LABELS[1]: "#f59e0b",
            CLASS_LABELS[2]: "#ef4444",
        },
        template="plotly_dark",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Category Breakdown")
    counts = df_filtered["los_category"].value_counts().reindex(CLASS_LABELS.values()).fillna(0)
    fig_pie = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        color_discrete_map={
            CLASS_LABELS[0]: "#22c55e",
            CLASS_LABELS[1]: "#f59e0b",
            CLASS_LABELS[2]: "#ef4444",
        },
        template="plotly_dark",
    )
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_pie, use_container_width=True)

with tab_predict:
    st.markdown("### Predict LOS for a New Patient")

    if artifacts is None:
        st.info("The predictor will be available after running `python models.py`.")
    else:
        feature_columns = artifacts["metadata"]["feature_columns"]
        patient = default_patient(df, feature_columns)
        source = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            patient["gender"] = st.selectbox(
                "Gender",
                sorted(source["gender"].dropna().unique().tolist()),
                index=0,
            )
            patient["rcount"] = st.selectbox(
                "Readmission Count",
                sorted(source["rcount"].dropna().unique().tolist()),
                index=0,
            )
            patient["hematocrit"] = st.slider(
                "Hematocrit",
                float(source["hematocrit"].min()),
                float(source["hematocrit"].max()),
                float(source["hematocrit"].median()),
            )
            patient["neutrophils"] = st.slider(
                "Neutrophils",
                float(source["neutrophils"].min()),
                float(source["neutrophils"].max()),
                float(source["neutrophils"].median()),
            )

        with col_b:
            patient["sodium"] = st.slider(
                "Sodium",
                float(source["sodium"].min()),
                float(source["sodium"].max()),
                float(source["sodium"].median()),
            )
            patient["glucose"] = st.slider(
                "Glucose",
                float(source["glucose"].min()),
                float(source["glucose"].max()),
                float(source["glucose"].median()),
            )
            patient["bloodureanitro"] = st.slider(
                "Blood Urea Nitrogen",
                float(source["bloodureanitro"].min()),
                float(source["bloodureanitro"].max()),
                float(source["bloodureanitro"].median()),
            )
            patient["creatinine"] = st.slider(
                "Creatinine",
                float(source["creatinine"].min()),
                float(source["creatinine"].max()),
                float(source["creatinine"].median()),
            )

        with col_c:
            patient["bmi"] = st.slider(
                "BMI",
                float(source["bmi"].min()),
                float(source["bmi"].max()),
                float(source["bmi"].median()),
            )
            patient["pulse"] = st.slider(
                "Pulse",
                int(source["pulse"].min()),
                int(source["pulse"].max()),
                int(source["pulse"].median()),
            )
            patient["respiration"] = st.slider(
                "Respiration",
                float(source["respiration"].min()),
                float(source["respiration"].max()),
                float(source["respiration"].median()),
            )
            patient["secondarydiagnosisnonicd9"] = st.slider(
                "Secondary Diagnoses",
                int(source["secondarydiagnosisnonicd9"].min()),
                int(source["secondarydiagnosisnonicd9"].max()),
                int(source["secondarydiagnosisnonicd9"].median()),
            )

        st.markdown("#### Clinical Flags")
        flag_cols = [
            c
            for c in feature_columns
            if c in source.columns
            and pd.api.types.is_numeric_dtype(source[c])
            and set(source[c].dropna().unique()).issubset({0, 1})
        ]
        flag_columns = st.columns(4)
        for i, col in enumerate(flag_cols):
            with flag_columns[i % 4]:
                patient[col] = int(st.checkbox(col, value=bool(patient[col])))

        if st.button("Predict LOS", use_container_width=True):
            input_df = pd.DataFrame([{col: patient[col] for col in feature_columns}])
            pred_days = float(artifacts["regression_model"].predict(input_df)[0])
            pred_class = int(artifacts["classification_model"].predict(input_df)[0])
            pred_label = CLASS_LABELS.get(pred_class, str(pred_class))

            color = {
                CLASS_LABELS[0]: "#22c55e",
                CLASS_LABELS[1]: "#f59e0b",
                CLASS_LABELS[2]: "#ef4444",
            }.get(pred_label, "#38bdf8")

            res1, res2 = st.columns(2)
            with res1:
                metric_card(f"{max(pred_days, 1):.1f} days", "Predicted LOS", color)
            with res2:
                metric_card(pred_label, "Predicted Risk Class", color)

            st.markdown(
                """
                <div class="insight-box">
                This prediction is produced by the trained pipelines saved by models.py,
                so preprocessing, encoding, scaling, and model logic match the training run.
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_models:
    st.markdown("### Model Performance")

    if artifacts is None:
        st.info("Run `python models.py` to generate the model metrics.")
    else:
        metrics = artifacts["metrics"]

        reg_rows = [
            {"Model": model, **values}
            for model, values in metrics["regression"].items()
        ]
        reg_df = pd.DataFrame(reg_rows)
        fig_reg = make_subplots(rows=1, cols=2, subplot_titles=("R² Score", "RMSE"))
        fig_reg.add_trace(
            go.Bar(x=reg_df["Model"], y=reg_df["R2"], marker_color="#38bdf8"),
            row=1,
            col=1,
        )
        fig_reg.add_trace(
            go.Bar(x=reg_df["Model"], y=reg_df["RMSE"], marker_color="#22c55e"),
            row=1,
            col=2,
        )
        fig_reg.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_reg, use_container_width=True)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        clf_rows = [
            {
                "Model": model,
                "Accuracy": values["Accuracy"],
                "Precision": values["Precision"],
                "Recall": values["Recall"],
                "F1": values["F1"],
                "LongStayRecall": values["LongStayRecall"],
            }
            for model, values in metrics["classification"].items()
        ]
        clf_df = pd.DataFrame(clf_rows)
        fig_clf = px.bar(
            clf_df,
            x="Model",
            y=["Accuracy", "F1", "LongStayRecall"],
            barmode="group",
            template="plotly_dark",
        )
        fig_clf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_clf, use_container_width=True)
        st.dataframe(clf_df, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            <div class="insight-box">
            Best regression model: <b>{metrics["best_regression_model"]}</b>.
            Best classification model: <b>{metrics["best_classification_model"]}</b>.
            The dashboard reads these values from saved_models/metrics.json.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_features:
    st.markdown("### Feature Importance")

    if artifacts is None:
        st.info("Run `python models.py` to generate feature importance.")
    else:
        fi = artifacts["feature_importance"].head(20).sort_values("Importance")
        fig = px.bar(
            fi,
            x="Importance",
            y="Feature",
            orientation="h",
            template="plotly_dark",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        numeric = df.select_dtypes(include=np.number)
        corr = numeric.corr(numeric_only=True)
        fig_hm = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            template="plotly_dark",
        )
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            title="Numeric Feature Correlation Matrix",
        )
        st.plotly_chart(fig_hm, use_container_width=True)
