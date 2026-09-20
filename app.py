"""TwinCare-Glyco -- doctor-facing Digital Twin dashboard.

A pure inference/visualization consumer: it loads the artifacts produced by
`python src/pipeline.py` (a trained forecasting model, the fused patient
dataset, and the static EHR profiles) and never re-trains or recomputes the
data pipeline itself. That separation means the dashboard starts instantly
and can't drift from the model that was actually evaluated.

Run with:
    streamlit run dashboard/app.py
"""
from pathlib import Path
import sys

import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import (  # noqa: E402
    ALL_FEATURES, RISK_THRESHOLD, explain_dynamic_drivers, explain_static_risk_factors,
    forecast_to_risk_probability,
)

st.set_page_config(page_title="TwinCare-Glyco", layout="wide")

DATA_PATH = ROOT / "data" / "processed" / "patient_twin_dataset.csv"
STATIC_PATH = ROOT / "data" / "processed" / "static_patient_profiles.csv"
MODEL_PATH = ROOT / "models" / "deployed_model.joblib"
META_PATH = ROOT / "models" / "feature_columns.json"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    static_df = pd.read_csv(STATIC_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return df, static_df, meta


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "No trained artifacts found yet. Run `python src/pipeline.py` from the "
        "project root first -- it generates data/processed/ and models/."
    )
    st.stop()

df, static_df, meta = load_data()
model = load_model()
resid_std = meta["residual_std_val"]

RISK_BANDS = [
    (0.0, 0.15, "Low", "#2e7d32"),
    (0.15, 0.40, "Moderate", "#f9a825"),
    (0.40, 0.70, "High", "#ef6c00"),
    (0.70, 1.01, "Very High", "#c62828"),
]


def risk_band(p: float):
    for lo, hi, label, color in RISK_BANDS:
        if lo <= p < hi:
            return label, color
    return "Very High", "#c62828"


# ---------------------------------------------------------------------------
# Sidebar -- patient selector
# ---------------------------------------------------------------------------

st.sidebar.title("TwinCare-Glyco")
st.sidebar.caption("Digital Twin proof-of-concept -- Type 2 Diabetes glucose monitoring")

patient_ids = sorted(df["patient_id"].unique())
selected_patient = st.sidebar.selectbox("Select patient", patient_ids)
split_filter = st.sidebar.radio(
    "Timeline shown", ["test (unseen by the model)", "all monitored days"], index=0
)

patient_df = df[df["patient_id"] == selected_patient].sort_values("timestamp").reset_index(drop=True)
view_df = (
    patient_df[patient_df["split"] == "test"] if split_filter.startswith("test") else patient_df
).reset_index(drop=True)
profile = static_df[static_df["patient_id"] == selected_patient].iloc[0]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this prototype**\n\n"
    "Glucose in the raw sample dataset barely moves minute to minute, so a "
    "meal/exercise response layer (documented in `src/pipeline.py`) was added "
    "on top of the real trace to give the twin genuine dynamics to learn "
    "from. The chart below distinguishes the original sensor reading from "
    "this augmented signal."
)

# ---------------------------------------------------------------------------
# Header + static profile
# ---------------------------------------------------------------------------

st.title(f"Digital Twin -- Patient {selected_patient}")

latest = patient_df.iloc[-1]
label, color = risk_band(latest["predicted_hyperglycemia_risk"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Age", f"{int(profile['age'])}")
c2.metric("HbA1c", f"{profile['hba1c']}%")
c3.metric("BMI", f"{profile['bmi']}")
c4.metric("Medication", profile["medication"])
c5.metric("Diabetes duration", f"{int(profile['diabetes_duration_years'])} yrs")

st.markdown("### Current Digital Twin state")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Glucose now", f"{latest['glucose']:.0f} mg/dL")
c2.metric("Forecast (+60 min)", f"{latest['predicted_glucose_60min']:.0f} mg/dL",
          delta=f"{latest['predicted_glucose_60min'] - latest['glucose']:+.0f}")
c3.metric("Hyperglycemia risk", f"{latest['predicted_hyperglycemia_risk']*100:.0f}%")
c4.markdown(
    f"<div style='padding-top:8px'><span style='background-color:{color};"
    f"color:white;padding:6px 14px;border-radius:6px;font-weight:600'>{label} risk</span></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Trend charts
# ---------------------------------------------------------------------------

st.markdown("### Glucose trend and forecast")
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=view_df["timestamp"], y=view_df["glucose_measured"],
                          name="Raw sensor reading", line=dict(color="#90a4ae", width=1, dash="dot")))
fig.add_trace(go.Scatter(x=view_df["timestamp"], y=view_df["glucose"],
                          name="Digital Twin signal (augmented)", line=dict(color="#1565c0", width=2)))
fig.add_trace(go.Scatter(x=view_df["timestamp"], y=view_df["predicted_glucose_60min"],
                          name="60-min forecast", line=dict(color="#6a1b9a", width=2, dash="dash")))
fig.add_hline(y=RISK_THRESHOLD, line_dash="dot", line_color="red",
              annotation_text="Hyperglycemia threshold (140 mg/dL)")
fig.add_trace(go.Scatter(x=view_df["timestamp"], y=view_df["predicted_hyperglycemia_risk"],
                          name="Predicted risk", line=dict(color="#ef6c00", width=1.5), yaxis="y2"),
              secondary_y=True)
fig.update_yaxes(title_text="Glucose (mg/dL)", secondary_y=False)
fig.update_yaxes(title_text="Hyperglycemia risk probability", range=[0, 1], secondary_y=True)
fig.update_layout(height=460, legend=dict(orientation="h", y=1.12), margin=dict(t=40))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

row = patient_df.iloc[-1]


def _contribution_chart(contrib, title):
    fig = go.Figure(go.Bar(
        x=contrib["forecast_contribution_mgdl"], y=contrib["feature"], orientation="h",
        marker_color=["#c62828" if v > 0 else "#2e7d32" for v in contrib["forecast_contribution_mgdl"]],
    ))
    fig.update_layout(height=280, margin=dict(t=10, l=10), xaxis_title=title)
    return fig


exp1, exp2 = st.columns(2)
with exp1:
    st.markdown("#### Baseline risk factors (why this patient, generally)")
    static_contrib = explain_static_risk_factors(model, row, population_reference=df, top_k=6)
    st.plotly_chart(
        _contribution_chart(static_contrib, "Effect vs. a population-typical patient (mg/dL)"),
        use_container_width=True,
    )
with exp2:
    st.markdown("#### Right-now drivers (why this moment, for this patient)")
    dynamic_contrib = explain_dynamic_drivers(model, row, patient_history=patient_df, top_k=6)
    st.plotly_chart(
        _contribution_chart(dynamic_contrib, f"Effect vs. {selected_patient}'s own typical moment (mg/dL)"),
        use_container_width=True,
    )
st.caption(
    "Red = pushes the forecast up (toward hyperglycemia); green = pushes it down. "
    "Computed by single-feature occlusion, not a claim of causal effect size."
)

# ---------------------------------------------------------------------------
# What-if scenario simulator
# ---------------------------------------------------------------------------

st.markdown("### What-if scenario simulator")
st.caption("Adjust the patient's recent behaviour and see the Digital Twin's forecast update live.")

wc1, wc2, wc3 = st.columns(3)
carbs_override = wc1.slider("Carbs eaten in last 60 min (g)", 0, 120, int(row["carbs_60min_sum"]), step=5)
bolus_override = wc2.slider("Insulin bolus in last 60 min (units)", 0.0, 15.0, float(row["insulin_bolus_60min_sum"]), step=0.5)
steps_override = wc3.slider("Exercise steps in last 60 min", 0, 6000, int(row["exercise_steps_60min_sum"]), step=100)

scenario = row.copy()
scenario["carbs_60min_sum"] = carbs_override
scenario["insulin_bolus_60min_sum"] = bolus_override
scenario["exercise_steps_60min_sum"] = steps_override

scenario_forecast = float(model.predict(scenario.to_frame().T[ALL_FEATURES])[0])
scenario_risk = float(forecast_to_risk_probability(np.array([scenario_forecast]), resid_std)[0])

s1, s2, s3 = st.columns(3)
s1.metric("Baseline forecast", f"{row['predicted_glucose_60min']:.0f} mg/dL")
s2.metric("Scenario forecast", f"{scenario_forecast:.0f} mg/dL",
          delta=f"{scenario_forecast - row['predicted_glucose_60min']:+.0f}")
s3.metric("Scenario risk", f"{scenario_risk*100:.0f}%",
          delta=f"{(scenario_risk - row['predicted_hyperglycemia_risk'])*100:+.0f} pts")

st.caption(
    "This scenario reuses the same trained forecasting model with three inputs "
    "swapped -- it does not retrain anything. Effect sizes reflect the "
    "physiological augmentation documented in src/pipeline.py, not a claim "
    "about this specific patient's real physiology."
)

# ---------------------------------------------------------------------------
# Cohort overview
# ---------------------------------------------------------------------------

st.markdown("### Cohort risk overview (most recent reading per patient)")
latest_per_patient = df.sort_values("timestamp").groupby("patient_id").tail(1).sort_values(
    "predicted_hyperglycemia_risk", ascending=False
)
cohort_fig = go.Figure(go.Bar(
    x=latest_per_patient["predicted_hyperglycemia_risk"], y=latest_per_patient["patient_id"],
    orientation="h",
    marker_color=[risk_band(p)[1] for p in latest_per_patient["predicted_hyperglycemia_risk"]],
))
cohort_fig.update_layout(height=320, margin=dict(t=10, l=10), xaxis_title="Predicted hyperglycemia risk",
                          xaxis_range=[0, 1])
st.plotly_chart(cohort_fig, use_container_width=True)
