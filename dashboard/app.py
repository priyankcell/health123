from __future__ import annotations

import pathlib
import sys

import pandas as pd
import streamlit as st

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from twincare_glyco.features import BaselineProfile, DynamicState, risk_band
from twincare_glyco.model import train_baseline_model, predict_spike_probability


st.set_page_config(page_title="TwinCare-Glyco", layout="wide")
st.title("DIGITAL TWIN — Patient P102")
st.caption("Type 2 Diabetes | Forecast horizon: next 2 hours")

model = train_baseline_model()

profile = BaselineProfile(
    age=58,
    diabetes=1,
    hypertension=1,
    hba1c=8.2,
    bmi=29.7,
)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Current state")
    glucose = st.slider("Glucose (mg/dL)", 70, 300, 158)
    glucose_slope = st.slider("Glucose slope (mg/dL/hour)", -20, 35, 18)
    glucose_variability = st.slider("Glucose variability", 5, 70, 30)
    heart_rate = st.slider("Heart rate (bpm)", 45, 150, 92)
    hrv = st.slider("HRV (ms)", 8, 120, 31)
    sleep_hours = st.slider("Sleep (hours)", 2.0, 10.0, 4.8, step=0.1)
    steps = st.slider("Steps", 200, 15000, 2100, step=100)

state = DynamicState(
    glucose=float(glucose),
    glucose_slope=float(glucose_slope),
    glucose_variability=float(glucose_variability),
    heart_rate=float(heart_rate),
    hrv=float(hrv),
    sleep_hours=float(sleep_hours),
    steps=int(steps),
)

probability = predict_spike_probability(model, profile, state)
band = risk_band(probability)

with col2:
    st.subheader("Prediction")
    st.metric("Glucose spike risk", f"{probability:.0%}")
    st.metric("Risk band", band)

st.subheader("Twin status over time")
timeline_states = [
    DynamicState(140, 5, 24, 84, 40, 6.7, 3800),
    DynamicState(145, 8, 27, 86, 37, 6.2, 3300),
    DynamicState(151, 12, 29, 89, 34, 5.6, 2700),
    state,
]
timeline_hours = ["08:00", "09:00", "10:00", "11:00"]
timeline_probs = [predict_spike_probability(model, profile, ts) for ts in timeline_states]
timeline_df = pd.DataFrame({"time": timeline_hours, "risk_probability": timeline_probs})
st.line_chart(timeline_df.set_index("time"))
st.dataframe(
    pd.DataFrame(
        {
            "time": timeline_hours,
            "risk": [f"{p:.0%}" for p in timeline_probs],
            "band": [risk_band(p) for p in timeline_probs],
        }
    ),
    hide_index=True,
)

st.subheader("Scenario simulation")
sim_col1, sim_col2 = st.columns(2)
with sim_col1:
    extra_activity = st.slider("Extra activity (minutes)", 0, 90, 30, step=5)
with sim_col2:
    extra_sleep = st.slider("Extra sleep (hours)", 0.0, 2.0, 1.0, step=0.1)

simulated_steps = min(15000, steps + extra_activity * 100)
simulated_sleep = min(10.0, sleep_hours + extra_sleep)
sim_state = DynamicState(
    glucose=float(glucose),
    glucose_slope=float(glucose_slope - (extra_activity * 0.05)),
    glucose_variability=float(glucose_variability),
    heart_rate=float(max(45, heart_rate - extra_activity * 0.1)),
    hrv=float(min(120, hrv + extra_sleep * 2.0)),
    sleep_hours=float(simulated_sleep),
    steps=int(simulated_steps),
)
sim_prob = predict_spike_probability(model, profile, sim_state)

st.write(f"Current risk: **{probability:.0%}** ({band})")
st.write(f"Simulated risk: **{sim_prob:.0%}** ({risk_band(sim_prob)})")
