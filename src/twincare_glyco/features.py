from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineProfile:
    age: int
    diabetes: int
    hypertension: int
    hba1c: float
    bmi: float


@dataclass(frozen=True)
class DynamicState:
    glucose: float
    glucose_slope: float
    glucose_variability: float
    heart_rate: float
    hrv: float
    sleep_hours: float
    steps: int


def fuse_features(profile: BaselineProfile, state: DynamicState) -> list[float]:
    return [
        float(profile.age),
        float(profile.diabetes),
        float(profile.hypertension),
        float(profile.hba1c),
        float(profile.bmi),
        float(state.glucose),
        float(state.glucose_slope),
        float(state.glucose_variability),
        float(state.heart_rate),
        float(state.hrv),
        float(state.sleep_hours),
        float(state.steps),
    ]


def risk_band(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Moderate"
    if probability < 0.80:
        return "High"
    return "Very High"
