from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import BaselineProfile, DynamicState, fuse_features


FEATURE_NAMES = [
    "age",
    "diabetes",
    "hypertension",
    "hba1c",
    "bmi",
    "glucose",
    "glucose_slope",
    "glucose_variability",
    "heart_rate",
    "hrv",
    "sleep_hours",
    "steps",
]


def _generate_synthetic_training_data(n_samples: int = 4000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    age = rng.integers(30, 85, n_samples)
    diabetes = rng.binomial(1, 0.8, n_samples)
    hypertension = rng.binomial(1, 0.35, n_samples)
    hba1c = rng.normal(7.4, 1.1, n_samples).clip(4.8, 12.0)
    bmi = rng.normal(28, 4.8, n_samples).clip(18, 45)

    glucose = rng.normal(140, 35, n_samples).clip(70, 300)
    glucose_slope = rng.normal(6, 8, n_samples).clip(-20, 35)
    glucose_variability = rng.normal(28, 10, n_samples).clip(5, 70)
    heart_rate = rng.normal(82, 12, n_samples).clip(45, 150)
    hrv = rng.normal(38, 14, n_samples).clip(8, 120)
    sleep_hours = rng.normal(6.5, 1.5, n_samples).clip(2, 10)
    steps = rng.integers(200, 15000, n_samples)

    X = np.column_stack(
        [
            age,
            diabetes,
            hypertension,
            hba1c,
            bmi,
            glucose,
            glucose_slope,
            glucose_variability,
            heart_rate,
            hrv,
            sleep_hours,
            steps,
        ]
    )

    # Synthetic "event in next 2h" mechanism to keep the PoC explainable.
    score = (
        0.017 * (glucose - 130)
        + 0.07 * glucose_slope
        + 0.03 * (glucose_variability - 25)
        + 0.35 * (hba1c - 7.0)
        + 0.008 * (heart_rate - 80)
        - 0.015 * (hrv - 35)
        - 0.25 * (sleep_hours - 6.5)
        - 0.00009 * (steps - 4000)
        + 0.2 * hypertension
    )
    p = 1 / (1 + np.exp(-(score - 0.8)))
    y = rng.binomial(1, p)
    return X, y


def train_baseline_model(seed: int = 42) -> ClassifierMixin:
    X, y = _generate_synthetic_training_data(seed=seed)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed, solver="liblinear"),
    )
    model.fit(X, y)
    return model


def predict_spike_probability(model: ClassifierMixin, profile: BaselineProfile, state: DynamicState) -> float:
    x = np.array([fuse_features(profile, state)])
    return float(model.predict_proba(x)[0, 1])
