"""TwinCare-Glyco data pipeline.

Fuses a synthetic-but-clinically-grounded static EHR profile with dynamic
CGM/wearable-style data (the GlucoBench sample) into a per-patient Digital
Twin state, forecasts near-term glucose, derives a hyperglycemia-risk
probability from that forecast, and saves every artifact the dashboard needs.

Read this before the code -- three decisions shape everything below, and each
one exists because we measured the data first instead of assuming it would
behave like a typical CGM trace.

1. STATIC PROFILE FROM A HELD-OUT BASELINE WINDOW.
   Each patient's first BASELINE_DAYS of monitoring are used ONLY to derive
   their static EHR profile (age, BMI, HbA1c, medication, ...). The
   supervised task (train/val/test) never sees that window. This mirrors how
   a real EHR extract works -- HbA1c reflects a period *before* live
   monitoring starts -- and keeps the static features from leaking
   information about the exact rows being predicted.

2. THE RAW SAMPLE HAS NO LEARNABLE SHORT-TERM DYNAMICS -- SO WE SAY SO, AND
   FIX IT TRANSPARENTLY INSTEAD OF HIDING IT.
   A direct audit of the raw `GlucoBench_benchmark_dataset-selected-columns`
   file (see notebooks/TwinCare_Glyco.ipynb, section 2) shows glucose here
   moves in a very narrow, slow-drifting band per patient: the largest
   60-minute change anywhere in the whole file is under 12 mg/dL, even
   immediately after a 90g-carb meal. Two things follow from that:
     - A "will glucose spike in the next hour" classifier is, in this
       sample, almost perfectly explained by "is glucose already high right
       now" -- a naive persistence rule gets ~99.8% precision/recall for
       free. Training a model on that target would look impressive and mean
       nothing.
     - A 60-minute point forecast is dominated the same way: a plain
       persistence baseline (predict "no change") beats both a Ridge
       regression and a Random Forest fit on the fused features (MAE ~1.5
       mg/dL vs 1.5-2.5 mg/dL). There's nothing left for a model to learn.
   The project brief explicitly allows generating synthetic time-series data
   when a suitable dynamic dataset isn't available. We use that allowance
   narrowly: rather than discarding the real CGM trace, we add a documented,
   physiologically-motivated response layer on top of it (see step 5 below)
   that reproduces the meal/insulin/exercise dynamics a real sensor would
   show but this particular sample doesn't contain. Every row keeps its
   original reading in `glucose_measured`; the modelling column `glucose` is
   the augmented "digital twin" signal, and the gap between the two is
   reported, not hidden.

3. PURGED, TIME-BASED, PER-PATIENT SPLITS.
   Because a forecast looks FORECAST_HORIZON minutes into the future, a naive
   time split can let a training row's target peek at data technically
   inside the validation window. We purge a horizon-sized gap at each split
   boundary (per patient) so that never happens.

Run directly to regenerate every artifact under models/ and data/processed/:
    python src/pipeline.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "GlucoBench_benchmark_dataset-selected-columns.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

BASELINE_DAYS = 3                          # held out per patient, used only for the static profile
FORECAST_HORIZON = pd.Timedelta(minutes=60)  # how far ahead the Digital Twin forecasts glucose
FORECAST_TOLERANCE = pd.Timedelta(minutes=10)  # matching slack when locating the "future" reading
RISK_THRESHOLD = 140.0                     # mg/dL -- ADA post-prandial hyperglycemia target
RANDOM_SEED = 42

# --- Physiological augmentation constants (see docstring, point 2) ---------
CARB_GAIN = 1.0          # mg/dL per gram of carbohydrate, at average insulin sensitivity
INSULIN_GAIN = 8.0       # mg/dL lowered per unit of bolus insulin
EXERCISE_GAIN = 0.012    # mg/dL lowered per step taken in one exercise bout
MEAL_PEAK_MIN = 45       # minutes to peak post-meal effect
MEAL_WINDOW_MIN = 240    # minutes over which a meal's effect fully decays
EX_PEAK_MIN = 30
EX_WINDOW_MIN = 120
AUGMENTATION_NOISE_STD = 1.2  # mg/dL, small physiological/sensor noise added on top
GLUCOSE_PHYSIOLOGICAL_BOUNDS = (50.0, 400.0)

NUMERIC_FEATURES = [
    "glucose", "glucose_30min_mean", "glucose_30min_std", "glucose_30min_min", "glucose_30min_max",
    "glucose_2h_mean", "glucose_2h_std", "glucose_slope_30min",
    "carbs_60min_sum", "insulin_bolus_60min_sum", "insulin_basal", "exercise_steps_60min_sum",
    "minutes_since_meal", "hour_sin", "hour_cos", "cgm_quality_flag",
    "age", "bmi", "hba1c", "diabetes_duration_years", "previous_hyperglycemic_episodes",
    "diabetes_status_int",
]
CATEGORICAL_FEATURES = ["sex", "medication", "smoking_status"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "glucose_future_60min"

# Split for explainability: static features never change for a given patient,
# so "what's driving the forecast right now" only makes sense over the
# DYNAMIC subset, compared against that patient's own typical values -- not
# the population's. Static features get their own panel, compared against
# the population, to answer a different question ("why is this patient's
# baseline risk elevated at all").
STATIC_NUMERIC_FEATURES = [
    "age", "bmi", "hba1c", "diabetes_duration_years", "previous_hyperglycemic_episodes",
    "diabetes_status_int",
]
DYNAMIC_NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES if f not in STATIC_NUMERIC_FEATURES]


# ---------------------------------------------------------------------------
# 1. Load dynamic (wearable/CGM-style) data
# ---------------------------------------------------------------------------

def load_dynamic_data(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={"user_id": "patient_id"})
    return df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Synthesize a static EHR profile per patient from the held-out baseline
#    window, using clinically grounded relationships plus a small amount of
#    patient-seeded randomness (deterministic and reproducible).
# ---------------------------------------------------------------------------

def _seeded_rng(patient_id: str) -> np.random.Generator:
    seed = int(hashlib.sha256(patient_id.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed)


def build_static_profiles(dynamic_df: pd.DataFrame, baseline_days: int = BASELINE_DAYS) -> pd.DataFrame:
    rows = []
    for pid, g in dynamic_df.groupby("patient_id"):
        cutoff = g["timestamp"].min() + pd.Timedelta(days=baseline_days)
        base = g[g["timestamp"] < cutoff]

        mean_glucose = float(base["glucose"].mean())
        std_glucose = float(base["glucose"].std())
        frac_high = float((base["glucose"] >= RISK_THRESHOLD).mean())

        rng = _seeded_rng(pid)
        age = int(rng.integers(38, 76))
        sex = str(rng.choice(["F", "M"]))
        bmi = round(float(np.clip(rng.normal(24 + (mean_glucose - 90) * 0.09, 2.0), 19, 42)), 1)

        # ADAG study formula relates estimated average glucose (eAG, mg/dL) to
        # HbA1c: eAG = 28.7 * A1C - 46.7  =>  A1C = (eAG + 46.7) / 28.7.
        # The baseline-window mean glucose stands in for eAG.
        hba1c = float(np.clip((mean_glucose + 46.7) / 28.7 + rng.normal(0, 0.15), 4.8, 11.5))
        hba1c = round(hba1c, 1)

        diabetes_status = bool(hba1c >= 6.5 or mean_glucose >= 100)
        diabetes_duration_years = (
            int(np.clip(round((hba1c - 5.0) * 2.2 + rng.normal(0, 1.5)), 0, 22)) if diabetes_status else 0
        )

        if not diabetes_status:
            medication = "None"
        elif hba1c < 7.0:
            medication = "Metformin"
        elif hba1c < 8.5:
            medication = "Metformin + Sulfonylurea"
        else:
            medication = "Insulin + Metformin"

        previous_hyperglycemic_episodes = int(np.clip(round(frac_high * 40 + rng.normal(0, 1.5)), 0, 30))
        smoking_status = str(rng.choice(["Never", "Former", "Current"], p=[0.6, 0.3, 0.1]))

        # Insulin sensitivity multiplier for the physiological augmentation
        # layer: worse baseline control (higher HbA1c) -> a bigger, more
        # sluggish glucose excursion for the same meal. 5.0% HbA1c -> 0.59x;
        # 7.8% (roughly this cohort's median) -> ~1.0x; 11.5% -> ~1.67x.
        response_multiplier = float(np.clip(0.5 + 0.18 * (hba1c - 5.0), 0.5, 2.2))

        rows.append(dict(
            patient_id=pid, age=age, sex=sex, bmi=bmi, hba1c=hba1c,
            diabetes_status=diabetes_status, diabetes_duration_years=diabetes_duration_years,
            medication=medication, previous_hyperglycemic_episodes=previous_hyperglycemic_episodes,
            smoking_status=smoking_status, response_multiplier=response_multiplier,
            baseline_mean_glucose=round(mean_glucose, 1), baseline_glucose_std=round(std_glucose, 1),
        ))
    return pd.DataFrame(rows)


def monitoring_window(dynamic_df: pd.DataFrame, baseline_days: int = BASELINE_DAYS) -> pd.DataFrame:
    """The post-baseline slice of each patient's timeline -- this is the only
    part the supervised task ever trains or evaluates on."""
    parts = []
    for _, g in dynamic_df.groupby("patient_id"):
        cutoff = g["timestamp"].min() + pd.Timedelta(days=baseline_days)
        parts.append(g[g["timestamp"] >= cutoff])
    return pd.concat(parts).sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Data fusion
# ---------------------------------------------------------------------------

def fuse(dynamic_df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    fused = dynamic_df.merge(static_df, on="patient_id", how="left")
    fused["diabetes_status_int"] = fused["diabetes_status"].astype(int)
    return fused


# ---------------------------------------------------------------------------
# 4. Physiological augmentation layer (see module docstring, point 2).
#    Adds a causal, superposed impulse-response effect for every meal and
#    exercise event: a gamma-shaped kernel peaking a fixed number of minutes
#    later and fully decaying within a bounded window, scaled by the
#    patient's own insulin-sensitivity multiplier. This is what turns a flat
#    trace into one where the Digital Twin's fused features (carbs, bolus,
#    exercise, patient risk profile) actually explain future glucose.
# ---------------------------------------------------------------------------

def _response_kernel(tau_minutes: np.ndarray, peak_minutes: float) -> np.ndarray:
    """Unit-peak gamma-like kernel: 0 at tau=0, peaks at tau=peak_minutes,
    decays back toward 0 afterward. Only defined for tau >= 0 (causal)."""
    x = tau_minutes / peak_minutes
    return x * np.exp(1 - x)


def _simulate_one_patient(g: pd.DataFrame, response_multiplier: float,
                           rng: np.random.Generator) -> np.ndarray:
    t = g["timestamp"].values
    minutes = (t - t[0]) / np.timedelta64(1, "m")
    n = len(g)
    delta = np.zeros(n)

    carbs = g["carbs"].values
    bolus = g["insulin_bolus"].values
    steps = g["exercise_steps"].values

    for j in np.where(carbs > 0)[0]:
        tau = minutes - minutes[j]
        mask = (tau >= 0) & (tau <= MEAL_WINDOW_MIN)
        if not mask.any():
            continue
        net_effect = carbs[j] * CARB_GAIN * response_multiplier - bolus[j] * INSULIN_GAIN
        delta[mask] += _response_kernel(tau[mask], MEAL_PEAK_MIN) * net_effect

    for j in np.where(steps > 0)[0]:
        tau = minutes - minutes[j]
        mask = (tau >= 0) & (tau <= EX_WINDOW_MIN)
        if not mask.any():
            continue
        delta[mask] -= _response_kernel(tau[mask], EX_PEAK_MIN) * steps[j] * EXERCISE_GAIN

    delta += rng.normal(0, AUGMENTATION_NOISE_STD, size=n)
    return delta


def simulate_physiological_response(fused_df: pd.DataFrame) -> pd.DataFrame:
    df = fused_df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    df["glucose_measured"] = df["glucose"]

    parts = []
    for pid, g in df.groupby("patient_id"):
        g = g.copy()
        rng = _seeded_rng(pid + "_augmentation")
        multiplier = float(g["response_multiplier"].iloc[0])
        delta = _simulate_one_patient(g, multiplier, rng)
        g["glucose"] = np.clip(g["glucose_measured"].values + delta, *GLUCOSE_PHYSIOLOGICAL_BOUNDS)
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Feature engineering -- every feature here only looks backward in time,
#    so it is safe to compute the same way at inference time in the
#    dashboard.
# ---------------------------------------------------------------------------

def _engineer_one_patient(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").set_index("timestamp")

    g["glucose_30min_mean"] = g["glucose"].rolling("30min").mean()
    g["glucose_30min_std"] = g["glucose"].rolling("30min").std().fillna(0.0)
    g["glucose_30min_min"] = g["glucose"].rolling("30min").min()
    g["glucose_30min_max"] = g["glucose"].rolling("30min").max()
    g["glucose_2h_mean"] = g["glucose"].rolling("120min").mean()
    g["glucose_2h_std"] = g["glucose"].rolling("120min").std().fillna(0.0)

    g["carbs_60min_sum"] = g["carbs"].rolling("60min").sum()
    g["insulin_bolus_60min_sum"] = g["insulin_bolus"].rolling("60min").sum()
    g["exercise_steps_60min_sum"] = g["exercise_steps"].rolling("60min").sum()

    g["glucose_slope_30min"] = g["glucose"] - g["glucose_30min_mean"]

    last_meal_time = g.index.to_series().where(g["carbs"] > 0).ffill()
    minutes_since_meal = (g.index.to_series() - last_meal_time).dt.total_seconds() / 60
    g["minutes_since_meal"] = minutes_since_meal.fillna(999.0).clip(upper=999.0)

    hour = g.index.hour + g.index.minute / 60
    g["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    g["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return g.reset_index()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    parts = [_engineer_one_patient(g) for _, g in df.groupby("patient_id")]
    return pd.concat(parts, ignore_index=True).sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Forecast target: the glucose reading nearest to timestamp + horizon
#    (matched with a small tolerance to tolerate irregular CGM sampling).
#    Also keep two zero-effort baselines for honest comparison:
#      - persistence: forecast = current glucose
#      - linear trend: forecast = current + (current - 30min-ago trend)
# ---------------------------------------------------------------------------

def build_forecast_target(df: pd.DataFrame, horizon: pd.Timedelta = FORECAST_HORIZON,
                           tolerance: pd.Timedelta = FORECAST_TOLERANCE) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby("patient_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        target_times = pd.DataFrame({"timestamp": g["timestamp"] + horizon}).sort_values("timestamp")
        future = pd.merge_asof(
            target_times,
            g[["timestamp", "glucose"]].rename(columns={"glucose": TARGET_COL}),
            on="timestamp", direction="nearest", tolerance=tolerance,
        )
        g[TARGET_COL] = future[TARGET_COL].values
        parts.append(g)
    out = pd.concat(parts, ignore_index=True).sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    out["persistence_forecast"] = out["glucose"]
    out["trend_forecast"] = out["glucose"] + out["glucose_slope_30min"]
    out["current_hyperglycemia_status"] = (out["glucose"] >= RISK_THRESHOLD).astype(int)
    return out


# ---------------------------------------------------------------------------
# 7. Purged, per-patient, time-based split
# ---------------------------------------------------------------------------

def time_based_split(df: pd.DataFrame, horizon: pd.Timedelta = FORECAST_HORIZON,
                      train_frac: float = 0.70, val_frac: float = 0.15) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for _, g in df.groupby("patient_id"):
        g = g.sort_values("timestamp")
        start, end = g["timestamp"].min(), g["timestamp"].max()
        total = end - start
        train_cut = start + total * train_frac
        val_cut = start + total * (train_frac + val_frac)

        parts["train"].append(g[g["timestamp"] + horizon <= train_cut])
        parts["val"].append(g[(g["timestamp"] > train_cut) & (g["timestamp"] + horizon <= val_cut)])
        parts["test"].append(g[g["timestamp"] > val_cut])

    splits = {k: pd.concat(v).reset_index(drop=True) for k, v in parts.items()}
    return {k: v.dropna(subset=[TARGET_COL]).reset_index(drop=True) for k, v in splits.items()}


# ---------------------------------------------------------------------------
# 8. Model training -- forecasting glucose FORECAST_HORIZON minutes ahead
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def build_models() -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([
            ("prep", build_preprocessor()),
            ("reg", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessor()),
            ("reg", RandomForestRegressor(
                n_estimators=400, max_depth=10, min_samples_leaf=8,
                random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]),
    }


def evaluate_regression(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def baseline_metrics(split_df: pd.DataFrame) -> dict:
    """The two zero-training baselines every real model must beat to justify
    the extra complexity: flat persistence, and a naive linear extrapolation
    of the current 30-minute trend."""
    y_true = split_df[TARGET_COL]
    return {
        "persistence": evaluate_regression(y_true, split_df["persistence_forecast"]),
        "trend_extrapolation": evaluate_regression(y_true, split_df["trend_forecast"]),
    }


def meal_or_exercise_subset_metrics(split_df: pd.DataFrame, forecast: np.ndarray) -> dict:
    """Where fused behavioural features should matter most: rows with a meal
    or exercise bout in the last hour. Reported separately because a global
    average can hide (or fake) where the model actually earns its keep."""
    mask = (split_df["carbs_60min_sum"] > 0) | (split_df["exercise_steps_60min_sum"] > 0)
    if mask.sum() < 5:
        return {"n": int(mask.sum())}
    y_true = split_df.loc[mask, TARGET_COL]
    return {
        "n": int(mask.sum()),
        "model": evaluate_regression(y_true, forecast[mask.values]),
        "persistence": evaluate_regression(y_true, split_df.loc[mask, "persistence_forecast"]),
    }


# ---------------------------------------------------------------------------
# 9. Deriving a hyperglycemia-risk PROBABILITY from a point forecast.
#    Assuming approximately normal forecast residuals (estimated on the
#    validation set), P(true future glucose >= threshold) is the upper tail
#    of a normal distribution centered on the forecast.
# ---------------------------------------------------------------------------

def residual_std(y_true, y_pred) -> float:
    return float(np.std(np.asarray(y_true) - np.asarray(y_pred), ddof=1))


def forecast_to_risk_probability(forecast, resid_std: float, threshold: float = RISK_THRESHOLD):
    resid_std = max(resid_std, 1e-6)
    z = (threshold - np.asarray(forecast)) / resid_std
    return 1.0 - norm.cdf(z)


# ---------------------------------------------------------------------------
# 10. Explainability without extra dependencies: per-prediction "what would
#     change this forecast" via single-feature occlusion to the population
#     median. (If `shap` is installed, the notebook adds a TreeExplainer view
#     on top of this -- but the dashboard never requires it.)
# ---------------------------------------------------------------------------

def _occlusion_contributions(pipeline: Pipeline, row: pd.Series, reference_medians: pd.Series,
                              features: list[str], top_k: int) -> pd.DataFrame:
    base = pipeline.predict(row.to_frame().T[ALL_FEATURES])[0]
    deltas = []
    for feat in features:
        perturbed = row.copy()
        perturbed[feat] = reference_medians[feat]
        pred = pipeline.predict(perturbed.to_frame().T[ALL_FEATURES])[0]
        deltas.append((feat, base - pred))
    out = pd.DataFrame(deltas, columns=["feature", "forecast_contribution_mgdl"])
    out["abs_contribution"] = out["forecast_contribution_mgdl"].abs()
    return out.sort_values("abs_contribution", ascending=False).head(top_k).drop(columns="abs_contribution")


def explain_dynamic_drivers(pipeline: Pipeline, row: pd.Series, patient_history: pd.DataFrame,
                             top_k: int = 5) -> pd.DataFrame:
    """What's different about THIS moment compared to this same patient's own
    typical moment -- e.g. "carbs_60min_sum is 60g higher than usual for this
    patient, which is pushing the forecast up by 12 mg/dL". Comparing against
    the population here would just re-detect that this patient has a
    different baseline than everyone else, which the static panel already
    covers."""
    return _occlusion_contributions(pipeline, row, patient_history[DYNAMIC_NUMERIC_FEATURES].median(),
                                     DYNAMIC_NUMERIC_FEATURES, top_k)


def explain_static_risk_factors(pipeline: Pipeline, row: pd.Series, population_reference: pd.DataFrame,
                                 top_k: int = 5) -> pd.DataFrame:
    """Why this patient's baseline forecast sits where it does relative to
    the rest of the monitored cohort -- e.g. "this patient's HbA1c is high
    enough that, on its own, it raises their forecast by 40 mg/dL relative to
    a population-typical patient"."""
    return _occlusion_contributions(pipeline, row, population_reference[STATIC_NUMERIC_FEATURES].median(),
                                     STATIC_NUMERIC_FEATURES, top_k)


# ---------------------------------------------------------------------------
# 11. End-to-end pipeline
# ---------------------------------------------------------------------------

def run() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dynamic_df = load_dynamic_data()
    static_df = build_static_profiles(dynamic_df)
    monitoring_df = monitoring_window(dynamic_df)

    fused = fuse(monitoring_df, static_df)
    augmented = simulate_physiological_response(fused)
    featured = engineer_features(augmented)
    targeted = build_forecast_target(featured)

    splits = time_based_split(targeted)

    metrics = {split: baseline_metrics(df) for split, df in splits.items()}
    metrics = {"baselines": metrics}

    models = build_models()
    fitted, forecasts = {}, {}
    for name, pipe in models.items():
        pipe.fit(splits["train"][ALL_FEATURES], splits["train"][TARGET_COL])
        fitted[name] = pipe
        metrics[name] = {}
        forecasts[name] = {}
        for split_name in ("train", "val", "test"):
            d = splits[split_name]
            pred = pipe.predict(d[ALL_FEATURES])
            forecasts[name][split_name] = pred
            metrics[name][split_name] = evaluate_regression(d[TARGET_COL], pred)
        metrics[name]["meal_or_exercise_subset_test"] = meal_or_exercise_subset_metrics(
            splits["test"], forecasts[name]["test"]
        )

    best_name = min(("ridge", "random_forest"), key=lambda n: metrics[n]["val"]["mae"])
    best_pipeline = fitted[best_name]
    metrics["selected_model"] = best_name

    val_resid_std = residual_std(splits["val"][TARGET_COL], forecasts[best_name]["val"])
    metrics["residual_std_val"] = val_resid_std

    tagged_parts = []
    for split_name, df in splits.items():
        d = df.copy()
        d["split"] = split_name
        forecast = best_pipeline.predict(d[ALL_FEATURES])
        d["predicted_glucose_60min"] = forecast
        d["predicted_hyperglycemia_risk"] = forecast_to_risk_probability(forecast, val_resid_std)
        tagged_parts.append(d)
    full_tagged = pd.concat(tagged_parts, ignore_index=True).sort_values(["patient_id", "timestamp"])
    full_tagged.to_csv(PROCESSED_DIR / "patient_twin_dataset.csv", index=False)
    static_df.to_csv(PROCESSED_DIR / "static_patient_profiles.csv", index=False)

    for name, pipe in fitted.items():
        joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")
    joblib.dump(best_pipeline, MODELS_DIR / "deployed_model.joblib")

    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump({
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET_COL,
            "risk_threshold_mgdl": RISK_THRESHOLD,
            "forecast_horizon_minutes": int(FORECAST_HORIZON.total_seconds() // 60),
            "baseline_days": BASELINE_DAYS,
            "selected_model": best_name,
            "residual_std_val": val_resid_std,
        }, f, indent=2)

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    m = run()
    print(json.dumps(m, indent=2))
