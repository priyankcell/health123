# TwinCare-Glyco

A Digital Twin proof-of-concept for near-term hyperglycemia risk monitoring
in Type 2 Diabetes, fusing a synthetic-but-clinically-grounded static EHR
profile with dynamic wearable/CGM-style data.

```
Static EHR profile  ---+
                        |--> Data fusion --> Feature engineering --> Forecast model --> Risk score --> Dashboard
Dynamic CGM/wearable ---+
```

## Why this exists

A first pass at this project used dummy placeholder data and an arbitrary
"glucose > 180 in the next hour" spike label. Before building on that, the
real `GlucoBench_benchmark_dataset-selected-columns.csv` sample was audited
directly: glucose in it moves almost nowhere on a clinically relevant
timescale (largest 60-minute change in the whole file: under 12 mg/dL), so a
naive spike-prediction task collapses into "is it already high right now" --
a trivial, non-predictive model. That finding, and the redesign it led to
(a documented physiological augmentation layer, a regression forecast target
instead of a binary spike label, honest persistence-baseline comparisons
throughout, and purged time-based splits), is walked through in full in
`notebooks/TwinCare_Glyco.ipynb`, sections 2 and 4.

## Project structure

```
Healthcare Project/
├── data/
│   ├── raw/                          # original GlucoBench CSV
│   └── processed/                    # fused + engineered dataset, static profiles
├── notebooks/
│   ├── TwinCare_Glyco.ipynb          # full walkthrough: audit -> fusion -> model -> explainability
│   └── healthcase_original_draft.ipynb  # archived first draft (dummy data, kept for reference)
├── src/
│   └── pipeline.py                   # single source of truth: data fusion, features, model, artifacts
├── dashboard/
│   └── app.py                        # Streamlit doctor-facing Digital Twin dashboard
├── models/                           # generated: trained model + metrics (run pipeline.py first)
├── requirements.txt
└── README.md
```

`src/pipeline.py` is the only place the modeling logic lives. The notebook
narrates and visualizes it; the dashboard only ever loads its saved
artifacts. Neither duplicates the logic, so they can't drift apart.

## How to run

```bash
pip install -r requirements.txt

# 1. Generate all artifacts (data/processed/, models/)
python src/pipeline.py

# 2. Launch the dashboard
streamlit run dashboard/app.py

# (optional) re-run the full narrated notebook end to end
jupyter nbconvert --to notebook --execute notebooks/TwinCare_Glyco.ipynb
```

## What it predicts

- **Target:** patient glucose 60 minutes ahead (regression), derived into a
  hyperglycemia-risk probability (P(glucose ≥ 140 mg/dL) assuming normal
  forecast residuals estimated on a held-out validation set).
- **Models compared:** persistence baseline, linear-trend baseline, Ridge
  regression, Random Forest -- selected by validation MAE.
- **Where the fused model actually adds value:** on the full test set,
  persistence remains a strong baseline (most timestamps have nothing
  dynamic happening). Restricted to the clinically interesting rows --
  those following a meal or exercise event in the last hour -- the fused
  model beats persistence (see notebook section 7). That distinction is
  reported explicitly rather than hidden behind one blended metric.

## Known limitations

- The dynamic excursions are partly synthetic: a documented,
  physiologically-motivated meal/insulin/exercise response layer was added on
  top of the real (very low-volatility) CGM trace, because the brief permits
  synthetic time-series generation when a suitable dynamic dataset isn't
  available. The original sensor reading is preserved in every saved file as
  `glucose_measured`, next to the augmented `glucose` used for modeling.
- Only 10 patients, ~6 usable monitoring days each after holding out a
  baseline window -- small enough that individual patients can dominate
  aggregate metrics.
- The static EHR profile is synthesized from each patient's own baseline
  glucose statistics via clinically-grounded formulas (the ADAG HbA1c/eAG
  relationship, stepped diabetes therapy), not sourced from a real EHR.
- Risk probabilities assume normal, homoscedastic forecast residuals; a real
  deployment should validate calibration rather than trust that assumption.
- This is a decision-support proof-of-concept, not a diagnostic or
  deployment-ready clinical system.
