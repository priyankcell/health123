# TwinCare-Glyco (Digital Twin PoC)

## 1. Team
Competition team repository for a proof-of-concept Digital Twin prototype.

## 2. Problem Statement
Build a virtual representation of a patient that combines:
- **historical baseline** (EHR-like profile), and
- **dynamic current state** (wearable-style stream),

to predict one specific adverse event early enough to be useful.

## 3. Healthcare Use Case
- **Condition:** Type 2 Diabetes
- **Adverse event:** Near-term glucose spike
- **Forecast horizon:** Next 2 hours

## 4. Why a Digital Twin?
The twin updates patient risk as new data arrives (not a one-time static score).

## 5. Data Sources (POC)
- Synthetic baseline profile (demographics, comorbidity, HbA1c, activity baseline)
- Synthetic dynamic signals (glucose, HR, HRV, sleep, steps trend)

> This repository is designed for synthetic/prototype use only (no real patient data).

## 6. Data Fusion and Features
The model fuses static + dynamic features:
- age, diabetes, hypertension, baseline HbA1c
- current glucose, glucose slope, glucose variability
- heart rate, HRV, sleep hours, recent steps

## 7. Prediction Target
Binary label:
- `1`: glucose spike expected within next 2 hours
- `0`: otherwise

## 8. Model
This PoC uses a **Logistic Regression** baseline trained on synthetic fused examples and outputs:
- `P(glucose_spike_next_2h)`

Risk bands (prototype design):
- 0–30%: Low
- 30–60%: Moderate
- 60–80%: High
- 80%+: Very High

## 9. Dashboard (Streamlit)
Single-page clinician-style view includes:
- Patient baseline summary
- Current dynamic metrics
- Current risk probability + band
- Risk trend over time (dynamic twin behavior)
- "What-if" simulation (`+30 min activity`, `+1h sleep`)

## 10. Repository Structure
```text
.
├── README.md
├── requirements.txt
├── dashboard/
│   └── app.py
├── src/
│   └── twincare_glyco/
│       ├── __init__.py
│       ├── features.py
│       └── model.py
└── tests/
    └── test_model.py
```

## 11. Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 12. Run Dashboard
```bash
streamlit run /home/runner/work/health123/health123/dashboard/app.py
```

## 13. Run Tests
```bash
python -m unittest discover -s /home/runner/work/health123/health123/tests -v
```

## 14. Ethics / Privacy
- Use synthetic/anonymized data only.
- This is a proof-of-concept and **not** a clinical decision system.

## 15. License
Add your preferred open-source license before final competition submission.