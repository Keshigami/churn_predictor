---
title: Telecom Churn Predictor
group: portfolio
role: ML Engineer
port: 8111
status: working
one_liner: Scores a telecom customer's churn risk with a Logistic Regression and a Random Forest trained once on synthetic data, in a Streamlit dashboard.
---
# Telecom Churn Predictor

## Demo
<!-- demo:auto -->
![demo](demo/demo.gif)

Full walkthrough with captions: [demo/demo.mp4](demo/demo.mp4) (60 s, 1280x800, animated intro and end card). Run it yourself: `just setup && just up`, then open http://localhost:8111/.

Two-minute walkthrough:
1. Telecom churn: logistic regression versus random forest on the held-out 20 percent
2. Random forest feature importances and the logistic coefficients
3. Score one customer: use the sample customer (short tenure, fiber, month-to-month)
4. Predict: high risk of churn
5. Now the safe customer: five years in, two-year contract, DSL
6. Predict: low risk
<!-- /demo:auto -->

## Brief
Telecom operators lose revenue when customers switch providers, and the retention team cannot call everyone. This app
predicts which customers are likely to churn so outreach goes to the riskiest first. "Solved" means a scored list with a
model whose numbers are visible and reproducible, and a form to score one customer on demand.

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | One venv per app, `uv sync` reproducible |
| Model or core logic | scikit-learn: ColumnTransformer + LogisticRegression / RandomForestClassifier | Interpretable baseline vs non-linear champion, both cheap to train |
| Interface | Streamlit (shared dark theme from `common/streamlit_theme.py`) | Dashboard with a form in a few lines; `AppTest` makes it testable |
| Data | `data/telecom_churn.csv`, 1,000 synthetic rows from `data_gen.py` (seed 42) | Same generator as the original portfolio |
| Tests | pytest + `streamlit.testing.v1.AppTest` (6 tests) | Training contract, scoring, unknown categories, UI renders and predicts |

## Libraries used
`scikit-learn` (pipelines, both models, metrics), `pandas` (data frames), `joblib` (model bundle), `streamlit` (UI and
AppTest), `numpy` (data generation), `ruff` (lint), `pytest`.

## How it was built
1. Read the original `app.py`: it trained both models on every page load inside Streamlit, label-encoded categoricals
   with a single `LabelEncoder` reused across columns (so the encodings were wrong), and the prediction form sent a
   hand-built array with zeros for every categorical. The data generator and CSV were kept as they were.
2. Moved training into `train.py`: proper `ColumnTransformer` with fixed category lists (`features.py`), stratified
   80/20 split, accuracy / F1 / ROC-AUC for both models, Random Forest importances and LR coefficients, training means
   and standard deviations saved for the monitoring API's drift check. Champion chosen by ROC-AUC.
3. Rebuilt the dashboard to load `models/churn.joblib` and `models/metrics.json`: comparison table, two charts, a form
   with every feature, "Use sample customer" and "Use safe customer" buttons, stable widget keys (`input`, `run`).
4. Tests: training writes a bundle with sane metrics; the bundle scores a customer; an unknown category does not crash;
   AppTest renders the page, fills the form and gets a risk card.

## Results
`models/metrics.json`, this machine, full data (training takes 3 s so there is no quick mode):

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression (champion) | 0.740 | 0.581 | 0.806 |
| Random Forest | 0.730 | 0.542 | 0.787 |

ROC-AUC 0.81 means the model ranks a random churner above a random non-churner 81% of the time. The original brief
wanted accuracy > 0.80 and F1 > 0.70; the synthetic churn rule is noisy by construction (probability drawn from four
indicators), so that target is not reachable on this data and the README says so instead of quoting it.

## Demo details
- Run: `just churn_predictor-setup && just churn_predictor-up`, open `http://127.0.0.1:8111/`.
- Two-minute walkthrough: (1) read the comparison table and say why LR is champion here; (2) look at the importance
  chart (tenure, charges, contract); (3) press "Use sample customer", then "Predict" for a red high-risk card;
  (4) press "Use safe customer", "Predict", green card; flip the contract to Month-to-month and watch it move.
- Video clip: `demo/clips/portfolio__churn_predictor.webm` (recorded by `just demo-video`).
- Sample inputs: the two buttons. Sample customer (3 months, $104, fiber, monthly) scores about 90% churn; safe
  customer (60 months, $30, DSL, two-year) scores under 10%.

## Limits and next steps
- The data is synthetic; nothing here says how the model would do on a real telecom book.
- No calibration or threshold tuning: 0.5 is used as the cut, a retention team would pick a cut by call capacity.
- Feature set is the generator's 11 columns; real data would add usage, support tickets and payment history.
- Next: XGBoost with the same pipeline, SHAP for per-customer reasons, and this model is already served by
  `model_monitoring_api` (port 8115), which is the "deploy as an API with monitoring" step from the original plan.
