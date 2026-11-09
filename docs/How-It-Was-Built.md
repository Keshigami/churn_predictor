# How it was built

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

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | One venv per app, `uv sync` reproducible |
| Model or core logic | scikit-learn: ColumnTransformer + LogisticRegression / RandomForestClassifier | Interpretable baseline vs non-linear champion, both cheap to train |
| Interface | Streamlit (shared dark theme from `common/streamlit_theme.py`) | Dashboard with a form in a few lines; `AppTest` makes it testable |
| Data | `data/telecom_churn.csv`, 1,000 synthetic rows from `data_gen.py` (seed 42) | Same generator as the original portfolio |
| Tests | pytest + `streamlit.testing.v1.AppTest` (6 tests) | Training contract, scoring, unknown categories, UI renders and predicts |

## Libraries
`scikit-learn` (pipelines, both models, metrics), `pandas` (data frames), `joblib` (model bundle), `streamlit` (UI and
AppTest), `numpy` (data generation), `ruff` (lint), `pytest`.
