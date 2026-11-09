<!-- public-top -->
# Telecom Churn Predictor

Scores a telecom customer's churn risk with a Logistic Regression and a Random Forest trained once on synthetic data, in a Streamlit dashboard. Portfolio, ML Engineer.

**Watch it first:** [the 90-second film](demo/film.mp4) (what it does, how it is built, the proof, the answer) or the [60-second cut](demo/film-60s.mp4). The plain walkthrough with captions is [demo/demo.mp4](demo/demo.mp4).

## Try it in three commands

```bash
git clone https://github.com/Keshigami/churn_predictor.git && cd churn_predictor
just setup        # creates the venv with uv (Python 3.11)
just up           # http://127.0.0.1:8111/  (just down to stop)
```

You need [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just) (`brew install uv just` on a Mac). Everything the app needs is in the repo (synthetic sample data where a dataset is used). No API key is needed anywhere in this project. `just test` runs the lint and the offline tests.

Step by step: [docs/Getting-Started.md](docs/Getting-Started.md), then [how it was built](docs/How-It-Was-Built.md), [results](docs/Results.md), [limits](docs/Limits-and-Next-Steps.md) and [the film](docs/The-Film.md). The same pages are on the [wiki](https://github.com/Keshigami/churn_predictor/wiki).
<!-- /public-top -->
# Telecom Churn Predictor

## Demo walkthrough
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

Binary classification for customer retention: a Logistic Regression baseline against a Random Forest, trained once,
served in a Streamlit dashboard with a form to score a single customer. Port **8111**.

## 1. Problem statement and business impact
- **The problem**: telecom companies lose revenue when customers switch providers. Finding likely churners early lets
  a retention team call the right people first.
- **Goal**: predict churn with usable recall so outreach is prioritised automatically instead of by hand.
- **Metric of success (original brief)**: accuracy > 80%, F1 > 0.70. See the honest numbers below: the synthetic data
  does not get there, and the README says so.

## 2. Technical solution
- **Approach**: one preprocessing pipeline (one-hot categoricals with fixed category lists, passthrough numerics) feeding
  two models. `train.py` fits both on an 80/20 stratified split, writes `models/churn.joblib` (both pipelines plus the
  training feature statistics used by the monitoring API's drift check) and `models/metrics.json`.
- **Stack**: Python 3.11 via uv, scikit-learn 1.9, pandas, joblib, Streamlit (tests with `streamlit.testing.v1.AppTest`).
- **Flow**: `[telecom_churn.csv] -> [ColumnTransformer] -> {LogisticRegression | RandomForest} -> probability -> dashboard`

## 3. Results on this machine (`models/metrics.json`, 1,000 synthetic rows, 200 held out)
| Model | Accuracy | F1 | ROC-AUC | Train | One prediction |
|---|---|---|---|---|---|
| Logistic Regression (champion) | 0.740 | 0.581 | 0.806 | 0.02 s | 1.7 ms |
| Random Forest | 0.730 | 0.542 | 0.787 | 0.15 s | 6.1 ms |

Top Random Forest importances: TotalCharges 0.155, Tenure 0.151, MonthlyCharges 0.142, Contract=Month-to-month 0.128,
InternetService=Fiber optic 0.045. The champion is picked by ROC-AUC, and on this data the linear model wins: the
synthetic churn rule is a weighted sum of four indicators, which is exactly what a logistic regression fits. The original
write-up assumed the Random Forest would be the champion and quoted "~85% accuracy"; the measured numbers replace that.

## 4. How to run
```bash
just churn_predictor-setup      # uv sync, then train if models/ is empty
just churn_predictor-train      # re-train: writes models/churn.joblib and models/metrics.json
just churn_predictor-test       # ruff + pytest (6 tests: training, scoring, unknown categories, AppTest UI)
just churn_predictor-up         # Streamlit on http://127.0.0.1:8111/ (pid in .pid, log in app.log)
just churn_predictor-down
```
Or directly: `uv run python train.py && uv run streamlit run app.py`.

## 5. Two-minute interviewer walkthrough
1. Open http://127.0.0.1:8111/. The comparison table at the top is read from `models/metrics.json`; point out that the
   champion badge says `logistic_regression` and why (section 3).
2. Left chart: Random Forest importances (tenure and charges first). Right chart: standardised LR coefficients, where
   month-to-month and fiber push toward churn.
3. Press **Use sample customer** (3 months tenure, $104/month, fiber, monthly contract) then **Predict**: a red
   "High risk" card with the probability and the other model's opinion next to it.
4. Press **Use safe customer** then **Predict**: green "Low risk". Change the contract to Month-to-month and watch
   the probability move.

## 6. What is real and what is simulated
- Real: the training, the two fitted pipelines, every number on screen, the one-customer scoring.
- Simulated: the dataset. `data_gen.py` is the original generator (seed 42, so `data/telecom_churn.csv` is byte-identical
  to the portfolio copy). Churn is drawn from a hand-written probability, so the ceiling is low and there are no seasonal
  or usage signals.
- Not here: the "latency < 50 ms" claim is now measured (1.7 ms / 6.1 ms per single prediction).

## 7. Files
`train.py` (training), `features.py` (schema shared with the monitoring API), `app.py` (dashboard), `data_gen.py`,
`tests/` (pytest + AppTest), `models/` (artifacts, git-ignored joblib), `.streamlit/config.toml` (theme).

## Standalone repo

Exported from Max's `caltech-capstone` workspace. `common/` is vendored (LLM door, UI tokens). Run: `just setup && just test && just up`.
LLM-backed features read `ANTHROPIC_API_KEY` (or `LLM_PROVIDER` + `OPENAI_API_KEY` / `GOOGLE_API_KEY`); with no key they run in a deterministic offline mode.
Film: `demo/film.mp4` (90 s, what it does, how it is built, the proof, the answer) and `demo/film-60s.mp4` (60 s cut); `demo/demo.mp4` is the plain walkthrough.
