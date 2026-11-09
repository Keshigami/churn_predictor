# Telecom churn predictor · carousel

Spec: film/specs/portfolio__churn_predictor.json · 8 slides, 1080x1350.

## 1/8 Hook

- Three months, fiber, monthly: 92% churn risk.

## 2/8 The question

- 338 of 1,000 customers churn.
- The team cannot call everyone. Who first?
- Churn 338

## 3/8 What it does

- Telecom churn predictor
- Portfolio · ML Engineer
- Two models, one champion, every number measured
- max@baluyot ~ $ curl -s :8115/predict -d '{"Tenure":3,"MonthlyCharges":104,"TotalCharges":312}'  # model_monitoring_api, same churn.joblib

## 4/8 How it is built

- 1. data/telecom_churn.csv, 1,000 synthetic rows fro — pandas
- 2. LogisticRegression / RandomForestClassif — scikit-learn
- 3. scikit-learn: ColumnTransformer — scikit-learn · pandas · joblib
- 4. Streamlit (shared dark theme from common — Streamlit
- 5. Dashboard — Streamlit
- Champion by ROC-AUC, not by assumption

## 5/8 The result

- 2 models on the same holdout
- Random forest 0.787
- Logistic regression 0.806 (winner)

## 6/8 The proof

- Two models, same 200 held-out rows
- screen: portfolio/churn_predictor/demo/poster.png

## 7/8 The answer and the limit

- 0.806
- ROC-AUC · logistic regression · 200 held out
- Synthetic data: no real customers scored yet.
- And it ships.

## 8/8 Try it

- Try it: Use sample customer, Predict.
- max@baluyot ~ $ git clone https://github.com/Keshigami/churn_predictor.git
- max@baluyot ~ $ just churn_predictor-setup
- max@baluyot ~ $ just churn_predictor-up
- github.com/Keshigami/churn_predictor
- Maximus Friedrich Baluyot
- AI & ML Specialist · Applied AI · ML Engineering · Automation

_Maximus Friedrich Baluyot · Caltech CTME · PGC AIML_
