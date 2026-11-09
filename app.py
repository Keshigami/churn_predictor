"""Streamlit dashboard: LR vs RF comparison from models/metrics.json and a form to score one customer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import CATEGORICAL, CHOICES, EXAMPLE_CUSTOMER, customer_frame

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
sys.path.insert(0, str(HERE.parents[1]))
from common.streamlit_theme import apply  # noqa: E402

SAMPLE = {**EXAMPLE_CUSTOMER, "Tenure": 3, "MonthlyCharges": 104.0, "TotalCharges": 312.0}
SAFE = {
    "Tenure": 60, "MonthlyCharges": 30.0, "TotalCharges": 1800.0, "SeniorCitizen": 0,
    "InternetService": "DSL", "Contract": "Two year", "PaymentMethod": "Bank transfer",
    "PaperlessBilling": "No", "Gender": "Male", "Partner": "Yes", "Dependents": "Yes",
}  # fmt: skip

st.set_page_config(page_title="Churn Predictor", layout="wide")


@st.cache_resource
def load_bundle():
    bundle = joblib.load(MODELS / "churn.joblib")
    metrics = json.loads((MODELS / "metrics.json").read_text())
    return bundle, metrics


def set_customer(c: dict) -> None:
    st.session_state["input"] = int(c["Tenure"])
    st.session_state["monthly"] = float(c["MonthlyCharges"])
    st.session_state["total"] = float(c["TotalCharges"])
    st.session_state["senior"] = bool(c["SeniorCitizen"])
    for col in CATEGORICAL:
        st.session_state[f"cat_{col}"] = c[col]


if not (MODELS / "churn.joblib").exists():
    st.error("No trained model. Run `uv run python train.py` (or `just churn_predictor-train`) first.")
    st.stop()

bundle, metrics = load_bundle()
champion = bundle["champion"]
apply(
    "Telecom customer churn predictor",
    "Logistic Regression baseline against a Random Forest, trained once by train.py on 1,000 synthetic "
    "customers. Flags high-risk customers so retention outreach is prioritised automatically.",
    badge=f"champion: {champion} · port 8111",
)

st.subheader("Model comparison (held-out 20%)")
rows = pd.DataFrame(metrics["models"]).T
rows.index = [f"{i} (champion)" if i == champion else i for i in rows.index]
st.dataframe(rows[["accuracy", "f1", "roc_auc", "train_seconds", "single_prediction_ms"]], width="stretch")
m1, m2, m3 = st.columns(3)
m1.metric("Champion ROC-AUC", f"{metrics['models'][champion]['roc_auc']:.3f}")
m2.metric("Champion F1", f"{metrics['models'][champion]['f1']:.3f}")
m3.metric("Training rows / churn rate", f"{metrics['n_train']} / {metrics['churn_rate']:.0%}")

left, right = st.columns(2)
with left:
    st.subheader("Random Forest feature importances")
    fi = pd.Series(metrics["feature_importances"]).head(12).sort_values()
    st.bar_chart(fi, horizontal=True)
    st.caption("Share of impurity reduction. Tenure and charges dominate; contract is the top category.")
with right:
    st.subheader("Logistic Regression coefficients")
    coef = pd.Series(metrics["lr_coefficients"])
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index).head(12).sort_values()
    st.bar_chart(coef, horizontal=True)
    st.caption("Standardised log-odds. Positive pushes toward churn (monthly contract, fiber).")

st.divider()
st.subheader("Score one customer")
b1, b2, _ = st.columns([1, 1, 4])
if b1.button("Use sample customer", key="sample", help="A short-tenure fiber customer on a monthly contract"):
    set_customer(SAMPLE)
if b2.button("Use safe customer", key="sample_safe"):
    set_customer(SAFE)
if "input" not in st.session_state:
    set_customer(EXAMPLE_CUSTOMER)

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    tenure = c1.slider("Tenure (months)", 1, 72, key="input")
    monthly = c2.number_input("Monthly charges ($)", 0.0, 500.0, key="monthly")
    total = c3.number_input("Total charges ($)", 0.0, 50000.0, key="total")
    cats = {}
    cols = st.columns(4)
    for i, col in enumerate(CATEGORICAL):
        cats[col] = cols[i % 4].selectbox(col, CHOICES[col], key=f"cat_{col}")
    senior = st.checkbox("Senior citizen", key="senior")
    names = list(bundle["models"])
    model_name = st.radio("Model", names, index=names.index(champion), horizontal=True, key="model")
    submitted = st.form_submit_button("Predict", key="run", type="primary")

if submitted:
    customer = {"Tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total}
    customer["SeniorCitizen"] = int(senior)
    customer.update(cats)
    frame = customer_frame(customer)
    prob = float(bundle["models"][model_name].predict_proba(frame)[0][1])
    other = {n: float(m.predict_proba(frame)[0][1]) for n, m in bundle["models"].items() if n != model_name}
    with st.container(key="result"):
        if prob >= 0.5:
            st.error(f"High risk of churn: {prob:.1%} ({model_name})")
        else:
            st.success(f"Low risk of churn: {prob:.1%} ({model_name})")
        st.caption("Other model: " + ", ".join(f"{n} {p:.1%}" for n, p in other.items()))
        st.json(customer)
else:
    st.info("Pick a sample or edit the form, then press Predict.")
