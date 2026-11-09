"""Feature schema shared by train.py, the Streamlit app and the monitoring API."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = [
    "InternetService", "Contract", "PaymentMethod", "PaperlessBilling", "Gender", "Partner", "Dependents"
]  # fmt: skip
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Churn"

CHOICES = {
    "InternetService": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    "PaperlessBilling": ["Yes", "No"],
    "Gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
}

EXAMPLE_CUSTOMER = {
    "Tenure": 6,
    "MonthlyCharges": 95.0,
    "TotalCharges": 570.0,
    "SeniorCitizen": 0,
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes",
    "Gender": "Female",
    "Partner": "No",
    "Dependents": "No",
}


def make_preprocessor() -> ColumnTransformer:
    """One-hot the categoricals with fixed category lists so unseen values never break scoring."""
    return ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            (
                "cat",
                OneHotEncoder(categories=[CHOICES[c] for c in CATEGORICAL], handle_unknown="ignore"),
                CATEGORICAL,
            ),
        ]
    )


def customer_frame(customer: dict) -> pd.DataFrame:
    """A one-row DataFrame in the training column order; missing fields fall back to the example."""
    row = {**EXAMPLE_CUSTOMER, **{k: v for k, v in customer.items() if k in FEATURES}}
    return pd.DataFrame([row], columns=FEATURES)
