"""Synthetic telecom churn data: same generator and seed as the original portfolio, so the CSV reproduces."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def generate_churn_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    data = {
        "CustomerID": range(1, n_samples + 1),
        "Tenure": np.random.randint(1, 72, n_samples),
        "MonthlyCharges": np.random.uniform(20, 120, n_samples),
        "TotalCharges": np.zeros(n_samples),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "PaymentMethod": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_samples
        ),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples),
        "Gender": np.random.choice(["Female", "Male"], n_samples),
        "SeniorCitizen": np.random.choice([0, 1], n_samples),
        "Partner": np.random.choice(["Yes", "No"], n_samples),
        "Dependents": np.random.choice(["Yes", "No"], n_samples),
    }
    df = pd.DataFrame(data)
    df["TotalCharges"] = df["Tenure"] * df["MonthlyCharges"] + np.random.normal(0, 10, n_samples)
    # Simplified churn logic: month-to-month and fiber raise risk, long tenure lowers it.
    churn_prob = (
        0.5 * (df["Contract"] == "Month-to-month")
        + 0.3 * (df["InternetService"] == "Fiber optic")
        - 0.2 * (df["Tenure"] > 24)
        + 0.1 * (df["MonthlyCharges"] > 80)
    )
    churn_prob = (churn_prob - churn_prob.min()) / (churn_prob.max() - churn_prob.min())
    df["Churn"] = (np.random.random(n_samples) < churn_prob).astype(int)
    return df


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "telecom_churn.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_churn_data().to_csv(out, index=False)
    print(f"wrote {out}")
