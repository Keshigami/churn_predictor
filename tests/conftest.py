from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def trained_models() -> Path:
    """Train once into models/ if the bundle is missing, so tests run from a clean checkout."""
    out = HERE / "models"
    if not (out / "churn.joblib").exists():
        from train import train

        train(HERE / "data" / "telecom_churn.csv", out)
    return out
