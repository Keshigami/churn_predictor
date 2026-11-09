import json

import joblib

from features import EXAMPLE_CUSTOMER, customer_frame
from train import train


def test_train_writes_bundle_and_metrics(tmp_path):
    metrics = train(__import__("pathlib").Path(__file__).parents[1] / "data" / "telecom_churn.csv", tmp_path)
    assert set(metrics["models"]) == {"logistic_regression", "random_forest"}
    for m in metrics["models"].values():
        assert 0.6 < m["accuracy"] <= 1.0
        assert 0.0 < m["f1"] <= 1.0
        assert 0.6 < m["roc_auc"] <= 1.0
    assert metrics["champion"] in metrics["models"]
    assert len(metrics["feature_importances"]) > 10
    assert (tmp_path / "churn.joblib").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["champion"] == metrics["champion"]


def test_bundle_scores_a_customer(trained_models):
    bundle = joblib.load(trained_models / "churn.joblib")
    frame = customer_frame(EXAMPLE_CUSTOMER)
    for name, pipe in bundle["models"].items():
        p = pipe.predict_proba(frame)[0][1]
        assert 0.0 <= p <= 1.0, name
    assert "Tenure" in bundle["training_stats"]
    assert bundle["training_stats"]["Contract"]["frequencies"]


def test_unknown_category_does_not_crash(trained_models):
    bundle = joblib.load(trained_models / "churn.joblib")
    frame = customer_frame({**EXAMPLE_CUSTOMER, "Contract": "Lifetime"})
    assert 0.0 <= bundle["models"]["random_forest"].predict_proba(frame)[0][1] <= 1.0
