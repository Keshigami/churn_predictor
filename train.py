"""Train Logistic Regression (baseline) and Random Forest (champion) once; save the bundle and metrics.

    uv run python train.py [--data data/telecom_churn.csv] [--out models]

Writes models/churn.joblib (both fitted pipelines plus training statistics for drift checks)
and models/metrics.json (accuracy, F1, ROC-AUC per model, feature importances).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import CATEGORICAL, FEATURES, NUMERIC, TARGET, make_preprocessor

HERE = Path(__file__).resolve().parent


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", make_preprocessor()),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", make_preprocessor()),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
    }


def evaluate(pipe: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = pipe.predict(x)
    proba = pipe.predict_proba(x)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }


def feature_names(pipe: Pipeline) -> list[str]:
    return [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]


def train(data: Path, out: Path, test_size: float = 0.2, seed: int = 42) -> dict:
    df = pd.read_csv(data)
    x, y = df[FEATURES], df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    models = build_models()
    metrics: dict = {"models": {}, "n_train": int(len(x_train)), "n_test": int(len(x_test))}
    for name, pipe in models.items():
        t0 = time.perf_counter()
        pipe.fit(x_train, y_train)
        metrics["models"][name] = evaluate(pipe, x_test, y_test)
        metrics["models"][name]["train_seconds"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        pipe.predict_proba(x_test.iloc[:1])
        metrics["models"][name]["single_prediction_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    rf = models["random_forest"]
    names = feature_names(rf)
    importances = dict(zip(names, rf.named_steps["clf"].feature_importances_.round(4).tolist(), strict=True))
    metrics["feature_importances"] = dict(sorted(importances.items(), key=lambda kv: -kv[1]))
    lr = models["logistic_regression"]
    coefs = lr.named_steps["clf"].coef_[0]
    metrics["lr_coefficients"] = dict(zip(feature_names(lr), coefs.round(4).tolist(), strict=True))

    # Champion = best ROC-AUC. Training statistics feed the monitoring API's drift check.
    champion = max(metrics["models"], key=lambda m: metrics["models"][m]["roc_auc"])
    metrics["champion"] = champion
    metrics["training_stats"] = {
        c: {"mean": float(x_train[c].mean()), "std": float(x_train[c].std(ddof=0))} for c in NUMERIC
    }
    metrics["training_stats"].update(
        {
            c: {"frequencies": {k: float(v) for k, v in x_train[c].value_counts(normalize=True).items()}}
            for c in CATEGORICAL
        }
    )
    metrics["churn_rate"] = float(y.mean())

    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "models": models,
        "champion": champion,
        "features": FEATURES,
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "training_stats": metrics["training_stats"],
        "metrics": metrics["models"],
    }
    joblib.dump(bundle, out / "churn.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "telecom_churn.csv")
    ap.add_argument("--out", type=Path, default=HERE / "models")
    args = ap.parse_args()
    if not args.data.exists():
        from data_gen import generate_churn_data

        args.data.parent.mkdir(parents=True, exist_ok=True)
        generate_churn_data().to_csv(args.data, index=False)
    m = train(args.data, args.out)
    for name, r in m["models"].items():
        print(f"{name:20s} acc={r['accuracy']:.3f} f1={r['f1']:.3f} auc={r['roc_auc']:.3f}")
    print(f"champion: {m['champion']}; top features: {list(m['feature_importances'])[:5]}")
    print(f"saved {args.out / 'churn.joblib'} and {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
