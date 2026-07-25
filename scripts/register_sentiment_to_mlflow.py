from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline


def flatten_numeric_metrics(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}

    for key, value in data.items():
        metric_name = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"

        if isinstance(value, bool):
            continue

        if isinstance(value, int | float):
            metrics[metric_name] = float(value)

        elif isinstance(value, dict):
            metrics.update(flatten_numeric_metrics(value, prefix=metric_name))

    return metrics


def log_metrics_file(metrics_path: str | None) -> None:
    if not metrics_path:
        return

    path = Path(metrics_path)
    if not path.exists():
        print(f"Metrics file not found, skipping: {path}")
        return

    metrics_json = json.loads(path.read_text(encoding="utf-8"))
    metrics = flatten_numeric_metrics(metrics_json)

    for key, val in metrics.items():
        mlflow.log_metric(key, val)

    mlflow.log_artifact(str(path), artifact_path="reports")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register sentiment model to MLFLOW server")
    parser.add_argument("--vectorizer-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--registered-model-name", default="ecommerce-review-sentiment")
    parser.add_argument("--experiment-name", default="ecommerce_review_sentiment")
    parser.add_argument("--run-name", default="sentiment_local_baseline")
    parser.add_argument("--training-env", default="local")
    parser.add_argument("--dataset-version", default="appliances-local-v1")
    parser.add_argument("--alias", default="local-baseline")

    args = parser.parse_args()

    vectorizer_path = Path(args.vectorizer_path)
    model_path = Path(args.model_path)

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(model_path)

    pipeline = Pipeline(steps=[("tfidf", vectorizer), ("classifier", classifier)])

    input_example = pd.DataFrame(
        {
            "text": [
                "This product works great and is easy to use.",
                "It broke after two days and was disappointing.",
            ]
        }
    )

    predictions = pipeline.predict(input_example["text"])

    signature = infer_signature(input_example, predictions)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "task": "sentiment-classification",
                "model_family": "tfidf_sklearn",
                "training_env": args.training_env,
                "dataset_version": args.dataset_version,
                "vectorizer_path": str(vectorizer_path),
                "model_path": str(model_path),
            }
        )

        log_metrics_file(args.metrics_path)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=args.registered_model_name,
            signature=signature,
            input_example=input_example,
        )

    client = MlflowClient()

    versions = client.search_model_versions(f"name='{args.registered_model_name}'")

    if not versions:
        raise RuntimeError(f"No versions found for {args.registered_model_name}")

    latest = max(versions, key=lambda version: int(version.version))

    client.set_model_version_tag(
        args.registered_model_name,
        latest.version,
        "task",
        "sentiment-classification",
    )

    client.set_model_version_tag(
        args.registered_model_name,
        latest.version,
        "model_family",
        "tfidf_sklearn",
    )

    client.set_model_version_tag(
        args.registered_model_name,
        latest.version,
        "training_env",
        args.training_env,
    )

    client.set_model_version_tag(
        args.registered_model_name,
        latest.version,
        "dataset_version",
        args.dataset_version,
    )

    client.set_registered_model_alias(
        args.registered_model_name,
        args.alias,
        latest.version,
    )

    client.set_registered_model_alias(
        args.registered_model_name,
        "champion-sentiment",
        latest.version,
    )

    print(f"Registered model: {args.registered_model_name}")
    print(f"Version: {latest.version}")
    print(f"Alias: {args.alias} -> version {latest.version}")
    print(f"Alias: champion-sentiment -> version {latest.version}")


if __name__ == "__main__":
    main()
