from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


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

    metrics_json = read_json(path)
    metrics = flatten_numeric_metrics(metrics_json)

    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    mlflow.log_artifact(str(path), artifact_path="reports")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register summary LoRA adapter to MLflow.")
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--registered-model-name", default="ecommerce-review-summary")
    parser.add_argument("--experiment-name", default="ecommerce_review_summary")
    parser.add_argument("--run-name", default="summary_lora")
    parser.add_argument("--summary-model-version", default="summary-lora-local-v1")
    parser.add_argument("--training-env", default="local")
    parser.add_argument("--dataset-version", default="appliances-summary-local-v1")
    parser.add_argument("--alias", default="local-lora")
    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)

    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")

    adapter_config_path = lora_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json: {adapter_config_path}")

    has_safetensors = (lora_dir / "adapter_model.safetensors").exists()

    if not has_safetensors:
        raise FileNotFoundError("Missing adapter weights.")

    adapter_config = read_json(adapter_config_path)

    base_model_name = adapter_config.get("base_model_name_or_path") or "unknown-base-model"

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "task": "review-summarization",
                "model_type": "lora_adapter",
                "base_model_name": base_model_name,
                "summary_model_version": args.summary_model_version,
                "training_env": args.training_env,
                "dataset_version": args.dataset_version,
                "peft_type": adapter_config.get("peft_type", "LORA"),
                "r": adapter_config.get("r"),
                "lora_alpha": adapter_config.get("lora_alpha"),
                "lora_dropout": adapter_config.get("lora_dropout"),
                "target_modules": str(adapter_config.get("target_modules")),
            }
        )

        log_metrics_file(args.metrics_path)

        mlflow.log_artifacts(
            str(lora_dir),
            artifact_path="lora_adapter",
        )

        client = MlflowClient()

        try:
            client.get_registered_model(args.registered_model_name)
        except Exception:
            client.create_registered_model(args.registered_model_name)

        model_version = client.create_model_version(
            name=args.registered_model_name,
            source=f"runs:/{run.info.run_id}/lora_adapter",
            run_id=run.info.run_id,
        )

    client.set_model_version_tag(
        args.registered_model_name,
        model_version.version,
        "task",
        "review-summarization",
    )
    client.set_model_version_tag(
        args.registered_model_name,
        model_version.version,
        "model_type",
        "lora_adapter",
    )
    client.set_model_version_tag(
        args.registered_model_name,
        model_version.version,
        "base_model_name",
        str(base_model_name),
    )
    client.set_model_version_tag(
        args.registered_model_name,
        model_version.version,
        "summary_model_version",
        args.summary_model_version,
    )
    client.set_model_version_tag(
        args.registered_model_name,
        model_version.version,
        "training_env",
        args.training_env,
    )
    client.set_model_version_tag(
        args.registered_model_name,
        model_version.version,
        "dataset_version",
        args.dataset_version,
    )

    client.set_registered_model_alias(
        args.registered_model_name,
        args.alias,
        model_version.version,
    )
    client.set_registered_model_alias(
        args.registered_model_name,
        "champion-summary",
        model_version.version,
    )

    print(f"Registered model: {args.registered_model_name}")
    print(f"Version: {model_version.version}")
    print(f"Source: runs:/{run.info.run_id}/lora_adapter")
    print(f"Alias: {args.alias} -> version {model_version.version}")
    print(f"Alias: champion-summary -> version {model_version.version}")
    print(f"Base model: {base_model_name}")


if __name__ == "__main__":
    main()
