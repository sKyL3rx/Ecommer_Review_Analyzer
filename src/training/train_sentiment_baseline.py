from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

import mlflow

DEFAULT_TRAIN_PATH = Path("data/processed/train.csv")
DEFAULT_VAL_PATH = Path("data/processed/val.csv")
DEFAULT_MODEL_DIR = Path("artifacts/models")
DEFAULT_REPORT_DIR = Path("artifacts/reports")
DEFAULT_EXPERIMENT_NAME = "amazon_appliances_sentiment"
DEFAULT_RUN_NAME = "tfidf_logreg_baseline"
VALID_LABELS = ["negative", "neutral", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline sentiment classifier using TF-IDF + Logistic Regression."
    )
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val_path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--target_column", type=str, default="sentiment_label")
    parser.add_argument("--max_features", type=int, default=30000)
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--class_weight", type=str, default="balanced")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--disable_mlflow",
        action="store_true",
        default=False,
        help="Disable MLflow logging even if mlflow is installed.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, text_column: str, target_column: str) -> None:
    missing = {text_column, target_column} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_dataset(path: Path, text_column: str, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    validate_columns(df, text_column, target_column)

    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    df[target_column] = df[target_column].fillna("").astype(str)
    df = df[df[text_column].str.strip() != ""]
    df = df[df[target_column].isin(VALID_LABELS)].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid rows left after cleaning dataset: {path}")

    return df


def build_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        lowercase=False,
    )


def build_model(args: argparse.Namespace) -> LogisticRegression:
    class_weight: str | dict[str, float] | None = args.class_weight
    if class_weight.lower() == "none":
        class_weight = None

    return LogisticRegression(
        max_iter=args.max_iter,
        class_weight=class_weight,
        random_state=args.random_state,
        n_jobs=-1,
    )


def compute_metrics(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }


def save_outputs(
    *,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    metrics: dict[str, float],
    report_text: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_dir: Path,
    report_dir: Path,
) -> dict[str, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = model_dir / "tfidf.joblib"
    model_path = model_dir / "sentiment_model.joblib"
    metrics_path = report_dir / "val_metrics.json"
    report_path = report_dir / "val_classification_report.txt"
    summary_path = report_dir / "training_summary.json"

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_path.write_text(report_text, encoding="utf-8")

    summary = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_label_distribution": train_df["sentiment_label"].value_counts().to_dict(),
        "val_label_distribution": val_df["sentiment_label"].value_counts().to_dict(),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "vectorizer_path": vectorizer_path,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "report_path": report_path,
        "summary_path": summary_path,
    }


def configure_mlflow(args: argparse.Namespace) -> bool:
    if args.disable_mlflow or mlflow is None:
        return False

    mlflow.set_experiment(args.experiment_name)
    return True


def main() -> None:
    args = parse_args()

    train_df = load_dataset(args.train_path, args.text_column, args.target_column)
    val_df = load_dataset(args.val_path, args.text_column, args.target_column)

    X_train = train_df[args.text_column]
    y_train = train_df[args.target_column]
    X_val = val_df[args.text_column]
    y_val = val_df[args.target_column]

    vectorizer = build_vectorizer(args)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = build_model(args)

    use_mlflow = configure_mlflow(args)
    run_context = mlflow.start_run(run_name=args.run_name) if use_mlflow else None

    try:
        if use_mlflow:
            mlflow.log_params(
                {
                    "model_type": "logistic_regression",
                    "vectorizer": "tfidf",
                    "train_path": str(args.train_path),
                    "val_path": str(args.val_path),
                    "text_column": args.text_column,
                    "target_column": args.target_column,
                    "max_features": args.max_features,
                    "ngram_min": args.ngram_min,
                    "ngram_max": args.ngram_max,
                    "min_df": args.min_df,
                    "max_iter": args.max_iter,
                    "class_weight": args.class_weight,
                    "random_state": args.random_state,
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                }
            )

        model.fit(X_train_vec, y_train)
        val_pred = model.predict(X_val_vec)

        metrics = compute_metrics(y_val, val_pred)

        report_text = classification_report(
            y_val,
            val_pred,
            labels=VALID_LABELS,
            digits=4,
            zero_division=0,
        )

        output_paths = save_outputs(
            vectorizer=vectorizer,
            model=model,
            metrics=metrics,
            report_text=report_text,
            train_df=train_df,
            val_df=val_df,
            model_dir=args.model_dir,
            report_dir=args.report_dir,
        )

        if use_mlflow:
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
            mlflow.log_artifact(str(output_paths["metrics_path"]))
            mlflow.log_artifact(str(output_paths["report_path"]))
            mlflow.log_artifact(str(output_paths["summary_path"]))
            mlflow.sklearn.log_model(model, artifact_path="model")

        print("Training completed.")
        print(f"Train rows: {len(train_df):,}")
        print(f"Validation rows: {len(val_df):,}")
        print("Validation metrics:")
        for key, value in metrics.items():
            print(f"- {key}: {value:.4f}")
        print("Saved artifacts:")
        for key, path in output_paths.items():
            print(f"- {key}: {path}")

    finally:
        if run_context is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
