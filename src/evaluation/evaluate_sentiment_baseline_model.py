from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import mlflow

DEFAULT_TEST_PATH = Path("data/processed/test.csv")
DEFAULT_MODEL_PATH = Path("artifacts/models/sentiment_model.joblib")
DEFAULT_VECTORIZER_PATH = Path("artifacts/models/tfidf.joblib")
DEFAULT_REPORT_DIR = Path("artifacts/reports")
DEFAULT_EXPERIMENT_NAME = "amazon_appliances_sentiment"
DEFAULT_RUN_NAME = "tfidf_logreg_test_eval"
VALID_LABELS = ["negative", "neutral", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained baseline sentiment classifier on the test split "
            "and save metrics, reports, and confusion matrix artifacts."
        )
    )
    parser.add_argument("--test_path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vectorizer_path", type=Path, default=DEFAULT_VECTORIZER_PATH)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--target_column", type=str, default="sentiment_label")
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--disable_mlflow",
        action="store_true",
        help="Disable MLflow logging even if mlflow is installed.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, text_column: str, target_column: str) -> None:
    missing = {text_column, target_column} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_test_dataset(path: Path, text_column: str, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")

    df = pd.read_csv(path)
    validate_columns(df, text_column, target_column)

    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    df[target_column] = df[target_column].fillna("").astype(str)
    df = df[df[text_column].str.strip() != ""]
    df = df[df[target_column].isin(VALID_LABELS)].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid rows left after cleaning test dataset: {path}")

    return df


def load_artifact(path: Path, artifact_name: str):
    if not path.exists():
        raise FileNotFoundError(f"{artifact_name} not found: {path}")
    return joblib.load(path)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=VALID_LABELS)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Sentiment Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    *,
    metrics: dict[str, float],
    report_text: str,
    test_df: pd.DataFrame,
    y_pred: pd.Series,
    y_proba,
    report_dir: Path,
    target_column: str,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = report_dir / "metrics.json"
    report_path = report_dir / "classification_report.txt"
    confusion_matrix_path = report_dir / "confusion_matrix.png"
    predictions_path = report_dir / "test_predictions.csv"
    summary_path = report_dir / "evaluation_summary.json"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_path.write_text(report_text, encoding="utf-8")
    save_confusion_matrix(test_df[target_column], y_pred, confusion_matrix_path)

    prediction_df = test_df.copy()
    prediction_df["predicted_sentiment"] = y_pred

    if y_proba is not None:
        for idx, label in enumerate(VALID_LABELS):
            if idx < y_proba.shape[1]:
                prediction_df[f"proba_{label}"] = y_proba[:, idx]
        prediction_df["prediction_confidence"] = y_proba.max(axis=1)

    prediction_df.to_csv(predictions_path, index=False)

    summary = {
        "test_rows": int(len(test_df)),
        "test_label_distribution": test_df[target_column].value_counts().to_dict(),
        "predicted_label_distribution": pd.Series(y_pred).value_counts().to_dict(),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "metrics_path": metrics_path,
        "report_path": report_path,
        "confusion_matrix_path": confusion_matrix_path,
        "predictions_path": predictions_path,
        "summary_path": summary_path,
    }


def configure_mlflow(args: argparse.Namespace) -> bool:
    if args.disable_mlflow or mlflow is None:
        return False

    mlflow.set_experiment(args.experiment_name)
    return True


def main() -> None:
    args = parse_args()

    test_df = load_test_dataset(args.test_path, args.text_column, args.target_column)
    vectorizer = load_artifact(args.vectorizer_path, "Vectorizer")
    model = load_artifact(args.model_path, "Model")
    target_column = args.target_column

    X_test = test_df[args.text_column]
    y_test = test_df[args.target_column]
    X_test_vec = vectorizer.transform(X_test)

    use_mlflow = configure_mlflow(args)
    run_context = mlflow.start_run(run_name=args.run_name) if use_mlflow else None

    try:
        if use_mlflow:
            mlflow.log_params(
                {
                    "test_path": str(args.test_path),
                    "model_path": str(args.model_path),
                    "vectorizer_path": str(args.vectorizer_path),
                    "text_column": args.text_column,
                    "target_column": args.target_column,
                    "test_rows": len(test_df),
                }
            )

        y_pred = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec) if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred)
        report_text = classification_report(
            y_test,
            y_pred,
            labels=VALID_LABELS,
            digits=4,
            zero_division=0,
        )

        output_paths = save_outputs(
            metrics=metrics,
            report_text=report_text,
            test_df=test_df,
            y_pred=pd.Series(y_pred),
            y_proba=y_proba,
            report_dir=args.report_dir,
            target_column = target_column,
        )

        if use_mlflow:
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
            mlflow.log_artifact(str(output_paths["metrics_path"]))
            mlflow.log_artifact(str(output_paths["report_path"]))
            mlflow.log_artifact(str(output_paths["confusion_matrix_path"]))
            mlflow.log_artifact(str(output_paths["predictions_path"]))
            mlflow.log_artifact(str(output_paths["summary_path"]))

        print("Evaluation completed.")
        print(f"Test rows: {len(test_df):,}")
        print("Test metrics:")
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
