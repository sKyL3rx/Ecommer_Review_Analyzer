from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.app.clients.summarizer_client import VLLMSummaryGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions for test.jsonl and measure latency."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("data/processed/phase2/trl_data/test.jsonl"),
    )

    parser.add_argument(
        "--output_df",
        type=Path,
        default=Path("artifacts/predictions/summary_predictions_testset.csv"),
    )

    parser.add_argument(
        "--output_metrics",
        type=Path,
        default=Path("artifacts/eval/metrics_summary_model.json"),
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="vLLM/OpenAI-compatible base URL",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default="demo-key",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="summary-sft",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_prompt_and_reference(messages: list[dict[str, str]]) -> tuple[str, str]:
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    assistant_msgs = [m["content"] for m in messages if m.get("role") == "assistant"]

    if not user_msgs:
        raise ValueError("No user message found in messages.")
    if not assistant_msgs:
        raise ValueError("No assistant message found in messages.")

    return user_msgs[-1].strip(), assistant_msgs[-1].strip()


def compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.array(latencies_ms, dtype=float)
    return {
        "count": int(len(arr)),
        "mean_ms": round(float(arr.mean()), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p55_ms": round(float(np.percentile(arr, 55)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
    }


def main() -> None:
    args = parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    args.output_df.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input_path)
    rows_for_df = []
    latencies_ms: list[float] = []

    summary_client = VLLMSummaryGenerator(
        base_url=args.base_url, max_tokens=args.max_tokens, temperature=args.temperature
    )
    for idx, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Row {idx} missing valid 'messages' field.")

        prompt, reference_answer = extract_prompt_and_reference(messages)

        start = time.perf_counter()

        generated_answer = summary_client(prompt=prompt, context="no")

        latency_ms = (time.perf_counter() - start) * 1000.0

        rows_for_df.append(
            {
                "prompt": prompt,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "latency_ms": round(latency_ms, 2),
                "messages": messages,
            }
        )
        latencies_ms.append(latency_ms)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(rows)} examples...")

    df = pd.DataFrame(rows_for_df)

    df.to_csv(args.output_df, index=False)

    metrics = compute_latency_stats(latencies_ms)
    with args.output_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Saved testset predictions to: {args.output_df}")
    print(f"Saved latency metrics to: {args.output_metrics}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
