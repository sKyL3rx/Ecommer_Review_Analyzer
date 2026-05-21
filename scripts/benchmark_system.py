from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API_BASE_URL = os.getenv("BENCH_API_BASE_URL", "http://localhost:8000")
CONFIG_PATH = Path(
    os.getenv("BENCH_CONFIG_PATH", "configs/benchmark_system/appliances_test.json")
)

OUTPUT_PATH = Path(
    os.getenv(
        "BENCH_OUTPUT_PATH",
        "artifacts/benchmarks/local_system_benchmark.json",
    )
)

def select_product_ids_from_csv(config: dict[str, Any]) -> list[dict[str, Any]]:
    csv_path = Path(config["csv_path"])
    product_col = config.get("product_id_column", "product_id")
    top_k = int(config.get("top_k_products", 5))
    min_reviews = int(config.get("min_reviews_per_product", 20))

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, usecols=[product_col])

    counts = df[product_col].dropna().astype(str).value_counts()

    selected = counts[counts >= min_reviews].head(top_k)

    if selected.empty:
        raise RuntimeError(
            f"No products found with at least {min_reviews} reviews in {csv_path}"
        )

    return [
        {
            "product_id": product_id,
            "review_count_in_csv": int(review_count),
        }
        for product_id, review_count in selected.items()
    ]

def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Benchmark config not found: {CONFIG_PATH}")

    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

def timed_request(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: int = 60,
):
    url = f"{API_BASE_URL}{path}"
    start = time.perf_counter()

    response = requests.request(
        method=method,
        url=url,
        json=json_body,
        timeout=timeout,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    try:
        body = response.json()
    except Exception:
        body = response.text

    return {
        "method": method,
        "path": path,
        "status_code": response.status_code,
        "latency_ms": round(latency_ms, 2),
        "body": body,
    }


def wait_for_job(
    job_id: str,
    *,
    max_polls: int = 60,
    sleep_seconds: int = 2,
) -> dict[str, Any]:
    poll_samples = []
    started_at = time.perf_counter()
    final_status = None

    for _ in range(max_polls):
        response = timed_request("GET", f"/jobs/{job_id}", timeout=30)
        poll_samples.append(response)

        body = response.get("body")
        status = body.get("status") if isinstance(body, dict) else None

        if status in {"finished", "failed"}:
            final_status = status
            break

        time.sleep(sleep_seconds)

    return {
        "job_id": job_id,
        "final_status": final_status,
        "duration_seconds": round(time.perf_counter() - started_at, 2),
        "poll_samples": poll_samples,
    }

def summarize_latencies(samples: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [
        float(sample["latency_ms"])
        for sample in samples
        if "latency_ms" in sample
    ]

    if not latencies:
        return {}

    sorted_latencies = sorted(latencies)
    p95_index = int(0.95 * (len(sorted_latencies) - 1))

    return {
        "count": len(latencies),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted_latencies[p95_index], 2),
    }


def benchmark_product(
    product_id: str,
    *,
    review_count_in_csv: int,
    max_reviews: int,
    representative_k: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "product_id": product_id,
        "review_count_in_csv": review_count_in_csv,
        "steps": {},
    }

    get_product = timed_request(
        "GET",
        f"/products/{product_id}",
        timeout=30,
    )

    result["steps"]["get_product"] = get_product

    if get_product["status_code"] != 200:
        result["summary"] = {
            "status": "skipped",
            "reason": "product_not_found_or_api_error",
            "get_product_status_code": get_product["status_code"],
            "get_product_latency_ms": get_product["latency_ms"],
        }
        return result
    
    create_job = timed_request(
        "POST",
        f"/products/{product_id}/insights/jobs",
        json_body={
            "max_reviews": max_reviews,
            "representative_k": representative_k,
            "regenerate": False,
        },
        timeout=120,
    )

    result["steps"]["create_insight_job"] = create_job

    job_id = None
    if isinstance(create_job.get("body"), dict):
        job_id = create_job["body"].get("job_id")

    if job_id:
        job_wait = wait_for_job(job_id)
    else:
        job_wait = {
            "skipped": True,
            "reason": "No job_id returned. Insight probably already exists.",
            "duration_seconds": None,
            "final_status": None,
        }
    
    result["steps"]["job_wait"] = job_wait

    cached_reads = [
        timed_request(
            "GET",
            f"/products/{product_id}/insights",
            timeout=60,
        )
        for _ in range(5)
    ]
    result["steps"]["cached_insight_reads"] = cached_reads

    result["summary"] = {
        "status": "ok",
        "get_product_latency_ms": get_product["latency_ms"],
        "create_job_latency_ms": create_job["latency_ms"],
        "create_job_status_code": create_job["status_code"],
        "generation_final_status": job_wait.get("final_status"),
        "generation_duration_seconds": job_wait.get("duration_seconds"),
        "cached_insight_latency": summarize_latencies(cached_reads),
    }

    return result


def main() -> None:
    config = load_config()
    selected_products = select_product_ids_from_csv(config)

    max_reviews = int(config.get("max_reviews", 100))
    representative_k = int(config.get("representative_k", 5))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "api_base_url": API_BASE_URL,
        "benchmark_config": str(CONFIG_PATH),
        "dataset": {
            "name": config.get("name"),
            "csv_path": config.get("csv_path"),
            "product_id_column": config.get("product_id_column", "product_id"),
            "top_k_products": config.get("top_k_products"),
            "min_reviews_per_product": config.get("min_reviews_per_product"),
            "selected_products": selected_products,
            "max_reviews": max_reviews,
            "representative_k": representative_k,
        },
        "steps": {},
        "products": [],
        "summary": {},
    }

    health_samples = [
        timed_request("GET", "/health", timeout=30)
        for _ in range(5)
    ]

    results["steps"]["health"] = health_samples


    for item in selected_products:
        product_id = item["product_id"]
        review_count = item["review_count_in_csv"]

        print(f"Benchmarking product_id={product_id} reviews_in_csv={review_count}")

        product_result = benchmark_product(
            product_id=product_id,
            review_count_in_csv=review_count,
            max_reviews=max_reviews,
            representative_k=representative_k,
        )
        results["products"].append(product_result)

    
    create_job_samples = []
    cached_read_samples = []
    generation_durations = []

    successful_generations = 0
    failed_generations = 0
    skipped_or_existing = 0
    product_not_found = 0

    for product in results["products"]:
        summary = product.get("summary", {})

        if summary.get("reason") == "product_not_found_or_api_error":
            product_not_found += 1
            continue

        if "create_job_latency_ms" in summary:
            create_job_samples.append(
                {"latency_ms": summary["create_job_latency_ms"]}
            )

        cached_read_samples.extend(
            product.get("steps", {}).get("cached_insight_reads", [])
        )

        status = summary.get("generation_final_status")
        duration = summary.get("generation_duration_seconds")

        if duration is not None:
            generation_durations.append(duration)

        if status == "finished":
            successful_generations += 1
        elif status == "failed":
            failed_generations += 1
        else:
            skipped_or_existing += 1

    results["summary"] = {
        "health_latency": summarize_latencies(health_samples),
        "create_job_latency": summarize_latencies(create_job_samples),
        "cached_insight_latency": summarize_latencies(cached_read_samples),
        "generation_duration_seconds": {
            "count": len(generation_durations),
            "mean": round(statistics.mean(generation_durations), 2)
            if generation_durations
            else None,
            "max": round(max(generation_durations), 2)
            if generation_durations
            else None,
        },
        "successful_generations": successful_generations,
        "failed_generations": failed_generations,
        "skipped_or_existing_insights": skipped_or_existing,
        "product_not_found": product_not_found,
    }

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nWrote benchmark results to {OUTPUT_PATH}")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()

