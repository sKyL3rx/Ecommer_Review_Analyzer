from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests

TERMINAL_JOB_STATUSES = {"finished", "failed", "stopped", "canceled", "cancelled"}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    idx = round((len(sorted_values) - 1) * p)
    return sorted_values[idx]


def summarize_latencies(latencies_ms: list[float], statuses: list[int]) -> dict[str, Any]:
    success_count = sum(1 for status in statuses if 200 <= status < 300)
    total = len(statuses)

    return {
        "requests": total,
        "success_count": success_count,
        "success_rate": round(success_count / total, 4) if total else 0.0,
        "avg_ms": round(statistics.mean(latencies_ms), 2) if latencies_ms else 0.0,
        "p50_ms": round(percentile(latencies_ms, 0.50), 2),
        "p95_ms": round(percentile(latencies_ms, 0.95), 2),
        "p99_ms": round(percentile(latencies_ms, 0.99), 2),
        "min_ms": round(min(latencies_ms), 2) if latencies_ms else 0.0,
        "max_ms": round(max(latencies_ms), 2) if latencies_ms else 0.0,
        "status_counts": {str(status): statuses.count(status) for status in sorted(set(statuses))},
    }


def timed_get(url: str, timeout_seconds: int) -> tuple[requests.Response | None, float, str | None]:
    start = time.perf_counter()

    try:
        response = requests.get(url, timeout=timeout_seconds)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return response, elapsed_ms, None
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return None, elapsed_ms, str(exc)


def timed_post_json(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> tuple[requests.Response | None, float, str | None]:
    start = time.perf_counter()

    try:
        response = requests.post(url, json=payload, timeout=timeout_seconds)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return response, elapsed_ms, None
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return None, elapsed_ms, str(exc)


def require_json(response: requests.Response, context: str) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{context} did not return JSON: {response.text[:300]}") from exc

    if not isinstance(body, dict):
        raise RuntimeError(f"{context} JSON was not an object: {body!r}")

    return body


def get_product_ids(base_url: str, limit: int, timeout_seconds: int) -> list[str]:
    response, _, error = timed_get(
        f"{base_url}/products?limit={limit}&offset=0",
        timeout_seconds=timeout_seconds,
    )

    if response is None:
        raise RuntimeError(f"Failed to fetch products: {error}")

    if response.status_code != 200:
        raise RuntimeError(f"GET /products failed: {response.status_code} {response.text[:300]}")

    body = require_json(response, "GET /products")
    items = body.get("items", [])

    if not isinstance(items, list) or not items:
        raise RuntimeError("No products returned. Did you load serving data?")

    product_ids = [
        str(item.get("product_id")).strip()
        for item in items
        if isinstance(item, dict) and item.get("product_id")
    ]

    if not product_ids:
        raise RuntimeError("No product_id values returned from /products.")

    return product_ids


def benchmark_get_endpoint(
    name: str,
    url: str,
    request_count: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    statuses: list[int] = []
    errors: list[str] = []

    for _ in range(request_count):
        response, latency_ms, error = timed_get(url, timeout_seconds=timeout_seconds)
        latencies_ms.append(latency_ms)

        if response is None:
            statuses.append(0)
            errors.append(error or "unknown request error")
        else:
            statuses.append(response.status_code)

    result = summarize_latencies(latencies_ms, statuses)
    result["name"] = name
    result["url"] = url
    result["errors_sample"] = errors[:5]
    return result


def create_insight_job(
    base_url: str,
    product_id: str,
    max_reviews: int,
    representative_k: int,
    regenerate: bool,
    timeout_seconds: int,
) -> tuple[str | None, bool, float, int, dict[str, Any]]:
    response, latency_ms, error = timed_post_json(
        f"{base_url}/products/{product_id}/insights/jobs",
        {
            "max_reviews": max_reviews,
            "representative_k": representative_k,
            "regenerate": regenerate,
        },
        timeout_seconds=timeout_seconds,
    )

    if response is None:
        return None, False, latency_ms, 0, {"error": error}

    status_code = response.status_code
    body = require_json(response, "POST insight job")

    job_id = body.get("job_id")
    cached = bool(body.get("cached", False))

    return str(job_id) if job_id else None, cached, latency_ms, status_code, body


def poll_job_until_terminal(
    base_url: str,
    job_id: str,
    poll_interval_seconds: float,
    job_timeout_seconds: int,
    timeout_seconds: int,
) -> tuple[str, float, dict[str, Any]]:
    start = time.perf_counter()
    deadline = time.time() + job_timeout_seconds
    last_body: dict[str, Any] = {}

    while time.time() < deadline:
        response, _, error = timed_get(
            f"{base_url}/jobs/{job_id}",
            timeout_seconds=timeout_seconds,
        )

        if response is None:
            last_body = {"error": error}
            time.sleep(poll_interval_seconds)
            continue

        if response.status_code != 200:
            last_body = {
                "status_code": response.status_code,
                "body": response.text[:300],
            }
            time.sleep(poll_interval_seconds)
            continue

        body = require_json(response, "GET job status")
        last_body = body

        status = str(body.get("status", "")).lower()
        if status in TERMINAL_JOB_STATUSES:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return status, elapsed_ms, body

        time.sleep(poll_interval_seconds)

    elapsed_ms = (time.perf_counter() - start) * 1000
    return "timeout", elapsed_ms, last_body


def warmup_insight_job(
    base_url: str,
    product_id: str,
    max_reviews: int,
    representative_k: int,
    timeout_seconds: int,
    job_timeout_seconds: int,
) -> dict[str, Any]:
    job_id, cached, _, status_code, body = create_insight_job(
        base_url=base_url,
        product_id=product_id,
        max_reviews=max_reviews,
        representative_k=representative_k,
        regenerate=False,
        timeout_seconds=timeout_seconds,
    )

    if cached:
        return body

    if status_code != 200 or not job_id:
        raise RuntimeError(
            f"Could not create warmup insight job: status={status_code}, body={body}"
        )

    status, _, job_body = poll_job_until_terminal(
        base_url=base_url,
        job_id=job_id,
        poll_interval_seconds=1.0,
        job_timeout_seconds=job_timeout_seconds,
        timeout_seconds=timeout_seconds,
    )

    if status != "finished":
        raise RuntimeError(f"Warmup insight job did not finish: status={status}, body={job_body}")

    response, _, error = timed_get(
        f"{base_url}/products/{product_id}/insights",
        timeout_seconds=timeout_seconds,
    )

    if response is None:
        raise RuntimeError(f"Could not fetch warmup insight: {error}")

    if response.status_code != 200:
        raise RuntimeError(
            f"Warmup GET insight failed: {response.status_code} {response.text[:300]}"
        )

    return require_json(response, "GET warmup insight")


def benchmark_async_jobs(
    base_url: str,
    product_ids: list[str],
    job_count: int,
    max_reviews: int,
    representative_k: int,
    timeout_seconds: int,
    job_timeout_seconds: int,
) -> dict[str, Any]:
    submit_latencies_ms: list[float] = []
    submit_statuses: list[int] = []
    finish_latencies_ms: list[float] = []
    final_statuses: list[str] = []
    job_ids: list[str] = []
    errors: list[dict[str, Any]] = []

    for job_index in range(job_count):
        product_id = product_ids[job_index % len(product_ids)]

        job_id, cached, submit_ms, status_code, body = create_insight_job(
            base_url=base_url,
            product_id=product_id,
            max_reviews=max_reviews,
            representative_k=representative_k,
            regenerate=True,
            timeout_seconds=timeout_seconds,
        )

        submit_latencies_ms.append(submit_ms)
        submit_statuses.append(status_code)

        if status_code != 200 or not job_id:
            final_statuses.append("submit_failed")
            finish_latencies_ms.append(0.0)
            errors.append(
                {
                    "stage": "submit",
                    "product_id": product_id,
                    "status_code": status_code,
                    "body": body,
                }
            )
            continue

        job_ids.append(job_id)

        if cached:
            final_statuses.append("cached")
            finish_latencies_ms.append(0.0)
            continue

        final_status, finish_ms, final_body = poll_job_until_terminal(
            base_url=base_url,
            job_id=job_id,
            poll_interval_seconds=1.0,
            job_timeout_seconds=job_timeout_seconds,
            timeout_seconds=timeout_seconds,
        )

        final_statuses.append(final_status)
        finish_latencies_ms.append(finish_ms)

        if final_status != "finished":
            errors.append(
                {
                    "stage": "poll",
                    "product_id": product_id,
                    "job_id": job_id,
                    "status": final_status,
                    "body": final_body,
                }
            )

    finished_count = final_statuses.count("finished")
    terminal_count = len(final_statuses)

    return {
        "product_ids": product_ids,
        "job_count": job_count,
        "finished_count": finished_count,
        "success_rate": round(finished_count / terminal_count, 4) if terminal_count else 0.0,
        "final_status_counts": {
            status: final_statuses.count(status) for status in sorted(set(final_statuses))
        },
        "submit_latency": summarize_latencies(submit_latencies_ms, submit_statuses),
        "finish_latency_ms": {
            "avg_ms": round(statistics.mean(finish_latencies_ms), 2)
            if finish_latencies_ms
            else 0.0,
            "p50_ms": round(percentile(finish_latencies_ms, 0.50), 2),
            "p95_ms": round(percentile(finish_latencies_ms, 0.95), 2),
            "p99_ms": round(percentile(finish_latencies_ms, 0.99), 2),
            "min_ms": round(min(finish_latencies_ms), 2) if finish_latencies_ms else 0.0,
            "max_ms": round(max(finish_latencies_ms), 2) if finish_latencies_ms else 0.0,
        },
        "job_ids_sample": job_ids[:5],
        "errors_sample": errors[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local Product Review Intelligence serving stack."
    )
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--product-limit", type=int, default=10)
    parser.add_argument(
        "--product-id",
        action="append",
        default=None,
        help="Product ID to benchmark. Can be passed multiple times. Overrides --product-limit.",
    )
    parser.add_argument("--max-reviews", type=int, default=50)
    parser.add_argument("--representative-k", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=15)
    parser.add_argument("--job-timeout-seconds", type=int, default=120)
    parser.add_argument("--output", default="artifacts/benchmarks/local_serving_benchmark.json")
    args = parser.parse_args()

    base_url = args.api_base_url.rstrip("/")

    if args.product_id:
        product_ids = [
            str(product_id).strip() for product_id in args.product_id if str(product_id).strip()
        ]
    else:
        product_ids = get_product_ids(
            base_url=base_url,
            limit=args.product_limit,
            timeout_seconds=args.timeout_seconds,
        )

    if not product_ids:
        raise RuntimeError("No product IDs selected for benchmark.")

    primary_product_id = product_ids[0]

    warmup_insight_job(
        base_url=base_url,
        product_id=primary_product_id,
        max_reviews=args.max_reviews,
        representative_k=args.representative_k,
        timeout_seconds=args.timeout_seconds,
        job_timeout_seconds=args.job_timeout_seconds,
    )

    results = {
        "environment": {
            "api_base_url": base_url,
            "product_ids": product_ids,
            "primary_product_id": primary_product_id,
            "product_count": len(product_ids),
            "requests_per_get_endpoint": args.requests,
            "async_job_count": args.jobs,
            "max_reviews": args.max_reviews,
            "representative_k": args.representative_k,
        },
        "api_latency": {
            "health": benchmark_get_endpoint(
                name="health",
                url=f"{base_url}/health",
                request_count=args.requests,
                timeout_seconds=args.timeout_seconds,
            ),
            "products_limit_5": benchmark_get_endpoint(
                name="products_limit_5",
                url=f"{base_url}/products?limit=5&offset=0",
                request_count=args.requests,
                timeout_seconds=args.timeout_seconds,
            ),
        },
        "async_jobs": benchmark_async_jobs(
            base_url=base_url,
            product_ids=product_ids,
            job_count=args.jobs,
            max_reviews=args.max_reviews,
            representative_k=args.representative_k,
            timeout_seconds=args.timeout_seconds,
            job_timeout_seconds=args.job_timeout_seconds,
        ),
        "cached_insight": benchmark_get_endpoint(
            name="cached_insight",
            url=f"{base_url}/products/{primary_product_id}/insights",
            request_count=args.requests,
            timeout_seconds=args.timeout_seconds,
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Wrote benchmark report to {output_path}")


if __name__ == "__main__":
    main()
