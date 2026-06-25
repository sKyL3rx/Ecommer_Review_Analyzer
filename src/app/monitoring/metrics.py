from prometheus_client import Counter, Gauge, Histogram

api_requests_total = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "path", "status_code"],
)

api_request_latency_seconds = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "path", "status_code"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

insight_jobs_total = Counter(
    "insight_jobs_total",
    "Total insight jobs by final status",
    ["status"],
)

insight_job_duration_seconds = Histogram(
    "insight_job_duration_seconds",
    "Insight job duration in seconds",
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120),
)

summary_generation_duration_seconds = Histogram(
    "summary_generation_duration_seconds",
    "Summary generation duration in seconds",
    ["sentiment"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

cache_operations_total = Counter(
    "cache_operations_total",
    "Redis cache operations",
    ["operation", "result"],
)

insight_queue_depth = Gauge(
    "insight_queue_depth",
    "Number of jobs waiting in the insight RQ queue",
)

sentiment_source_total = Counter(
    "sentiment_source_total",
    "Resolved sentiment source counts",
    ["source"],
)
