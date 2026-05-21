from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

api_requests_total = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "path", "status_code"],
)

api_request_latency_seconds = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "path"],
)

insight_queue_depth = Gauge(
    "insight_queue_depth",
    "Number of jobs waiting in the insight RQ queue",
)

def metrics_response() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )