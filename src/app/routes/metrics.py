from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.app.monitoring.metrics import insight_queue_depth

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics() -> Response:
    try:
        from src.app.jobs.queue import insight_queue

        queue_count = getattr(insight_queue, "count", None)
        if callable(queue_count):
            queue_count = queue_count()
        elif queue_count is None:
            queue_count = len(insight_queue)

        insight_queue_depth.set(float(queue_count))
    except Exception:
        pass

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
