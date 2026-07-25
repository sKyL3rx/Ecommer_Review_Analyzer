from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.app.core.config import settings
from src.app.jobs.queue import insight_queue
from src.app.schemas import InsightJobRequest, InsightJobResponse
from src.app.storage.cache import (
    get_json_cache,
    product_insight_cache_key,
    set_json_cache,
)
from src.app.storage.db import get_session
from src.app.storage.models import InsightJob
from src.app.storage.repositories import get_product_insight
from src.app.workers.insight_tasks import generate_product_insight_task

router = APIRouter(tags=["insights"])


@router.post(
    "/products/{product_id}/insights/jobs",
    response_model=InsightJobResponse,
)
def create_insight_job(
    product_id: str,
    payload: InsightJobRequest,
    session: Session = Depends(get_session),
) -> InsightJobResponse:
    """
    Async write/generation path.

    Flow:
    1. Enqueue RQ job.
    2. Worker reads product/reviews from Postgres.
    3. Worker runs service.generate().
    4. Worker classifies sentiment and calls vLLM to generate summarization.
    5. Worker saves result to Postgres + Redis cache.

    If an insight already exists and regenerate=false, return immediately
    instead of enqueueing.
    """

    cache_key = product_insight_cache_key(
        product_id=product_id,
        model_version=settings.summary_model_version,
    )

    if not payload.regenerate:
        cached = get_json_cache(cache_key)
        if cached is not None:
            return InsightJobResponse(
                job_id=None,
                status="already_exists",
                product_id=product_id,
                cached=True,
                message=(
                    "Insight already exists in Redis cache. "
                    "Use regenerate=true to enqueue a new job."
                ),
            )

        existing = get_product_insight(
            session=session,
            product_id=product_id,
            model_version=settings.summary_model_version,
        )

        if existing is not None:
            set_json_cache(
                key=cache_key,
                value=existing.payload,
                ttl_seconds=settings.redis_cache_ttl_seconds
            )

            return InsightJobResponse(
                job_id=None,
                status="already_exists",
                product_id=product_id,
                cached=False,
                message=(
                    "Insight already exists in Postgres. Use regenerate=true to enqueue a new job."
                ),
            )

    job = insight_queue.enqueue(
        generate_product_insight_task,
        product_id,
        payload.max_reviews,
        payload.representative_k,
        payload.regenerate,
        job_timeout="15m",
        result_ttl=86400,
        failure_ttl=86400,
    )

    db_job = InsightJob(
        job_id=job.id,
        product_id=product_id,
        status=job.get_status(),
        error=None,
    )

    session.add(db_job)
    session.commit()

    return InsightJobResponse(
        job_id=job.id,
        status=job.get_status(),
        product_id=product_id,
        cached=False,
        message="Insight generation job enqueued.",
    )


@router.get("/products/{product_id}/insights")
def get_saved_product_insights(
    product_id: str,
    session: Session = Depends(get_session),
):
    """
    Read path for generated product insights.

    Flow:
    1. Check Redis cache.
    2. If cache miss, read from Postgres product_insights.
    3. Store Postgres result into Redis.
    4. Return saved insight payload.

    """
    cache_key = product_insight_cache_key(
        product_id=product_id,
        model_version=settings.summary_model_version,
    )
    cached = get_json_cache(cache_key)
    if cached is not None:
        return cached

    insight = get_product_insight(
        session=session,
        product_id=product_id,
        model_version=settings.summary_model_version,
    )

    if insight is None:
        raise HTTPException(
            status_code=404,
            detail="Product insight not found. Create a job first.",
        )

    set_json_cache(
        key=cache_key,
        value=insight.payload,
        ttl_seconds=3600,
    )

    return insight.payload
