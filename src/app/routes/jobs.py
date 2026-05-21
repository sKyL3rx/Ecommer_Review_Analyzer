
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from rq.job import Job
from sqlalchemy.orm import Session

from src.app.jobs.queue import redis_conn
from src.app.schemas import JobStatusResponse
from src.app.storage.db import get_session
from src.app.storage.models import InsightJob

router = APIRouter(tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    session: Session = Depends(get_session),
    ) -> JobStatusResponse:
    """
    Poll RQ job status.

    Redis/RQ is the source of truth for live job state.
    Postgres insight_jobs is updated as persistent job metadata.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
        ) from e

    status = job.get_status()
    result = job.result if job.is_finished else None
    error = str(job.exc_info) if job.is_failed else None

    db_job = session.get(InsightJob, job_id)
    if db_job is not None:
        db_job.status = status
        db_job.error = error

        if job.is_finished or job.is_failed:
            db_job.finished_at = datetime.utcnow()

        session.commit()
    
    return JobStatusResponse(
        job_id=job.id,
        status=status,
        result=result,
        error=error,
    )