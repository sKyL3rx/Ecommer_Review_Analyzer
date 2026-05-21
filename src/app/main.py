from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.app.jobs.queue import insight_queue
from src.app.monitoring.metrics import (
    api_request_latency_seconds,
    api_requests_total,
    insight_queue_depth,
    metrics_response,
)
from src.app.routes import health, insights, jobs, products
from src.app.storage.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="Product Review Intelligence API",
    version="0.1.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start

    path = request.url.path
    method = request.method
    status_code = str(response.status_code)

    api_requests_total.labels(
        method=method,
        path=path,
        status_code=status_code,
    ).inc()


    api_request_latency_seconds.labels(
        method=method,
        path=path,
    ).observe(latency)

    return response


@app.get("/metrics")
def metrics():
    insight_queue_depth.set(len(insight_queue))
    return metrics_response()   

app.include_router(health.router)
app.include_router(products.router)
app.include_router(insights.router)
app.include_router(jobs.router)






