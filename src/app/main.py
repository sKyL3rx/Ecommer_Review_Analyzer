from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI

from src.app.core.config import settings
from src.app.monitoring.metrics import (
    api_request_latency_seconds,
    api_requests_total,
)
from src.app.routes import health, insights, jobs, products
from src.app.routes import metrics as metrics_routes
from src.app.storage.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_create_tables:
        init_db()
    yield


app = FastAPI(
    title="Product Review Intelligence API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def record_api_metrics(request, call_next):
    start = perf_counter()
    method = request.method
    status_code = "500"

    path = request.url.path

    try:
        response = await call_next(request)
        status_code = str(response.status_code)

        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)

        return response

    finally:
        duration_seconds = perf_counter() - start

        api_requests_total.labels(
            method=method,
            path=path,
            status_code=status_code,
        ).inc()

        api_request_latency_seconds.labels(
            method=method,
            path=path,
            status_code=status_code,
        ).observe(duration_seconds)


app.include_router(health.router)
app.include_router(products.router)
app.include_router(insights.router)
app.include_router(jobs.router)
app.include_router(metrics_routes.router)
