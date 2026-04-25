from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

import ast
import json


from src.app.schemas import (
    InsightsRequest,
    InsightsResponse,
    ProductCard,
    ProductListResponse,
    InsightJobRequest,
    InsightJobResponse,
    JobStatusResponse,
)

from src.app.services.generate_product_insights import (
    CATALOG_PATH,
    ProductInsightsService,
)

from src.app.clients.sentiment_client import InferenceSentiment
from src.app.clients.summarizer_client import VLLMSummaryGenerator

from rq.job import Job

from src.app.core.config import settings
from src.app.jobs.queue import insight_queue, redis_conn
from src.app.workers.insight_tasks import generate_product_insight_task


app = FastAPI(
    title="Product Review Intelligence API",
    version="0.1.0",
)

sentiment_model = InferenceSentiment()
summarizer_client = VLLMSummaryGenerator(
    base_url=settings.vllm_base_url,
    api_key=settings.vllm_api_key,
    model=settings.app_model_version,
)


service = ProductInsightsService(
    sentiment_predictor =sentiment_model,
    summary_generator = summarizer_client
)

_catalog_df: pd.DataFrame | None = None

def load_catalog() -> pd.DataFrame:
    global _catalog_df

    if _catalog_df is None:
        if not CATALOG_PATH.exists():
            raise FileNotFoundError(f"Missing catalog file: {CATALOG_PATH}")

        df = pd.read_parquet(CATALOG_PATH)
        df["product_id"] = df["product_id"].astype(str).str.strip()

        for col in ["product_title", "store", "main_category", "price", "features_text", "description_text", "search_text"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        if "categories_list" in df.columns:
            df["categories_list"] = df["categories_list"].apply(
                lambda x: x if isinstance(x, list) else []
            )

        _catalog_df = df.reset_index(drop=True)

    return _catalog_df

def normalize_image_url(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        if text.startswith("http://") or text.startswith("https://"):
            return text

        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                return normalize_image_url(parsed)
            except Exception:
                pass

        return None

    if isinstance(value, list):
        for item in value:
            url = normalize_image_url(item)
            if url:
                return url
        return None

    if isinstance(value, dict):
        for key in ["large", "hi_res", "thumb", "url", "image_url"]:
            if key in value and value[key]:
                return str(value[key]).strip()
        return None

    return None


def catalog_row_to_product_card(row: pd.Series) -> ProductCard:
    return ProductCard(
        product_id=str(row.get("product_id", "")).strip(),
        product_title=str(row.get("product_title", "")).strip(),
        store=str(row.get("store", "")).strip() or None,
        main_category=str(row.get("main_category", "")).strip() or None,
        price=str(row.get("price", "")).strip() or None,
        average_rating=float(row["average_rating"]) if pd.notna(row.get("average_rating")) else None,
        rating_number=int(row.get("rating_number", 0) or 0),
        review_count=int(row.get("review_count", 0) or 0),
        avg_review_rating=float(row.get("avg_review_rating", 0.0) or 0.0),
        verified_review_count=int(row.get("verified_review_count", 0) or 0),
        total_helpful_votes=int(row.get("total_helpful_votes", 0) or 0),
        latest_review_ts=int(row["latest_review_ts"]) if pd.notna(row.get("latest_review_ts")) else None,
        image_url=normalize_image_url(row.get("image_url")),
        categories_list=row.get("categories_list", []) if isinstance(row.get("categories_list"), list) else [],
        features_text=str(row.get("features_text", "")).strip() or None,
        description_text=str(row.get("description_text", "")).strip() or None,
    )

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/products", response_model=ProductListResponse)
def list_products(
    query: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ProductListResponse:
    
    try:
        df = load_catalog().copy()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    
    if query:
        q = query.strip().lower()
        if q:
            if "search_text" in df.columns:
                mask = df["search_text"].str.contains(q, regex=False, na=False)
            else:
                mask = df["product_title"].str.lower().str.contains(q, regex=False, na=False)
            df = df[mask].copy()
        
    total = len(df)
    page_df = df.iloc[offset : offset + limit]

    items = [catalog_row_to_product_card(row) for _, row in page_df.iterrows()]
    return ProductListResponse(total=total, items=items)

@app.get("/products/{product_id}", response_model=ProductCard)
def get_product(product_id: str) -> ProductCard:
    try:
        df = load_catalog()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    matched = df[df["product_id"] == product_id]
    if matched.empty:
        raise HTTPException(status_code=404, detail=f"Product not found: {product_id}")

    return catalog_row_to_product_card(matched.iloc[0])

@app.post("/products/{product_id}/insights", response_model=InsightsResponse)
def generate_product_insights(
    product_id: str,
    payload: InsightsRequest,
) -> InsightsResponse:
    try:
        result = service.generate(
            product_id=product_id,
            max_reviews=payload.max_reviews,
            representative_k=payload.representative_k,
            regenerate=payload.regenerate,
        )
        return InsightsResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    
@app.post(
    "/products/{product_id}/insights/jobs",
    response_model=InsightJobResponse,
)
def create_insight_job(
    product_id: str,
    payload: InsightJobRequest,
    ) -> InsightJobResponse:
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
    
    return InsightJobResponse(
        job_id=job.id,
        status=job.get_status(),
        product_id=product_id,
    )

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    try:
        job = Job.fetch(job_id, connection = redis_conn)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") 
    
    result = job.result if job.is_finished else None
    error = str(job.exc_info) if job.is_failed else None

    return JobStatusResponse(
        job_id=job.id,
        status=job.get_status(),
        result=result,
        error=error,
    )