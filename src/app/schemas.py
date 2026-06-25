from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProductCard(BaseModel):
    product_id: str
    product_title: str
    store: str | None = None
    main_category: str | None = None
    price: str | None = None
    average_rating: float | None = None
    rating_number: int = 0
    review_count: int = 0
    avg_review_rating: float = 0.0
    verified_review_count: int = 0
    total_helpful_votes: int = 0
    latest_review_ts: int | None = None
    image_url: str | None = None
    categories_list: list[str] = Field(default_factory=list)
    features_text: str | None = None
    description_text: str | None = None


class ProductListResponse(BaseModel):
    total: int
    items: list[ProductCard]


class RepresentativeReview(BaseModel):
    review_id: str
    review_title: str | None = None
    review_text: str
    snippet: str
    sentiment_label: str
    sentiment_confidence: float
    helpful_vote: int
    verified_purchase: bool
    review_quality_score: float
    timestamp: int | None = None
    review_datetime: str | None = None


class SentimentCounts(BaseModel):
    positive: int = 0
    neutral: int = 0
    negative: int = 0


class SentimentRatios(BaseModel):
    positive: float = 0.0
    neutral: float = 0.0
    negative: float = 0.0


class SentimentDistribution(BaseModel):
    counts: SentimentCounts
    ratios: SentimentRatios
    total: int


class InsightsRequest(BaseModel):
    max_reviews: int = 100
    representative_k: int = 5
    regenerate: bool = False


class InsightsResponse(BaseModel):
    product_id: str
    product_info: dict[str, Any]
    sentiment_distribution: SentimentDistribution
    representative_reviews: dict[str, list[RepresentativeReview]]
    prompts: dict[str, str]
    summaries: dict[str, str]
    selected_review_count: int
    total_available_reviews: int
    latency_ms: float
    model_version: str


class InsightJobRequest(BaseModel):
    max_reviews: int = 100
    representative_k: int = 5
    regenerate: bool = False


class InsightJobResponse(BaseModel):
    job_id: str | None = None
    status: str
    product_id: str
    cached: bool = False
    message: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
