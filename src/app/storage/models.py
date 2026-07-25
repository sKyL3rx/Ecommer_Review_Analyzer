from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.app.storage.db import Base


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String, primary_key=True)

    product_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    store: Mapped[str | None] = mapped_column(String, nullable=True)
    main_category: Mapped[str | None] = mapped_column(String, nullable=True)
    price: Mapped[str | None] = mapped_column(String, nullable=True)

    average_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    rating_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    review_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_review_rating: Mapped[float | None] = mapped_column(Float, nullable=True)

    verified_review_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_helpful_votes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latest_review_ts: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    categories_list: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    features_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    description_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    search_text: Mapped[str | None] = mapped_column(Text, nullable=True)


class Review(Base):
    __tablename__ = "reviews"

    review_id: Mapped[str] = mapped_column(String, primary_key=True)
    product_id: Mapped[str] = mapped_column(String, index=True)

    review_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_char_len: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)

    timestamp: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    review_datetime: Mapped[str | None] = mapped_column(String, nullable=True)

    helpful_vote: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verified_purchase: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    has_review_image: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    predicted_sentiment: Mapped[str | None] = mapped_column(String, nullable=True)
    sentiment_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class ProductInsight(Base):
    __tablename__ = "product_insights"

    product_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_version: Mapped[str] = mapped_column(String, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class InsightJob(Base):
    __tablename__ = "insight_jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    product_id: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
