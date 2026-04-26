from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


CATALOG_PATH = Path("data/serving/appliances_demo_catalog.parquet")
REVIEWS_PATH = Path("data/serving/appliances_demo_reviews.parquet")
CACHE_DIR = Path("data/cache/product_insights")
DEFAULT_MODEL_VERSION = "summary-sft-candidate"

SentimentPredictor = Callable[[list[str]], list[dict[str, Any]]]
SummaryGenerator = Callable[[str, dict[str, Any]], str | dict[str, Any]]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def snippet(text: str, max_len: int = 220) -> str:
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def token_count(text: str) -> int:
    if not text:
        return 0
    return len(str(text).split())


def min_max_normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    min_v = s.min()
    max_v = s.max()

    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    return (s - min_v) / (max_v - min_v)


def compute_length_score(
    token_len: int,
    ideal_min: int = 20,
    ideal_max: int = 100,
    hard_max: int = 300,
) -> float:
    if token_len <= 0:
        return 0.0

    if token_len < ideal_min:
        return max(0.0, token_len / ideal_min)

    if ideal_min <= token_len <= ideal_max:
        return 1.0

    if token_len >= hard_max:
        return 0.2

    decay = 1.0 - ((token_len - ideal_max) / float(hard_max - ideal_max))
    return max(0.2, decay)


def add_quality_score(
    df: pd.DataFrame,
    weight_helpful: float = 0.40,
    weight_verified: float = 0.30,
    weight_length: float = 0.30,
    ideal_min_len: int = 20,
    ideal_max_len: int = 100,
    hard_max_len: int = 300,
) -> pd.DataFrame:
    out = df.copy()

    required_cols = ["helpful_vote", "verified_purchase", "review_char_len", "review_text"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for quality scoring: {missing}")

    out["review_token_count"] = out["review_text"].fillna("").astype(str).apply(token_count)
    out["helpful_vote_log"] = np.log1p(out["helpful_vote"].clip(lower=0))
    out["helpful_vote_norm"] = min_max_normalize(out["helpful_vote_log"])

    out["length_score"] = out["review_token_count"].apply(
        lambda x: compute_length_score(
            token_len=int(x),
            ideal_min=ideal_min_len,
            ideal_max=ideal_max_len,
            hard_max=hard_max_len,
        )
    )

    total_weight = weight_helpful + weight_verified + weight_length
    out["review_quality_score"] = (
        weight_helpful * out["helpful_vote_norm"]
        + weight_verified * out["verified_purchase"].astype(float)
        + weight_length * out["length_score"]
    ) / total_weight

    return out


def all_positive_sentiment_predictor(texts: list[str]) -> list[dict[str, Any]]:

    return [
        {
            "sentiment_label": "positive",
            "sentiment_confidence": 1.0,
        }
        for _ in texts
    ]


def fallback_summary_generator(prompt: str, context: dict[str, Any]) -> str:
    sentiment_label = context.get("sentiment_label", "unknown")
    count = context.get("review_count", 0)
    return f"Placeholder {sentiment_label} summary generated from {count} representative reviews."


def ensure_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)


class ProductInsightsService:
    def __init__(
        self,
        catalog_path: Path = CATALOG_PATH,
        reviews_path: Path = REVIEWS_PATH,
        cache_dir: Path = CACHE_DIR,
        sentiment_predictor: SentimentPredictor | None = None,
        summary_generator: SummaryGenerator | None = None,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> None:
        self.catalog_path = catalog_path
        self.reviews_path = reviews_path
        self.cache_dir = cache_dir
        self.sentiment_predictor = sentiment_predictor or all_positive_sentiment_predictor
        self.summary_generator = summary_generator or fallback_summary_generator
        self.model_version = model_version

        self._catalog_df: pd.DataFrame | None = None
        self._reviews_df: pd.DataFrame | None = None

    def _load_catalog(self) -> pd.DataFrame:
        if self._catalog_df is None:
            if not self.catalog_path.exists():
                raise FileNotFoundError(f"Missing catalog file: {self.catalog_path}")

            df = pd.read_parquet(self.catalog_path)
            df["product_id"] = df["product_id"].astype(str).str.strip()
            self._catalog_df = df

        return self._catalog_df

    def _load_reviews(self) -> pd.DataFrame:
        if self._reviews_df is None:
            if not self.reviews_path.exists():
                raise FileNotFoundError(f"Missing reviews file: {self.reviews_path}")

            df = pd.read_parquet(self.reviews_path)
            df["product_id"] = df["product_id"].astype(str).str.strip()
            self._reviews_df = df.reset_index(drop=True)

        return self._reviews_df

    def _cache_path(self, product_id: str) -> Path:
        ensure_cache_dir(self.cache_dir)
        safe_product_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in product_id)
        return self.cache_dir / f"{safe_product_id}.json"

    def _load_product_info(self, product_id: str) -> dict[str, Any]:
        catalog_df = self._load_catalog()
        matched = catalog_df[catalog_df["product_id"] == product_id]

        if matched.empty:
            raise ValueError(f"Product not found in catalog: {product_id}")

        row = matched.iloc[0].to_dict()

        return {
            "product_id": clean_text(row.get("product_id")),
            "product_title": clean_text(row.get("product_title")),
            "store": clean_text(row.get("store")),
            "main_category": clean_text(row.get("main_category")),
            "price": clean_text(row.get("price")),
            "average_rating": float(row.get("average_rating")) if pd.notna(row.get("average_rating")) else None,
            "rating_number": int(row.get("rating_number", 0) or 0),
            "review_count": int(row.get("review_count", 0) or 0),
            "avg_review_rating": float(row.get("avg_review_rating", 0.0) or 0.0),
            "verified_review_count": int(row.get("verified_review_count", 0) or 0),
            "total_helpful_votes": int(row.get("total_helpful_votes", 0) or 0),
            "latest_review_ts": int(row.get("latest_review_ts")) if pd.notna(row.get("latest_review_ts")) else None,
            "image_url": clean_text(row.get("image_url")),
            "categories_list": row.get("categories_list") if isinstance(row.get("categories_list"), list) else [],
            "features_text": clean_text(row.get("features_text")),
            "description_text": clean_text(row.get("description_text")),
            "search_text": clean_text(row.get("search_text")),
        }

    def _load_product_reviews(self, product_id: str) -> pd.DataFrame:
        reviews_df = self._load_reviews()
        product_reviews = reviews_df[reviews_df["product_id"] == product_id].copy()

        if product_reviews.empty:
            raise ValueError(f"No reviews found for product_id={product_id}")

        return product_reviews.reset_index(drop=True)

    def _select_reviews_for_inference(
        self,
        reviews_df: pd.DataFrame,
        max_reviews: int = 100,
    ) -> pd.DataFrame:
        """
        Bước 1: từ tất cả reviews của product, lấy ra một tập nhỏ hơn để phân tích.
        """
        scored = add_quality_score(reviews_df)

        scored = scored.sort_values(
            by=[
                "review_quality_score",
                "helpful_vote",
                "verified_purchase",
                "review_char_len",
                "timestamp",
            ],
            ascending=[False, False, False, False, False],
            kind="stable",
        ).reset_index(drop=True)

        if len(scored) <= max_reviews:
            return scored.copy()

        return scored.head(max_reviews).copy()

    def _predict_sentiment(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        texts = reviews_df["review_text"].fillna("").astype(str).tolist()
        predictions = self.sentiment_predictor(texts)

        if len(predictions) != len(reviews_df):
            raise ValueError("Sentiment predictor returned wrong number of predictions.")

        labels: list[str] = []
        confidences: list[float] = []

        for pred in predictions:
            label = str(pred.get("sentiment_label", "positive")).strip().lower()
            if label not in {"positive", "neutral", "negative"}:
                label = "positive"

            conf = float(pred.get("sentiment_confidence", 1.0))
            conf = max(0.0, min(1.0, conf))

            labels.append(label)
            confidences.append(conf)

        out = reviews_df.copy()
        out["sentiment_label"] = labels
        out["sentiment_confidence"] = confidences
        return out

    def _build_sentiment_distribution(self, reviews_df: pd.DataFrame) -> dict[str, Any]:
        total = len(reviews_df)
        counts_raw = reviews_df["sentiment_label"].value_counts().to_dict()

        counts = {
            "positive": int(counts_raw.get("positive", 0)),
            "neutral": int(counts_raw.get("neutral", 0)),
            "negative": int(counts_raw.get("negative", 0)),
        }

        if total == 0:
            ratios = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        else:
            ratios = {
                "positive": round(counts["positive"] / total, 4),
                "neutral": round(counts["neutral"] / total, 4),
                "negative": round(counts["negative"] / total, 4),
            }

        return {
            "counts": counts,
            "ratios": ratios,
            "total": int(total),
        }

    def _pick_representative_reviews(
        self,
        reviews_df: pd.DataFrame,
        sentiment_label: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        subset = reviews_df[reviews_df["sentiment_label"] == sentiment_label].copy()

        if subset.empty:
            return []

        subset = subset.sort_values(
            by=["review_quality_score", "sentiment_confidence", "helpful_vote", "timestamp"],
            ascending=[False, False, False, False],
            kind="stable",
        ).head(top_k)

        rows: list[dict[str, Any]] = []
        for _, row in subset.iterrows():
            review_datetime = row.get("review_datetime")
            if pd.notna(review_datetime):
                review_datetime = str(review_datetime)
            else:
                review_datetime = None

            rows.append(
                {
                    "review_id": clean_text(row.get("review_id")),
                    "review_title": clean_text(row.get("review_title")),
                    "review_text": clean_text(row.get("review_text")),
                    "snippet": snippet(row.get("review_text", "")),
                    "sentiment_label": clean_text(row.get("sentiment_label")),
                    "sentiment_confidence": round(float(row.get("sentiment_confidence", 0.0)), 4),
                    "helpful_vote": int(row.get("helpful_vote", 0) or 0),
                    "verified_purchase": bool(row.get("verified_purchase", False)),
                    "review_quality_score": round(float(row.get("review_quality_score", 0.0)), 4),
                    "timestamp": int(row.get("timestamp")) if pd.notna(row.get("timestamp")) else None,
                    "review_datetime": review_datetime,
                }
            )

        return rows

    def _build_sentiment_prompt(
        self,
        product_info: dict[str, Any],
        sentiment_label: str,
        representative_reviews: list[dict[str, Any]],
    ) -> str:
        sentiment_label = sentiment_label.lower().strip()

        review_lines = []
        for idx, item in enumerate(representative_reviews, start=1):
            review_lines.append(f"{idx}. {clean_text(item.get('review_text'))}")

        reviews_block = "\n".join(review_lines) if review_lines else "1. No reviews available."

        return f"""Summarize the main {sentiment_label} customer feedback for this product.

Product title: {product_info.get("product_title", "")}
Category: {product_info.get("main_category", "")}

Reviews:
{reviews_block}

Requirements:
- Mention only recurring themes supported by the reviews.
- Do not invent details.
- Prefer concrete themes over generic praise/complaints.
- Write one concise summary in 1-3 sentences.""".strip()

    def _generate_sentiment_summary(
        self,
        prompt: str,
        sentiment_label: str,
        representative_reviews: list[dict[str, Any]],
    ) -> str:
        if not representative_reviews:
            return ""

        result = self.summary_generator(
            prompt,
            {
                "sentiment_label": sentiment_label,
                "review_count": len(representative_reviews),
            },
        )

        if isinstance(result, dict):
            return clean_text(result.get("summary"))

        return clean_text(result)

    def generate(
        self,
        product_id: str,
        max_reviews: int = 100,
        representative_k: int = 5,
        regenerate: bool = False,
        product_info: dict[str, Any] | None = None,
        reviews_df: pd.DataFrame | None = None,
        use_file_cache: bool = True,
        ) -> dict[str, Any]:
        product_id = clean_text(product_id)
        if not product_id:
            raise ValueError("product_id must not be empty.")

        cache_path = self._cache_path(product_id)

        if use_file_cache and cache_path.exists() and not regenerate:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        start = time.perf_counter()

        if product_info is None:
            product_info = self._load_product_info(product_id)
        if reviews_df is None:
            all_reviews_df = self._load_product_reviews(product_id)
        else:
            all_reviews_df = reviews_df.copy()

            if "product_id" in all_reviews_df.columns:
                all_reviews_df["product_id"] = (
                    all_reviews_df["product_id"].astype(str).str.strip()
                    )
                all_reviews_df = all_reviews_df[
                    all_reviews_df["product_id"] == product_id
                    ].copy()

            if all_reviews_df.empty:
                raise ValueError(f"No reviews found for product_id={product_id}")

            all_reviews_df = all_reviews_df.reset_index(drop=True)

        # Step 1: choose top N reviews for this product
        selected_reviews_df = self._select_reviews_for_inference(
        all_reviews_df,
        max_reviews=max_reviews,
        )

        # Step 2: predict sentiment
        scored_reviews_df = self._predict_sentiment(selected_reviews_df)

        # Step 3: build sentiment distribution
        sentiment_distribution = self._build_sentiment_distribution(scored_reviews_df)

        # Step 4: pick representative reviews for each sentiment
        representative_reviews = {
            "positive": self._pick_representative_reviews(
            scored_reviews_df,
            sentiment_label="positive",
            top_k=representative_k,
        ),
        "neutral": self._pick_representative_reviews(
            scored_reviews_df,
            sentiment_label="neutral",
            top_k=representative_k,
        ),
        "negative": self._pick_representative_reviews(
            scored_reviews_df,
            sentiment_label="negative",
            top_k=representative_k,
        ),
        }

        # Step 5: build prompts safely
        positive_prompt = ""
        neutral_prompt = ""
        negative_prompt = ""

        if representative_reviews["positive"]:
            positive_prompt = self._build_sentiment_prompt(
            product_info=product_info,
            sentiment_label="positive",
            representative_reviews=representative_reviews["positive"],
        )

        if representative_reviews["neutral"]:
            neutral_prompt = self._build_sentiment_prompt(
            product_info=product_info,
            sentiment_label="neutral",
            representative_reviews=representative_reviews["neutral"],
        )

        if representative_reviews["negative"]:
            negative_prompt = self._build_sentiment_prompt(
            product_info=product_info,
            sentiment_label="negative",
            representative_reviews=representative_reviews["negative"],
        )

        # Step 6: summarize safely
        positive_summary = ""
        neutral_summary = ""
        negative_summary = ""

        if representative_reviews["positive"]:
            positive_summary = self._generate_sentiment_summary(
            positive_prompt,
            sentiment_label="positive",
            representative_reviews=representative_reviews["positive"],
        )

        if representative_reviews["neutral"]:
            neutral_summary = self._generate_sentiment_summary(
            neutral_prompt,
            sentiment_label="neutral",
            representative_reviews=representative_reviews["neutral"],
            )

        if representative_reviews["negative"]:
            negative_summary = self._generate_sentiment_summary(
            negative_prompt,
            sentiment_label="negative",
            representative_reviews=representative_reviews["negative"],
            )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        response = {
            "product_id": product_info["product_id"],
            "product_info": product_info,
            "sentiment_distribution": sentiment_distribution,
            "representative_reviews": representative_reviews,
            "prompts": {
                "positive": positive_prompt,
                "neutral": neutral_prompt,
                "negative": negative_prompt,
            },
            "summaries": {
                "positive": positive_summary,
                "neutral": neutral_summary,
                "negative": negative_summary,
            },
            "selected_review_count": int(len(scored_reviews_df)),
            "total_available_reviews": int(len(all_reviews_df)),
            "latency_ms": latency_ms,
            "model_version": self.model_version,
        }

        if use_file_cache:
            ensure_cache_dir(self.cache_dir)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(response, f, ensure_ascii=False, indent=2)

        return response

