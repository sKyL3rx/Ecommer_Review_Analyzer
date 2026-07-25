import argparse
import hashlib
from typing import Any

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from src.app.storage.db import SessionLocal, init_db
from src.app.storage.models import Product, Review


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def make_review_id(product_id: str, text: str, idx: int) -> str:
    raw = f"{product_id}|{idx}|{text[:100]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_products(catalog_df: pd.DataFrame, limit: int | None) -> set[str]:
    if limit:
        catalog_df = catalog_df.head(limit)

    product_ids: set[str] = set()

    with SessionLocal() as session:
        for _, row in catalog_df.iterrows():
            product_id = str(row.get("product_id", "")).strip()
            if not product_id:
                continue

            categories = row.get("categories_list", None)
            if categories is not None and pd.isna(categories) is False:
                if not isinstance(categories, list):
                    categories = list(categories) if hasattr(categories, "__iter__") else None
            else:
                categories = None

            stmt = insert(Product).values(
                product_id=product_id,
                product_title=str(row.get("product_title") or ""),
                store=str(row.get("store") or ""),
                main_category=str(row.get("main_category") or ""),
                price=str(row.get("price") or ""),
                average_rating=safe_float(row.get("average_rating")),
                rating_number=safe_int(row.get("rating_number")),
                review_count=safe_int(row.get("review_count")),
                avg_review_rating=safe_float(row.get("avg_review_rating")),
                verified_review_count=safe_int(row.get("verified_review_count")),
                total_helpful_votes=safe_int(row.get("total_helpful_votes")),
                latest_review_ts=safe_int(row.get("latest_review_ts")),
                image_url=str(row.get("image_url") or ""),
                categories_list=categories,
                features_text=str(row.get("features_text") or ""),
                description_text=str(row.get("description_text") or ""),
                search_text=str(row.get("search_text") or ""),
            )

            session.execute(stmt)
            product_ids.add(product_id)

        session.commit()

    return product_ids


def load_reviews(reviews_df: pd.DataFrame, product_ids: set[str] | None) -> int:
    inserted = 0

    with SessionLocal() as session:
        for idx, row in reviews_df.iterrows():
            product_id = str(row.get("product_id", "")).strip()
            if not product_id:
                continue

            if product_ids is not None and product_id not in product_ids:
                continue

            review_text = str(row.get("review_text") or "")
            if not review_text.strip():
                continue

            review_id = str(
                row.get("review_id") or make_review_id(product_id, review_text, int(idx))
            )

            stmt = insert(Review).values(
                review_id=review_id,
                product_id=product_id,
                review_title=str(row.get("review_title") or ""),
                review_text=review_text,
                review_char_len=safe_int(row.get("review_char_len")),
                timestamp=safe_int(row.get("timestamp")),
                review_datetime=str(row.get("review_datetime") or ""),
                helpful_vote=safe_int(row.get("helpful_vote")),
                verified_purchase=bool(row.get("verified_purchase"))
                if pd.notna(row.get("verified_purchase"))
                else None,
                has_review_image=bool(row.get("has_review_image"))
                if pd.notna(row.get("has_review_image"))
                else None,
                rating=safe_float(row.get("rating")),
                predicted_sentiment=(
                    str(row.get("predicted_sentiment")).strip().lower()
                    if "predicted_sentiment" in row and pd.notna(row.get("predicted_sentiment"))
                    else None
                ),
                sentiment_confidence=safe_float(row.get("sentiment_confidence")),
            )

            stmt = stmt.on_conflict_do_nothing(index_elements=["review_id"])
            session.execute(stmt)
            inserted += 1

            if inserted % 1000 == 0:
                session.commit()
                print(f"Loaded {inserted} reviews...")

        session.commit()

    return inserted


def main(catalog_path: str, reviews_path: str, limit: int | None) -> None:
    init_db()

    print(f"Reading catalog: {catalog_path}")
    catalog_df = pd.read_parquet(catalog_path)

    print(f"Reading reviews: {reviews_path}")
    reviews_df = pd.read_parquet(reviews_path)

    print("Loading products...")
    product_ids = load_products(catalog_df, limit=limit)
    print(f"Loaded/upserted {len(product_ids)} products")

    print("Loading reviews...")
    inserted_reviews = load_reviews(reviews_df, product_ids=product_ids)
    print(f"Loaded {inserted_reviews} reviews")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog_path", required=True)
    parser.add_argument("--reviews_path", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    main(
        catalog_path=args.catalog_path,
        reviews_path=args.reviews_path,
        limit=args.limit,
    )
