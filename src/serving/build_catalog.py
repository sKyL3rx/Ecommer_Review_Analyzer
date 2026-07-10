from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

REVIEWS_PATH = Path("data/raw/appliances_demo_reviews_full.csv")
META_PATH = Path("data/raw/appliances_demo_meta_full.csv")
OUTPUT_PATH = Path("data/serving/appliances_demo_catalog.parquet")


def clean_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text else None


def parse_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

        return [text]

    return [str(value).strip()]


def parse_first_image_url(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        if text.startswith("http://") or text.startswith("https://"):
            return text

        for loader in (json.loads, ast.literal_eval):
            # "["http:..///, "https:...."]"
            try:
                parsed = loader(text)
                return parse_first_image_url(parsed)
            except Exception:
                pass

        return None

    if isinstance(value, list):
        for item in value:
            url = parse_first_image_url(item)
            if url:
                return url
        return None

    if isinstance(value, dict):
        for key in ["large", "hi_res", "thumb", "url", "image_url"]:
            if key in value and value[key]:
                return str(value[key])
        return None

    return None


def flatten_text_list(value: Any) -> str | None:
    items = parse_list(value)
    if not items:
        return None
    return " | ".join(items)


def main() -> None:
    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {REVIEWS_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing file: {META_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reviews_df = pd.read_csv(REVIEWS_PATH)
    meta_df = pd.read_csv(META_PATH)

    reviews_df["product_id"] = reviews_df["product_id"].astype(str).str.strip()
    meta_df["product_id"] = meta_df["product_id"].astype(str).str.strip()

    reviews_df = reviews_df[reviews_df["product_id"] != ""].copy()
    meta_df = meta_df[meta_df["product_id"] != ""].copy()

    reviews_df["verified_purchase"] = reviews_df["verified_purchase"].fillna(False).astype(bool)
    reviews_df["rating"] = pd.to_numeric(reviews_df["rating"], errors="coerce")
    reviews_df["helpful_vote"] = pd.to_numeric(reviews_df["helpful_vote"], errors="coerce").fillna(
        0
    )
    reviews_df["timestamp"] = pd.to_numeric(reviews_df["timestamp"], errors="coerce")

    agg_reviews = reviews_df.groupby("product_id", as_index=False).agg(
        review_count=("product_id", "size"),
        avg_review_rating=("rating", "mean"),
        verified_review_count=("verified_purchase", "sum"),
        total_helpful_votes=("helpful_vote", "sum"),
        latest_review_ts=("timestamp", "max"),
    )

    meta_df = meta_df.drop_duplicates(subset=["product_id"], keep="first").copy()

    catalog_df = meta_df.merge(agg_reviews, on="product_id", how="inner")

    catalog_df["product_title"] = catalog_df["product_title"].apply(clean_text)
    catalog_df["store"] = catalog_df["store"].apply(clean_text)
    catalog_df["main_category"] = catalog_df["main_category"].apply(clean_text)
    catalog_df["price"] = catalog_df["price"].apply(clean_text)

    catalog_df["categories_list"] = catalog_df["categories"].apply(parse_list)
    catalog_df["features_text"] = catalog_df["features"].apply(flatten_text_list)
    catalog_df["description_text"] = catalog_df["description"].apply(flatten_text_list)
    catalog_df["image_url"] = catalog_df["images"].apply(parse_first_image_url)

    catalog_df["average_rating"] = pd.to_numeric(catalog_df["average_rating"], errors="coerce")
    catalog_df["rating_number"] = (
        pd.to_numeric(catalog_df["rating_number"], errors="coerce").fillna(0).astype(int)
    )
    catalog_df["avg_review_rating"] = catalog_df["avg_review_rating"].fillna(0.0)

    catalog_df["search_text"] = (
        (
            catalog_df["product_title"].fillna("")
            + " "
            + catalog_df["store"].fillna("")
            + " "
            + catalog_df["main_category"].fillna("")
            + " "
            + catalog_df["categories_list"].apply(
                lambda xs: " ".join(xs) if isinstance(xs, list) else ""
            )
            + " "
            + catalog_df["features_text"].fillna("")
            + " "
            + catalog_df["description_text"].fillna("")
        )
        .str.strip()
        .str.lower()
    )

    final_df = catalog_df[
        [
            "product_id",
            "product_title",
            "store",
            "main_category",
            "price",
            "average_rating",
            "rating_number",
            "review_count",
            "avg_review_rating",
            "verified_review_count",
            "total_helpful_votes",
            "latest_review_ts",
            "image_url",
            "categories_list",
            "features_text",
            "description_text",
            "search_text",
        ]
    ].copy()

    final_df = final_df.sort_values(
        by=["review_count", "rating_number", "average_rating"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    final_df.to_parquet(OUTPUT_PATH, index=False)

    print("Done.")
    print(f"reviews rows: {len(reviews_df):,}")
    print(f"meta rows: {len(meta_df):,}")
    print(f"catalog rows: {len(final_df):,}")
    print(f"saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
