from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported input file format: {path}")

REVIEWS_PATH = Path("data/raw/appliances_demo_reviews_full.csv")
OUTPUT_PATH = Path("data/serving/appliances_demo_reviews.parquet")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build serving reviews parquet.")
    parser.add_argument(
        "--reviews_path",
        type=Path,
        default=Path("data/raw/appliances_reviews_sample_50000.csv"),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/serving/appliances_demo_reviews.parquet"),
    )
    return parser.parse_args()

def clean_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text else None


def main() -> None:
    args = parse_args()
    reviews_path = args.reviews_path
    output_path = args.output_path

    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing file: {reviews_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    reviews_df = read_table(reviews_path)

    required_cols = ["product_id", "rating", "review_text"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["review_title"] = (
        df["review_title"].apply(clean_text) if "review_title" in df.columns else None
    )
    df["review_text"] = df["review_text"].apply(clean_text)

    df = df.dropna(subset=["product_id", "review_text"]).copy()
    df = df[df["product_id"] != ""].copy()

    if "helpful_vote" in df.columns:
        df["helpful_vote"] = (
            pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)
        )
    else:
        df["helpful_vote"] = 0

    if "verified_purchase" in df.columns:
        df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)
    else:
        df["verified_purchase"] = False

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["review_datetime"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    else:
        df["timestamp"] = pd.NA
        df["review_datetime"] = pd.NaT

    if "images" in df.columns:
        df["has_review_image"] = df["images"].notna() & (df["images"].astype(str).str.strip() != "")
    else:
        df["has_review_image"] = False

    df["review_char_len"] = df["review_text"].str.len()

    df = df.reset_index(drop=True)
    df["review_id"] = "rvw_" + (df.index + 1).astype(str)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    dedupe_cols = ["product_id", "review_text"]

    if "timestamp" in df.columns:
        dedupe_cols.append("timestamp")

    df = df.drop_duplicates(subset=dedupe_cols, keep="first").copy()

    final_cols = [
        "review_id",
        "product_id",
        "review_title",
        "review_text",
        "review_char_len",
        "timestamp",
        "rating",
        "review_datetime",
        "helpful_vote",
        "verified_purchase",
        "has_review_image",
    ]

    final_df = df[final_cols].copy()

    final_df = final_df.sort_values(
        by=["product_id", "helpful_vote", "timestamp"],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)

    final_df.to_parquet(output_path, index=False)

    print("Done.")
    print(f"reviews rows: {len(final_df):,}")
    print(f"unique products: {final_df['product_id'].nunique():,}")
    print(f"saved to: {output_path}")


if __name__ == "__main__":
    main()
