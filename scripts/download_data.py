from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT_DIR = Path("data/raw")
SAMPLE_N = 500_000


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and process the data.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="McAuley-Lab/Amazon-Reviews-2023",
        help="The name of the dataset to download from Hugging Face Datasets.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Appliances",
        help="The name of the dataset to download from Hugging Face Datasets.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUT_DIR,
        help="Directory to save the raw data.",
    )

    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="Nums of samples to take",
    )
    return parser.parse_args()


def main():
    args = arg_parser()
    dataset_name = args.dataset_name
    category = args.category
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_n = args.sample_n

    if sample_n is None:
        reviews_out = output_dir / "appliances_demo_reviews_full.csv"
        meta_out = output_dir / "appliances_demo_meta_full.csv"
    else:
        reviews_out = output_dir / "appliances_reviews_sample_{SAMPLE_N}k.csv"
        meta_out = output_dir / "appliances_meta_matched.csv"

    print(f"Loading dataset {dataset_name}...")

    review_dataset = load_dataset(
        dataset_name, f"raw_review_{category}", split="full", streaming=True, trust_remote_code=True
    )

    print(f"Processing and saving data to {output_dir}...")

    review_rows = []
    for i, row in enumerate(review_dataset):
        review_rows.append(
            {
                "product_id": row.get("parent_asin"),
                "rating": row.get("rating"),
                "review_title": row.get("title"),
                "review_text": row.get("text"),
                "timestamp": row.get("timestamp"),
                "helpful_vote": row.get("helpful_vote"),
                "verified_purchase": row.get("verified_purchase"),
                "images": row.get("images"),
            }
        )
        if sample_n is not None and i + 1 >= sample_n:
            break

    reviews_df = pd.DataFrame(review_rows)
    reviews_df = reviews_df.dropna(subset=["product_id", "rating", "review_text"])
    reviews_df = reviews_df[reviews_df["review_text"].astype(str).str.strip() != ""]

    product_ids = set(reviews_df["product_id"].unique())

    meta_ds = load_dataset(
        dataset_name,
        f"raw_meta_{category}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    meta_rows = []
    for row in meta_ds:
        pid = row.get("parent_asin")
        if pid in product_ids:
            meta_rows.append(
                {
                    "product_id": pid,
                    "product_title": row.get("title"),
                    "average_rating": row.get("average_rating"),
                    "rating_number": row.get("rating_number"),
                    "price": row.get("price"),
                    "store": row.get("store"),
                    "main_category": row.get("main_category"),
                    "categories": row.get("categories"),
                    "features": row.get("features"),
                    "description": row.get("description"),
                    "images": row.get("images"),
                    "details": row.get("details"),
                }
            )

    meta_df = pd.DataFrame(meta_rows)

    reviews_df.to_csv(reviews_out, index=False)
    meta_df.to_csv(meta_out, index=False)

    print("Reviews:", reviews_df.shape)
    print("Unique products in reviews:", reviews_df["product_id"].nunique())
    print("Matched meta:", meta_df.shape)
    print("Saved reviews to:", reviews_out)
    print("Saved meta to:", meta_out)


if __name__ == "__main__":
    main()
