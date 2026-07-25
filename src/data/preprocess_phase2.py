from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the data for phase 2")
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("data/interim/reviews_clean.csv"),
        help="Path to cleaned review-level data from Phase 1.",
    )

    parser.add_argument(
        "--output_jsonl",
        type=Path,
        default=Path("data/processed/phase2/product_sentiment_groups_ranked.jsonl"),
        help="Output JSONL path for grouped summary-ready data.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/processed/phase2/product_sentiment_group_stats_ranked.csv"),
        help="Output CSV path for grouped summary stats.",
    )

    parser.add_argument(
        "--max_reviews_per_group",
        type=int,
        default=5,
        help="Maximum selected representative reviews per product-sentiment group.",
    )

    parser.add_argument(
        "--min_reviews_per_group",
        type=int,
        default=2,
        help="Try to keep at least this many reviews per product-sentiment group.",
    )

    parser.add_argument(
        "--min_group_size",
        type=int,
        default=1,
        help="Skip product-sentiment groups smaller than this size.",
    )

    parser.add_argument(
        "--use_embeddings",
        action="store_true",
        help="Use a small embedding model to filter similar reviews.",
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used for similarity filtering.",
    )

    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold above which a review is considered too similar.",
    )

    parser.add_argument(
        "--weight_helpful",
        type=float,
        default=0.40,
        help="Weight for helpful_vote signal in review quality score.",
    )

    parser.add_argument(
        "--weight_verified",
        type=float,
        default=0.30,
        help="Weight for verified_purchase signal in review quality score.",
    )
    parser.add_argument(
        "--weight_length",
        type=float,
        default=0.30,
        help="Weight for length signal in review quality score.",
    )

    parser.add_argument(
        "--ideal_min_len",
        type=int,
        default=20,
        help="Minimum token length considered ideal.",
    )
    parser.add_argument(
        "--ideal_max_len",
        type=int,
        default=100,
        help="Maximum token length considered ideal.",
    )
    parser.add_argument(
        "--hard_max_len",
        type=int,
        default=300,
        help="Very long reviews are increasingly penalized beyond this range.",
    )

    args = parser.parse_args()
    return args


def validate_input_columns(df: pd.DataFrame) -> None:
    required = {
        "review_id",
        "product_id",
        "rating",
        "text",
        "sentiment_label",
        "timestamp",
        "helpful_vote",
        "verified_purchase",
        "text_len",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")


def min_max_normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    min_v = s.min()
    max_v = s.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (s - min_v) / (max_v - min_v)


def compute_length_score(
    token_len: int,
    ideal_min: int,
    ideal_max: int,
    hard_max: int,
) -> float:
    if token_len <= 0:
        return 0.0

    if token_len < ideal_min:
        return max(0.0, token_len / ideal_min)

    if ideal_min <= token_len <= ideal_max:
        return 1.0

    if token_len >= hard_max:
        return 0.2

    # Last case where token_len is between ideal_max and hard_max, I apply a linear decay
    decay = 1.0 - ((token_len - ideal_max) / float(hard_max - ideal_max))
    return max(0.2, decay)


def add_quality_score(
    df: pd.DataFrame,
    weight_helpful: float,
    weight_verified: float,
    weight_length: float,
    ideal_min_len: int,
    ideal_max_len: int,
    hard_max_len: int,
) -> pd.DataFrame:
    df = df.copy()

    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0)
    df["verified_purchase"] = (
        pd.to_numeric(df["verified_purchase"], errors="coerce").fillna(0).astype(int)
    )
    df["text_len"] = pd.to_numeric(df["text_len"], errors="coerce").fillna(0).astype(int)

    # Log-scale helpful votes to avoid skewness
    df["helpful_vote_log"] = np.log1p(df["helpful_vote"].clip(lower=0))
    df["helpful_vote_norm"] = min_max_normalize(df["helpful_vote_log"])

    df["length_score"] = df["text_len"].apply(
        lambda x: compute_length_score(
            token_len=int(x),
            ideal_min=ideal_min_len,
            ideal_max=ideal_max_len,
            hard_max=hard_max_len,
        )
    )

    total_weight = weight_helpful + weight_verified + weight_length

    df["review_quality_score"] = (
        weight_helpful * df["helpful_vote_norm"]
        + weight_verified * df["verified_purchase"].astype(float)
        + weight_length * df["length_score"]
    ) / total_weight

    return df


def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def diverse_selection(
    group_df: pd.DataFrame,
    max_reviews: int,
    min_reviews: int,
    similarity_threshold: float,
    use_embeddings: bool,
    embedder=None,
) -> pd.DataFrame:
    """
    Select high-quality reviews while avoiding near-duplicates.
    Strategy:
    - sort by quality score first
    - greedily keep a candidate if it is not too similar to already selected reviews
    - if too few are selected, top up with the next best remaining reviews
    """
    ranked = group_df.sort_values(
        by=["review_quality_score", "helpful_vote", "verified_purchase", "text_len"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    if len(ranked) <= max_reviews:
        return ranked.copy()

    if not use_embeddings:
        return ranked.head(max_reviews).copy()

    texts = ranked["text"].fillna("").astype(str).tolist()
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    selected_indices: list[int] = []
    skipped_indices: list[int] = []

    for idx in range(len(ranked)):
        if len(selected_indices) >= max_reviews:
            break

        if not selected_indices:
            selected_indices.append(idx)
            continue

        candidate_vec = embeddings[idx]
        sims = [float(np.dot(candidate_vec, embeddings[j])) for j in selected_indices]
        max_sim = max(sims) if sims else 0.0

        if max_sim < similarity_threshold:
            selected_indices.append(idx)
        else:
            skipped_indices.append(idx)

    # If threshold sim is too strict,a fallback with skipped_indices to ensure having at least min_reviews
    if len(selected_indices) < min_reviews:
        for idx in skipped_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= min(min_reviews, max_reviews):
                break

    # if still fewer than max_reviews, fill with remaining best-ranked reviews
    if len(selected_indices) < max_reviews:
        for idx in range(len(ranked)):
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= max_reviews:
                break

    selected_indices = selected_indices[:max_reviews]
    selected = ranked.iloc[selected_indices].copy()
    selected = selected.sort_values(
        by=["review_quality_score", "helpful_vote", "verified_purchase", "text_len"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return selected


def first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    return non_null.iloc[0]


def review_objects_from_df(selected_df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        "review_id",
        "review_title",
        "review_text",
        "text",
        "rating",
        "timestamp",
        "helpful_vote",
        "verified_purchase",
        "text_len",
        "review_quality_score",
        "helpful_vote_norm",
        "length_score",
    ]
    existing_cols = [c for c in cols if c in selected_df.columns]

    records: list[dict[str, Any]] = []

    for _, row in selected_df.iterrows():
        item = {}
        for col in existing_cols:
            val = row[col]

            if pd.isna(val):
                item[col] = None
            elif isinstance(val, (np.integer,)):
                item[col] = int(val)
            elif isinstance(val, (np.floating,)):
                item[col] = int(val)
            else:
                item[col] = val
        records.append(item)
    return records


def create_group_records(
    clean_df: pd.DataFrame,
    max_reviews_per_group: int,
    min_reviews_per_group: int,
    min_group_size: int,
    use_embeddings: bool,
    similarity_threshold: float,
    embedder=None,
) -> list[dict[str, Any]]:

    records: list[dict[str, Any]] = []

    grouped = clean_df.groupby(["product_id", "sentiment_label"], dropna=False)
    temp: dict[str, dict[str, Any]] = {}

    for (product_id, sentiment_label), group in grouped:
        group_size = len(group)
        if group_size < min_group_size:
            continue

        selected_df = diverse_selection(
            group_df=group,
            max_reviews=max_reviews_per_group,
            min_reviews=min_reviews_per_group,
            similarity_threshold=similarity_threshold,
            use_embeddings=use_embeddings,
            embedder=embedder,
        )

        product_key = str(product_id)

        product_entry = temp.setdefault(
            product_key,
            {
                "product_id": product_key,
                "product_title": first_non_null(group["product_title"])
                if "product_title" in group.columns
                else None,
                "average_rating": float(first_non_null(group["average_rating"]))
                if "average_rating" in group.columns
                and first_non_null(group["average_rating"]) is not None
                else None,
                "rating_number": int(first_non_null(group["rating_number"]))
                if "rating_number" in group.columns
                and first_non_null(group["rating_number"]) is not None
                else None,
                "price": float(first_non_null(group["price"]))
                if "price" in group.columns and first_non_null(group["price"]) is not None
                else None,
                "store": first_non_null(group["store"]) if "store" in group.columns else None,
                "main_category": first_non_null(group["main_category"])
                if "main_category" in group.columns
                else None,
                "review_count": 0,
                "avg_rating_from_reviews": 0.0,
                "positive_count": 0,
                "neutral_count": 0,
                "negative_count": 0,
                "positive_selected_count": 0,
                "neutral_selected_count": 0,
                "negative_selected_count": 0,
                "positive_reviews": [],
                "neutral_reviews": [],
                "negative_reviews": [],
                "positive_review_items": [],
                "neutral_review_items": [],
                "negative_review_items": [],
            },
        )
        product_entry["review_count"] += group_size
        product_entry["avg_rating_from_reviews"] += float(group["rating"].sum())
        product_entry[f"{sentiment_label}_count"] = group_size
        product_entry[f"{sentiment_label}_selected_count"] = len(selected_df)

        selected_texts = selected_df["text"].fillna("").astype(str).tolist()
        product_entry[f"{sentiment_label}_reviews"] = selected_texts
        product_entry[f"{sentiment_label}_review_items"] = review_objects_from_df(selected_df)

    for entry in temp.values():
        if entry["review_count"] > 0:
            entry["avg_rating_from_reviews"] = round(
                entry["avg_rating_from_reviews"] / entry["review_count"], 4
            )
            for sentiment in ["positive", "neutral", "negative"]:
                entry[f"{sentiment}_ratio"] = round(
                    entry[f"{sentiment}_count"] / entry["review_count"], 4
                )
        records.append(entry)

    records.sort(key=lambda x: x["review_count"], reverse=True)
    return records


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading cleaned reviews from: {args.input_path}")
    df = pd.read_csv(args.input_path)
    validate_input_columns(df)

    original_rows = len(df)

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()

    df = add_quality_score(
        df=df,
        weight_helpful=args.weight_helpful,
        weight_verified=args.weight_verified,
        weight_length=args.weight_length,
        ideal_min_len=args.ideal_min_len,
        ideal_max_len=args.ideal_max_len,
        hard_max_len=args.hard_max_len,
    )

    embedder = None
    if args.use_embeddings:
        print(f"Loading embedding model: {args.embedding_model}")
        embedder = load_embedder(args.embedding_model)

    group_records = create_group_records(
        clean_df=df,
        max_reviews_per_group=args.max_reviews_per_group,
        min_reviews_per_group=args.min_reviews_per_group,
        min_group_size=args.min_group_size,
        use_embeddings=args.use_embeddings,
        similarity_threshold=args.similarity_threshold,
        embedder=embedder,
    )

    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for record in group_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats_rows = []
    for record in group_records:
        scalar_record = {}
        for key, value in record.items():
            if isinstance(value, list):
                continue
            scalar_record[key] = value
        stats_rows.append(scalar_record)

    pd.DataFrame(stats_rows).to_csv(args.output_csv, index=False)

    print("Phase 2 summary-prep completed.")
    print(f"Input rows: {original_rows:,}")
    print(f"Rows after Phase 2 filtering: {len(df):,}")
    print(f"Products prepared: {len(group_records):,}")
    print(f"Saved JSONL: {args.output_jsonl}")
    print(f"Saved CSV:   {args.output_csv}")


if __name__ == "__main__":
    main()
