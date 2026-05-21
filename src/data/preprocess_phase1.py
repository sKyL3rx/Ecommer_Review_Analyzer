from __future__ import annotations

import argparse
import hashlib
import html
import re
import string
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim/")
PROCESSED_DIR_PHASE_1 = Path("data/processed/phase1")


DEFAULT_REVIEW_CANDIDATES = [
    RAW_DIR / "appliances_reviews_sample_500000k.csv",
]
DEFAULT_META_PATH = RAW_DIR / "appliances_meta_matched.csv"

REQUIRED_REVIEW_COLUMNS = {
    "product_id",
    "rating",
    "review_title",
    "review_text",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
}


#REGEX patterns for cleaning text (e.g. removing HTML tags)
HTML_TAG_RE = re.compile(r"<[^>]+>")

PUNCT_TO_REMOVE = string.punctuation.replace("?", "").replace("!", "")
PUNCT_TRANSLATION_TABLE = str.maketrans("", "", PUNCT_TO_REMOVE)


def parser_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description= "Preprocess the raw data for PHASE 1" \
    "(1) review-level sentiment classification")

    parser.add_argument(
        "--input_reviews",
        type=Path,
        default=None,
        help=(
            "Path to raw review CSV"
            "data/raw/appliances_reviews_sample_500000k.csv"
        ),
    )

    parser.add_argument(
        "--input_meta",
        type=Path,
        default=DEFAULT_META_PATH,
        help="Optional metadata CSV produced by the download script.",
    )
    parser.add_argument(
        "--interim_dir",
        type=Path,
        default=INTERIM_DIR,
        help="Directory to save interim artifacts (cleaned / grouped review jsonl / stats csv etc).",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=PROCESSED_DIR_PHASE_1,
        help="Directory to save train/val/test splits for PHASE 1.",
    )

    parser.add_argument(
    "--english_only",
    action="store_true",
    help="Keep only English reviews.",
    )

    parser.add_argument(
    "--lang_conf_threshold",
    type=float,
    default=0.90,
    help="Minimum confidence for English language filtering.",
    )

    parser.add_argument(
    "--n_splits",
    type=int,
    default=5,
    help="Number of folds for Stratified K-Fold on the non-test split.",
    )

    parser.add_argument(
    "--save_kfold",
    action="store_true",
    help="Save Stratified K-Fold train/val splits.",
    )
    


    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Fraction of data reserved for the test split.",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help=(
            "Fraction of the full dataset reserved for the validation split. "
            "The actual second split is adjusted after removing the test set."
        ),
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed used for splitting.",
    )


    parser.add_argument(
        "--min_review_len",
        type=int,
        default=3,
        help="Minimum number of tokens required in the cleaned text.",
    )



    return parser.parse_args()

def validate_review_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_REVIEW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Input review CSV is missing required columns: "
            f"{sorted(missing)}"
        )
    
def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value)

    # decode HTML entities
    text = html.unescape(text)

    # remove HTML tags
    text = re.sub(HTML_TAG_RE, " ", text)

    # remove all punctuation except ? and !
    text = text.translate(PUNCT_TRANSLATION_TABLE)

    # put spaces around ? and !
    text = re.sub(r"([!?])", r" \1 ", text)

    # normalize whitespace
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()

### LANGUAGE DETECTION UTILITIES
def init_language_identifier():
    from py3langid.langid import MODEL_FILE, LanguageIdentifier
    identifier = LanguageIdentifier.from_pickled_model(
        MODEL_FILE,
        norm_probs=True,
    )
    
    identifier.set_languages(["en", "es", "fr", "de", "it", "pt"])
    return identifier

def detect_language_with_conf(text: str, identifier) -> tuple[str, float]:
    text = str(text).strip()
    if not text:
        return "unknown", 0.0
    return identifier.classify(text)

### =======================================
def parse_verified_purchase(value: Any) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().lower()
    return int(normalized in {"1", "true", "yes", "y"})

def build_model_text(review_title: Any, review_text: Any) -> str:
    title = clean_text(review_title)
    body = clean_text(review_text)
    merged = f"{title} {body}" if title else body
    merged = merged.lower()
    merged = re.sub(r"\s+", " ", merged)
    return merged.strip()

def token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())

def map_sentiment(rating: Any) -> str:
    rating_value = float(rating)
    if rating_value >= 4.0:
        return "positive"
    if rating_value == 3.0:
        return "neutral"
    return "negative"


def build_review_id(row: pd.Series) -> str:
    base = "||".join(
        [
            str(row.get("product_id", "")),
            str(row.get("rating", "")),
            str(row.get("timestamp", "")),
            str(row.get("review_text", "")),
        ]
    )
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def join_metadata(clean_df: pd.DataFrame, meta_path: Path)  -> pd.DataFrame:
    if not meta_path.exists():
        return clean_df
    
    meta_df = pd.read_csv(meta_path)
    if meta_df.empty or "product_id" not in meta_df.columns:
        return clean_df
    
    meta_columns = [
        col
        for col in [
            "product_id",
            "product_title",
            "average_rating",
            "rating_number",
            "price",
            "store",
            "main_category",
        ]
        if col in meta_df.columns
    ]

    meta_df = meta_df[meta_columns].drop_duplicates(subset=["product_id"])

    return clean_df.merge(meta_df, on="product_id", how="left")


### DATA SPLITTING UTILITIES

def split_train_val(
    train_val_df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_fraction_of_train_val = val_size / (1.0 - test_size)
    if not 0.0 < val_fraction_of_train_val < 1.0:
        raise ValueError(
            "Invalid split config. Make sure 0 < val_size < 1 - test_size."
        )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction_of_train_val,
        stratify=train_val_df["sentiment_label"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def split_dataset_with_holdout(
    clean_df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        clean_df,
        test_size=test_size,
        stratify=clean_df["sentiment_label"],
        random_state=random_state,
    )
    return (
        train_val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

def save_kfold_splits(
    train_val_df: pd.DataFrame,
    processed_dir: Path,
    n_splits: int,
    random_state: int,
) -> None:
    folds_dir = processed_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    X = train_val_df.index.to_numpy()
    y = train_val_df["sentiment_label"].to_numpy()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_dir = folds_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        fold_train_df.to_csv(fold_dir / "train.csv", index=False)
        fold_val_df.to_csv(fold_dir / "val.csv", index=False)

def split_dataset(
    clean_df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train_val_df, test_df = train_test_split(
        clean_df,
        test_size=test_size,
        stratify=clean_df["sentiment_label"],
        random_state=random_state,
    )


    val_fraction_of_train_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction_of_train_val,
        stratify=train_val_df["sentiment_label"],
        random_state=random_state,
    )


    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

def main() -> None:
    args = parser_args()
    input_reviews_path = args.input_reviews
    input_meta_path = args.input_meta
    interim_dir = args.interim_dir
    processed_dir = args.processed_dir


    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw reviews from: {input_reviews_path}")
    raw_df = pd.read_csv(input_reviews_path)
    print("Validating review columns...")
    validate_review_columns(raw_df)

    original_rows = len(raw_df)

    df = raw_df.copy()

    df = df.dropna(subset=["product_id", "rating", "review_text"])

    df["review_title"] = df["review_title"].fillna("").map(clean_text)
    df["review_text"] = df["review_text"].map(clean_text)
    df = df[df["review_text"] != ""]

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)
    df["verified_purchase"] = df["verified_purchase"].map(parse_verified_purchase).astype(int)

    df = df.dropna(subset=["rating"])

    df["text"] = df.apply(
        lambda row: build_model_text(row["review_title"], row["review_text"]), axis=1
    )

    df["text_len"] = df["text"].map(token_count)

    df = df[df["text_len"] >= args.min_review_len]


    if args.english_only:
        identifier = init_language_identifier()
        lang_preds = df["text"].map(lambda x: detect_language_with_conf(x, identifier))
        df["detected_lang"] = lang_preds.map(lambda x: x[0])
        df["lang_conf"] = lang_preds.map(lambda x: x[1])

        before_lang = len(df)
        df = df[
            (df["detected_lang"] == "en")
            & (df["lang_conf"] >= args.lang_conf_threshold)
        ].copy()
        print(f"Kept English reviews: {len(df):,}/{before_lang:,}")

    df = df.drop_duplicates(subset=["product_id", "rating", "review_text", "timestamp"]).reset_index(drop=True)


    df["sentiment_label"] = df["rating"].map(map_sentiment)

    df["review_id"] = df.apply(build_review_id, axis=1)

    base_cols = [
        "review_id",
        "product_id",
        "rating",
        "review_title",
        "review_text",
        "text",
        "sentiment_label",
        "timestamp",
        "helpful_vote",
        "verified_purchase",
        "text_len",
    ]

    clean_df = df[
        base_cols
    ].copy()

    clean_df = join_metadata(clean_df, input_meta_path)

    clean_df = clean_df.sort_values(by=["product_id", "timestamp", "review_id"]).reset_index(drop=True)

    optional_cols = [col for col in ["detected_lang", "lang_conf"] if col in df.columns]

    clean_df = df[base_cols + optional_cols].copy()
    clean_df = join_metadata(clean_df, input_meta_path)
    clean_df = clean_df.sort_values(
        by=["product_id", "timestamp", "review_id"]
    ).reset_index(drop=True)

    train_val_df, test_df = split_dataset_with_holdout(
        clean_df=clean_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_df, val_df = split_train_val(
        train_val_df=train_val_df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    reviews_clean_path = interim_dir / "reviews_clean.csv"

    reviews_clean_path = interim_dir / "reviews_clean.csv"
    train_val_path = processed_dir / "train_val.csv"
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    clean_df.to_csv(reviews_clean_path, index=False)
    train_val_df.to_csv(train_val_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)


    if args.save_kfold:
        save_kfold_splits(
            train_val_df=train_val_df,
            processed_dir=processed_dir,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )



    print("Preprocess completed.")
    print(f"Original rows: {original_rows:,}")
    print(f"Clean rows: {len(clean_df):,}")
    print(f"Unique products: {clean_df['product_id'].nunique():,}")
    print("Sentiment distribution:")
    print(clean_df["sentiment_label"].value_counts(dropna=False))
    print("Saved files:")
    print(f"- {reviews_clean_path}")
    print(f"- {train_val_path}")
    print(f"- {train_path}")
    print(f"- {val_path}")
    print(f"- {test_path}")
    if args.save_kfold:
        print(f"- {processed_dir / 'folds'}")

if __name__ == "__main__": 
    main()













