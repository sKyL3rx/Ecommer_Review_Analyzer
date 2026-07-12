from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
import requests
from huggingface_hub import hf_hub_url
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OUT_DIR = Path("data/raw")
DEFAULT_CHUNK_SIZE = 10_000

REVIEW_COLUMNS = [
    "product_id",
    "rating",
    "review_title",
    "review_text",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
    "images",
]

META_COLUMNS = [
    "product_id",
    "product_title",
    "average_rating",
    "rating_number",
    "price",
    "store",
    "main_category",
    "categories",
    "features",
    "description",
    "images",
    "details",
]


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and process Amazon Reviews 2023 JSONL data. "
            "Sample outputs are saved as CSV; full outputs are saved as Parquet."
        )
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="McAuley-Lab/Amazon-Reviews-2023",
        help="Hugging Face dataset repository.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default="Appliances",
        help="Amazon review category.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUT_DIR,
        help="Directory used to save CSV or Parquet outputs.",
    )

    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help=(
            "Number of valid reviews to save. "
            "When omitted, download the full dataset and save it as Parquet."
        ),
    )

    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Dataset branch, tag, or commit.",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of rows written per batch.",
    )

    parser.add_argument(
        "--force_reviews",
        action="store_true",
        help="Rebuild reviews even when a complete sampled review CSV already exists.",
    )

    return parser.parse_args()


def create_http_session() -> requests.Session:
    """
    Create an HTTP session with retries for temporary network/server errors.
    """
    retry = Retry(
        total=10,
        connect=10,
        read=10,
        status=10,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=4,
    )

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": "Ecommer-Review-Analyzer/1.0",
            "Accept-Encoding": "identity",
        }
    )

    return session


def iter_remote_jsonl(
    url: str,
    source_name: str,
) -> Iterator[dict[str, Any]]:
    """
    Stream one JSON object per line.

    This deliberately avoids Hugging Face Datasets and PyArrow schema
    inference. Rows are allowed to have different fields.
    """
    session = create_http_session()

    try:
        with session.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=(30, 600),
        ) as response:
            response.raise_for_status()

            lines = response.iter_lines(
                chunk_size=1024 * 1024,
            )

            for line_number, raw_line in enumerate(
                lines,
                start=1,
            ):
                if not raw_line:
                    continue

                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in {source_name} "
                        f"at line {line_number:,}."
                    ) from exc

                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"Expected JSON object in {source_name} "
                        f"at line {line_number:,}, "
                        f"got {type(row).__name__}."
                    )

                yield row

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed while downloading {source_name}: {exc}"
        ) from exc

    finally:
        session.close()


def serialize_nested(value: Any) -> Any:
    """
    Serialize nested dictionaries/lists into JSON strings.
    """
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return value


def is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def to_optional_string(value: Any) -> str | None:
    if is_missing(value):
        return None

    return str(value)


def to_optional_float(value: Any) -> float | None:
    if is_missing(value):
        return None

    if isinstance(value, str):
        value = value.strip().replace(",", "").replace("$", "")
        if not value:
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_optional_int(value: Any) -> int | None:
    if is_missing(value):
        return None

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_optional_bool(value: Any) -> bool | None:
    if is_missing(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    return None


def review_parquet_schema():
    """
    Return the explicit schema used for full review Parquet output.

    PyArrow is imported lazily so sampled CSV downloads do not require it.
    """
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "PyArrow is required for full Parquet downloads. "
            "Install it with: python -m pip install pyarrow"
        ) from exc

    return pa.schema(
        [
            ("product_id", pa.string()),
            ("rating", pa.float64()),
            ("review_title", pa.string()),
            ("review_text", pa.string()),
            ("timestamp", pa.int64()),
            ("helpful_vote", pa.int64()),
            ("verified_purchase", pa.bool_()),
            ("images", pa.string()),
        ]
    )


def meta_parquet_schema():
    """
    Return the explicit schema used for full metadata Parquet output.
    """
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "PyArrow is required for full Parquet downloads. "
            "Install it with: python -m pip install pyarrow"
        ) from exc

    return pa.schema(
        [
            ("product_id", pa.string()),
            ("product_title", pa.string()),
            ("average_rating", pa.float64()),
            ("rating_number", pa.int64()),
            ("price", pa.float64()),
            ("store", pa.string()),
            ("main_category", pa.string()),
            ("categories", pa.string()),
            ("features", pa.string()),
            ("description", pa.string()),
            ("images", pa.string()),
            ("details", pa.string()),
        ]
    )


class ChunkedTableWriter:
    """
    Write rows incrementally as CSV or Parquet.

    The output format is selected from the final output path:
    - .csv: sampled dataset
    - .parquet: full dataset
    """

    def __init__(
        self,
        output_path: Path,
        columns: list[str],
        parquet_schema_factory: Callable[[], Any],
    ) -> None:
        self.output_path = output_path
        self.columns = columns
        self.suffix = output_path.suffix.lower()
        self.header_written = False
        self.parquet_writer = None
        self.parquet_schema_factory = parquet_schema_factory
        self.parquet_schema = None

        if self.suffix not in {".csv", ".parquet"}:
            raise ValueError(
                f"Unsupported output format: {output_path}. "
                "Expected .csv or .parquet."
            )

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        if self.suffix == ".csv":
            frame = pd.DataFrame(
                rows,
                columns=self.columns,
            )

            frame.to_csv(
                self.output_path,
                mode="a",
                index=False,
                header=not self.header_written,
            )

            self.header_written = True

        else:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "PyArrow is required for full Parquet downloads. "
                    "Install it with: python -m pip install pyarrow"
                ) from exc

            if self.parquet_schema is None:
                self.parquet_schema = self.parquet_schema_factory()

            table = pa.Table.from_pylist(
                rows,
                schema=self.parquet_schema,
            )

            if self.parquet_writer is None:
                self.parquet_writer = pq.ParquetWriter(
                    where=str(self.output_path),
                    schema=self.parquet_schema,
                    compression="zstd",
                )

            self.parquet_writer.write_table(table)

        rows.clear()

    def close(self, create_empty: bool = True) -> None:
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            self.parquet_writer = None
            return

        if not create_empty:
            return

        if self.suffix == ".csv":
            if not self.header_written:
                pd.DataFrame(
                    columns=self.columns
                ).to_csv(
                    self.output_path,
                    index=False,
                )
            return

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "PyArrow is required for full Parquet downloads. "
                "Install it with: python -m pip install pyarrow"
            ) from exc

        if self.parquet_schema is None:
            self.parquet_schema = self.parquet_schema_factory()

        empty_table = pa.Table.from_pylist(
            [],
            schema=self.parquet_schema,
        )

        pq.write_table(
            empty_table,
            str(self.output_path),
            compression="zstd",
        )


def build_dataset_urls(
    dataset_name: str,
    category: str,
    revision: str,
) -> tuple[str, str]:
    """
    Build direct Hugging Face download URLs.
    """
    review_url = hf_hub_url(
        repo_id=dataset_name,
        filename=(
            f"raw/review_categories/"
            f"{category}.jsonl"
        ),
        repo_type="dataset",
        revision=revision,
    )

    meta_url = hf_hub_url(
        repo_id=dataset_name,
        filename=(
            f"raw/meta_categories/"
            f"meta_{category}.jsonl"
        ),
        repo_type="dataset",
        revision=revision,
    )

    return review_url, meta_url


def read_existing_review_ids(
    reviews_path: Path,
    expected_rows: int | None,
) -> tuple[int, set[str]] | None:
    """
    Reuse a completed sampled review CSV.

    Full downloads are intentionally rebuilt because the expected full row
    count is unknown and completeness cannot be verified safely.
    """
    if not reviews_path.exists():
        return None

    if expected_rows is None:
        return None

    if reviews_path.suffix.lower() != ".csv":
        return None

    row_count = 0
    product_ids: set[str] = set()

    try:
        chunks = pd.read_csv(
            reviews_path,
            usecols=["product_id"],
            dtype={"product_id": "string"},
            chunksize=100_000,
        )

        for chunk in chunks:
            row_count += len(chunk)

            values = (
                chunk["product_id"]
                .dropna()
                .astype(str)
                .str.strip()
            )

            product_ids.update(values)

    except (
        OSError,
        ValueError,
        pd.errors.ParserError,
    ):
        return None

    product_ids.discard("")

    if row_count != expected_rows:
        return None

    return row_count, product_ids


def process_reviews(
    review_url: str,
    reviews_out: Path,
    sample_n: int | None,
    chunk_size: int,
) -> tuple[int, set[str]]:
    """
    Stream, validate, and save review rows.

    Sampled reviews are written as CSV. Full reviews are written as Parquet.
    """
    reviews_tmp = reviews_out.with_name(
        f"{reviews_out.stem}.tmp{reviews_out.suffix}"
    )

    reviews_tmp.unlink(missing_ok=True)

    buffer: list[dict[str, Any]] = []
    product_ids: set[str] = set()

    review_count = 0
    scanned_count = 0

    writer = ChunkedTableWriter(
        output_path=reviews_tmp,
        columns=REVIEW_COLUMNS,
        parquet_schema_factory=review_parquet_schema,
    )

    try:
        for row in iter_remote_jsonl(
            review_url,
            "review JSONL",
        ):
            scanned_count += 1

            product_id = row.get("parent_asin")
            rating = row.get("rating")
            review_text = row.get("text")

            if (
                product_id is None
                or rating is None
                or review_text is None
            ):
                continue

            product_id = str(product_id).strip()
            review_text = str(review_text).strip()

            if not product_id or not review_text:
                continue

            rating_value = to_optional_float(rating)
            if rating_value is None:
                continue

            buffer.append(
                {
                    "product_id": product_id,
                    "rating": rating_value,
                    "review_title": to_optional_string(
                        row.get("title")
                    ),
                    "review_text": review_text,
                    "timestamp": to_optional_int(
                        row.get("timestamp")
                    ),
                    "helpful_vote": to_optional_int(
                        row.get("helpful_vote")
                    ),
                    "verified_purchase": to_optional_bool(
                        row.get("verified_purchase")
                    ),
                    "images": to_optional_string(
                        serialize_nested(
                            row.get("images")
                        )
                    ),
                }
            )

            product_ids.add(product_id)
            review_count += 1

            if len(buffer) >= chunk_size:
                writer.write_rows(buffer)

                print(
                    f"Saved {review_count:,} valid reviews "
                    f"after scanning "
                    f"{scanned_count:,} rows..."
                )

            if (
                sample_n is not None
                and review_count >= sample_n
            ):
                break

        writer.write_rows(buffer)

        if review_count == 0:
            raise RuntimeError(
                "No valid reviews were downloaded."
            )

        if (
            sample_n is not None
            and review_count < sample_n
        ):
            raise RuntimeError(
                f"Only found {review_count:,} valid reviews; "
                f"requested {sample_n:,}."
            )

        writer.close()
        reviews_tmp.replace(reviews_out)

    except Exception:
        writer.close(create_empty=False)
        reviews_tmp.unlink(missing_ok=True)
        raise

    return review_count, product_ids


def process_metadata(
    meta_url: str,
    meta_out: Path,
    product_ids: set[str],
    chunk_size: int,
) -> tuple[int, set[str], int]:
    """
    Stream metadata JSONL without imposing a source-side fixed schema.

    Sampled metadata is written as CSV. Full metadata is written as Parquet.
    """
    meta_tmp = meta_out.with_name(
        f"{meta_out.stem}.tmp{meta_out.suffix}"
    )

    meta_tmp.unlink(missing_ok=True)

    buffer: list[dict[str, Any]] = []
    matched_ids: set[str] = set()

    matched_count = 0
    scanned_count = 0

    writer = ChunkedTableWriter(
        output_path=meta_tmp,
        columns=META_COLUMNS,
        parquet_schema_factory=meta_parquet_schema,
    )

    try:
        for row in iter_remote_jsonl(
            meta_url,
            "metadata JSONL",
        ):
            scanned_count += 1

            raw_product_id = row.get("parent_asin")

            if raw_product_id is None:
                continue

            product_id = str(
                raw_product_id
            ).strip()

            if not product_id:
                continue

            if product_id not in product_ids:
                continue

            if product_id in matched_ids:
                continue

            # subtitle, author, videos and unexpected fields are ignored.
            buffer.append(
                {
                    "product_id": product_id,
                    "product_title": to_optional_string(
                        row.get("title")
                    ),
                    "average_rating": to_optional_float(
                        row.get("average_rating")
                    ),
                    "rating_number": to_optional_int(
                        row.get("rating_number")
                    ),
                    "price": to_optional_float(
                        row.get("price")
                    ),
                    "store": to_optional_string(
                        row.get("store")
                    ),
                    "main_category": to_optional_string(
                        row.get("main_category")
                    ),
                    "categories": to_optional_string(
                        serialize_nested(
                            row.get("categories")
                        )
                    ),
                    "features": to_optional_string(
                        serialize_nested(
                            row.get("features")
                        )
                    ),
                    "description": to_optional_string(
                        serialize_nested(
                            row.get("description")
                        )
                    ),
                    "images": to_optional_string(
                        serialize_nested(
                            row.get("images")
                        )
                    ),
                    "details": to_optional_string(
                        serialize_nested(
                            row.get("details")
                        )
                    ),
                }
            )

            matched_ids.add(product_id)
            matched_count += 1

            if len(buffer) >= chunk_size:
                writer.write_rows(buffer)

                print(
                    f"Matched {matched_count:,} products "
                    f"after scanning "
                    f"{scanned_count:,} metadata rows..."
                )

            if len(matched_ids) == len(product_ids):
                break

        writer.write_rows(buffer)
        writer.close()
        meta_tmp.replace(meta_out)

    except Exception:
        writer.close(create_empty=False)
        meta_tmp.unlink(missing_ok=True)
        raise

    return matched_count, matched_ids, scanned_count


def main() -> None:
    args = arg_parser()

    if (
        args.sample_n is not None
        and args.sample_n <= 0
    ):
        raise ValueError(
            "--sample_n must be greater than 0."
        )

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk_size must be greater than 0."
        )

    output_dir: Path = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    category_slug = (
        args.category
        .lower()
        .replace(" ", "_")
    )

    if args.sample_n is None:
        # Only full downloads are stored as Parquet.
        reviews_out = output_dir / (
            f"{category_slug}_reviews_full.parquet"
        )

        meta_out = output_dir / (
            f"{category_slug}_meta_full.parquet"
        )

    else:
        # Sampled downloads remain CSV.
        reviews_out = output_dir / (
            f"{category_slug}_reviews_sample_"
            f"{args.sample_n}.csv"
        )

        meta_out = output_dir / (
            f"{category_slug}_meta_matched.csv"
        )

    review_url, meta_url = build_dataset_urls(
        dataset_name=args.dataset_name,
        category=args.category,
        revision=args.revision,
    )

    print(f"Dataset: {args.dataset_name}")
    print(f"Category: {args.category}")
    print(f"Review source: {review_url}")
    print(f"Metadata source: {meta_url}")
    print(
        "Output format: "
        f"{'Parquet' if args.sample_n is None else 'CSV'}"
    )

    existing_reviews = None

    if not args.force_reviews:
        existing_reviews = read_existing_review_ids(
            reviews_path=reviews_out,
            expected_rows=args.sample_n,
        )

    if existing_reviews is not None:
        review_count, product_ids = existing_reviews

        print(
            "\nReusing existing sampled review CSV:"
        )

        print(f"  File: {reviews_out}")
        print(f"  Rows: {review_count:,}")
        print(
            f"  Unique products: "
            f"{len(product_ids):,}"
        )

    else:
        print(
            "\nStreaming reviews without "
            "Datasets/Arrow schema inference..."
        )

        review_count, product_ids = process_reviews(
            review_url=review_url,
            reviews_out=reviews_out,
            sample_n=args.sample_n,
            chunk_size=args.chunk_size,
        )

        print(
            f"\nFinished reviews: "
            f"{review_count:,}"
        )

        print(
            f"Unique products: "
            f"{len(product_ids):,}"
        )

        print(
            f"Saved reviews to: "
            f"{reviews_out}"
        )

    print(
        "\nStreaming metadata without "
        "Datasets/Arrow schema inference..."
    )

    matched_count, matched_ids, scanned_meta = (
        process_metadata(
            meta_url=meta_url,
            meta_out=meta_out,
            product_ids=product_ids,
            chunk_size=args.chunk_size,
        )
    )

    missing_count = (
        len(product_ids)
        - len(matched_ids)
    )

    print("\nDone.")

    print(
        f"Reviews: "
        f"{review_count:,}"
    )

    print(
        f"Unique review products: "
        f"{len(product_ids):,}"
    )

    print(
        f"Metadata rows scanned: "
        f"{scanned_meta:,}"
    )

    print(
        f"Matched metadata: "
        f"{matched_count:,}"
    )

    print(
        f"Products without metadata: "
        f"{missing_count:,}"
    )

    print(
        f"Reviews file: "
        f"{reviews_out}"
    )

    print(
        f"Metadata file: "
        f"{meta_out}"
    )


if __name__ == "__main__":
    main()
