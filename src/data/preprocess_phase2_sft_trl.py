from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert synthetic gold summary dataset into TRL chat-format splits."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("data/processed/phase2/phase2_sft_summary_dataset.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/phase2/trl_data"),
    )
    parser.add_argument("--train_path", type=Path, default=None)
    parser.add_argument("--val_path", type=Path, default=None)
    parser.add_argument("--test_path", type=Path, default=None)
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=Path("data/processed/phase2/trl_data/split_summary.json"),
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Fraction reserved for val+test before splitting temp into val/test.",
    )
    parser.add_argument(
        "--val_fraction_of_temp",
        type=float,
        default=0.50,
        help="Fraction of temp split used as validation; rest becomes test.",
    )
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def to_messages(prompt: str, gold_summary: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": gold_summary.strip()},
    ]


def export_messages_only(split_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        {
            "messages": split_df.apply(
                lambda row: to_messages(row["prompt"], row["gold_summary"]),
                axis=1,
            )
        }
    )
    out_df.to_json(out_path, orient="records", lines=True, force_ascii=False)


def main() -> None:
    args = parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.train_path or output_dir / "train.jsonl"
    val_path = args.val_path or output_dir / "val.jsonl"
    test_path = args.test_path or output_dir / "test.jsonl"

    df = pd.read_json(args.input_path, lines=True)
    required = {"product_id", "sentiment", "prompt", "gold_summary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[["product_id", "sentiment", "prompt", "gold_summary"]].copy()
    df["product_id"] = df["product_id"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["gold_summary"] = df["gold_summary"].astype(str).str.strip()

    df = df[
        (df["product_id"] != "")
        & (df["prompt"] != "")
        & (df["gold_summary"] != "")
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid SFT rows found in: {args.input_path}")

    product_count = df["product_id"].nunique()
    if product_count < 3:
        raise ValueError(
            "Need at least 3 unique product_id groups for grouped train/val/test split."
        )

    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_idx, temp_idx = next(gss1.split(df, groups=df["product_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=1.0 - args.val_fraction_of_temp,
        random_state=args.random_state,
    )
    val_idx_rel, test_idx_rel = next(gss2.split(temp_df, groups=temp_df["product_id"]))

    val_df = temp_df.iloc[val_idx_rel].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx_rel].reset_index(drop=True)

    export_messages_only(train_df, train_path)
    export_messages_only(val_df, val_path)
    export_messages_only(test_df, test_path)

    summary = {
        "input_path": str(args.input_path),
        "rows_total": int(len(df)),
        "unique_products_total": int(product_count),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_products": int(train_df["product_id"].nunique()),
        "val_products": int(val_df["product_id"].nunique()),
        "test_products": int(test_df["product_id"].nunique()),
        "test_size": float(args.test_size),
        "val_fraction_of_temp": float(args.val_fraction_of_temp),
        "random_state": int(args.random_state),
    }

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("TRL split completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"train: {train_path}")
    print(f"val:   {val_path}")
    print(f"test:  {test_path}")


if __name__ == "__main__":
    main()