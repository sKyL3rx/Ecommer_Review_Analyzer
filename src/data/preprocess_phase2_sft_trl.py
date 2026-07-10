from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

INPUT_PATH = Path("data/processed/phase2/phase2_sft_summary_dataset.jsonl")
OUTPUT_DIR = Path("data/processed/phase2/trl_data")


def to_messages(prompt: str, gold_summary: str) -> list[dict]:
    return [
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": gold_summary.strip()},
    ]


def main() -> None:

    df = pd.read_json(INPUT_PATH, lines=True)

    df = df[["product_id", "sentiment", "prompt", "gold_summary"]].copy()

    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["gold_summary"] = df["gold_summary"].astype(str).str.strip()

    df = df[(df["prompt"] != "") & (df["gold_summary"] != "")].reset_index(drop=True)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df["product_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx_rel, test_idx_rel = next(gss2.split(temp_df, groups=temp_df["product_id"]))
    val_df = temp_df.iloc[val_idx_rel].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx_rel].reset_index(drop=True)

    def export_messages_only(split_df: pd.DataFrame, out_path: Path) -> None:
        out_df = pd.DataFrame(
            {
                "messages": split_df.apply(
                    lambda row: to_messages(row["prompt"], row["gold_summary"]),
                    axis=1,
                )
            }
        )
        out_df.to_json(out_path, orient="records", lines=True, force_ascii=False)

    export_messages_only(train_df, OUTPUT_DIR / "train.jsonl")
    export_messages_only(val_df, OUTPUT_DIR / "val.jsonl")
    export_messages_only(test_df, OUTPUT_DIR / "test.jsonl")

    print("Done.")
    print(f"train: {len(train_df)}")
    print(f"val:   {len(val_df)}")
    print(f"test:  {len(test_df)}")


if __name__ == "__main__":
    main()
