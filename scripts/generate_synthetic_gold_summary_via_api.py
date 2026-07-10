from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv

load_dotenv()


INPUT_PATH = Path("data/processed/phase2/product_sentiment_groups_ranked.jsonl")
OUTPUT_PATH = Path("data/processed/phase2/phase2_sft_summary_dataset.jsonl")
ERRORS_PATH = Path("data/processed/phase2/phase2_sft_summary_errors.jsonl")

SYSTEM_PROMPT = """
You are a careful product-review summarizer.

Your task is to summarize customer feedback for ONE sentiment group of ONE product.

Rules:
- Use only information supported by the reviews.
- Do not invent product features, defects, or use cases.
- Focus on recurring themes, not one-off comments.
- Keep the summary concise: 1 to 3 sentences.
- Write in plain English.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic gold summaries for Phase 2 SFT."
    )

    parser.add_argument("--input_path", type=str, default=str(INPUT_PATH))
    parser.add_argument("--output_path", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--errors_path", type=str, default=str(ERRORS_PATH))

    parser.add_argument("--provider", type=str, choices=["openai", "gemini"], required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_products", type=int, default=None)
    parser.add_argument(
        "--min_sentiments_present",
        type=int,
        default=1,
        help="Only process products with at least this many non-empty sentiment groups.",
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=0.0,
        help="Optional pause inside each task before calling the API.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip product_id + sentiment pairs that already exist in the output file.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel API calls.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_request_key(product_id: str | None, sentiment: str) -> str:
    return f"{product_id}||{sentiment}"


def load_existing_completed_keys(output_path: Path) -> set[str]:
    keys: set[str] = set()
    if not output_path.exists():
        return keys

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = make_request_key(obj.get("product_id"), obj.get("sentiment"))
                keys.add(key)
            except Exception:
                continue
    return keys


def count_present_sentiments(row: dict[str, Any]) -> int:
    count = 0
    for sentiment in ["positive", "neutral", "negative"]:
        items = row.get(f"{sentiment}_review_items", []) or []
        texts = row.get(f"{sentiment}_reviews", []) or []
        if len(items) > 0 or len(texts) > 0:
            count += 1
    return count


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def dedupe_preserve_order(reviews: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for review in reviews:
        text = normalize_text(review)
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append(text)

    return out


def select_top_k_reviews_from_items(items: list[dict[str, Any]], k: int) -> list[str]:
    texts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        text = item.get("text") or item.get("review_text") or ""
        text = normalize_text(text)

        if text:
            texts.append(text)

    texts = dedupe_preserve_order(texts)
    return texts[:k]


def select_top_k_reviews_from_texts(reviews: list[str], k: int) -> list[str]:
    reviews = dedupe_preserve_order(reviews)
    return reviews[:k]


def build_prompt(
    product_title: str | None,
    main_category: str | None,
    sentiment: str,
    reviews: list[str],
) -> str:
    title = product_title or "Unknown product"
    category = main_category or "Unknown category"
    bullet_reviews = "\n".join([f"{i + 1}. {r}" for i, r in enumerate(reviews)])

    return f"""
Summarize the main {sentiment} customer feedback for this product.

Product title: {title}
Category: {category}

Reviews:
{bullet_reviews}

Requirements:
- Mention only recurring themes supported by the reviews.
- Do not invent details.
- Prefer concrete themes over generic praise/complaints.
- Write one concise summary in 1-3 sentences.
""".strip()


def generate_with_openai(prompt: str, model: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=prompt,
    )
    return response.output_text.strip()


def generate_with_gemini(prompt: str, model: str) -> str:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return (response.text or "").strip()


def generate_summary(provider: str, prompt: str, model: str) -> str:
    if provider == "openai":
        return generate_with_openai(prompt=prompt, model=model)
    return generate_with_gemini(prompt=prompt, model=model)


def sha1_prompt(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def build_tasks(
    rows: list[dict[str, Any]], args: argparse.Namespace, completed_keys: set[str]
) -> tuple[list[dict[str, Any]], int, int]:
    sentiments = [
        ("positive", "positive_review_items", "positive_reviews"),
        ("neutral", "neutral_review_items", "neutral_reviews"),
        ("negative", "negative_review_items", "negative_reviews"),
    ]

    tasks: list[dict[str, Any]] = []
    skipped_products = 0
    skipped_existing = 0

    for row_idx, row in enumerate(rows):
        if args.max_products is not None and row_idx >= args.max_products:
            break

        if count_present_sentiments(row) < args.min_sentiments_present:
            skipped_products += 1
            continue

        product_id = row.get("product_id")
        product_title = row.get("product_title")
        main_category = row.get("main_category")

        for sentiment, item_field, text_field in sentiments:
            request_key = make_request_key(product_id, sentiment)

            if args.resume and request_key in completed_keys:
                skipped_existing += 1
                continue

            raw_items = row.get(item_field, []) or []
            raw_reviews = row.get(text_field, []) or []

            if raw_items:
                selected_reviews = select_top_k_reviews_from_items(raw_items, k=args.top_k)
                original_sentiment_count = len(raw_items)
            else:
                selected_reviews = select_top_k_reviews_from_texts(raw_reviews, k=args.top_k)
                original_sentiment_count = len(raw_reviews)

            if len(selected_reviews) == 0:
                continue

            prompt = build_prompt(
                product_title=product_title,
                main_category=main_category,
                sentiment=sentiment,
                reviews=selected_reviews,
            )

            tasks.append(
                {
                    "request_key": request_key,
                    "product_id": product_id,
                    "product_title": product_title,
                    "main_category": main_category,
                    "sentiment": sentiment,
                    "selected_reviews": selected_reviews,
                    "original_sentiment_count": original_sentiment_count,
                    "prompt": prompt,
                }
            )

    return tasks, skipped_products, skipped_existing


def process_task(
    task: dict[str, Any], provider: str, model: str, sleep_seconds: float
) -> dict[str, Any]:
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    gold_summary = generate_summary(
        provider=provider,
        prompt=task["prompt"],
        model=model,
    )

    return {
        "product_id": task["product_id"],
        "product_title": task["product_title"],
        "main_category": task["main_category"],
        "sentiment": task["sentiment"],
        "source_reviews": task["selected_reviews"],
        "source_review_count": len(task["selected_reviews"]),
        "original_sentiment_count": task["original_sentiment_count"],
        "prompt_hash": sha1_prompt(task["prompt"]),
        "prompt": task["prompt"],
        "gold_summary": gold_summary,
        "generator_provider": provider,
        "generator_model": model,
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    errors_path = Path(args.errors_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    completed_keys = load_existing_completed_keys(output_path) if args.resume else set()

    tasks, skipped_products, skipped_existing = build_tasks(
        rows=rows,
        args=args,
        completed_keys=completed_keys,
    )

    write_lock = Lock()
    completed_lock = Lock()

    written = 0
    error_count = 0

    with (
        output_path.open("a", encoding="utf-8") as out_f,
        errors_path.open("a", encoding="utf-8") as err_f,
    ):
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    process_task,
                    task,
                    args.provider,
                    args.model,
                    args.sleep_seconds,
                ): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    record = future.result()

                    with write_lock:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()

                    with completed_lock:
                        completed_keys.add(task["request_key"])

                    written += 1

                except Exception as e:
                    error_record = {
                        "product_id": task["product_id"],
                        "product_title": task["product_title"],
                        "main_category": task["main_category"],
                        "sentiment": task["sentiment"],
                        "source_review_count": len(task["selected_reviews"]),
                        "prompt_hash": sha1_prompt(task["prompt"]),
                        "error": str(e),
                        "generator_provider": args.provider,
                        "generator_model": args.model,
                    }

                    with write_lock:
                        err_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                        err_f.flush()

                    error_count += 1

    print(f"Prepared tasks: {len(tasks)}")
    print(f"Done. Wrote {written} Phase 2 SFT samples to {output_path}")
    print(f"Skipped existing completed pairs: {skipped_existing}")
    print(f"Skipped products by min_sentiments_present filter: {skipped_products}")
    print(f"Errors logged: {error_count} -> {errors_path}")


if __name__ == "__main__":
    main()
