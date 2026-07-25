import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from bert_score import score as bertscore_score
from dotenv import load_dotenv
from openai import OpenAI
from rouge_score import rouge_scorer

load_dotenv()

DEFAULT_TEST_PREDICTIONS = Path("./artifacts/predictions/summary_predictions_testset.csv")
DEFAULT_OUTPUT_PATH = Path("./artifacts/eval/metrics_summary_model.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate summary model with ROUGE-L, BERTScore F1, and OpenAI LLM-as-a-Judge"
    )

    parser.add_argument("--input_path", type=Path, default=DEFAULT_TEST_PREDICTIONS)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)

    parser.add_argument("--run_judge", action="store_true")
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--max_workers", type=int, default=8)

    return parser.parse_args()


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_judge_prompt(prompt: str, prediction: str) -> str:
    return f"""
You are evaluating a product review summary.

The original task prompt already contains:
- the target sentiment bucket
- product information
- the source customer reviews
- the summarization instructions

Your job is to evaluate the generated summary against the task prompt and the source reviews inside it.

Task prompt:
{prompt}

Generated summary:
{prediction}

Score the generated summary from 1 to 5 on these criteria:

1. faithfulness
- Are all claims supported by the source reviews?
- Penalize unsupported or invented claims heavily.

2. coverage
- Does the summary capture the main recurring points in the reviews?
- Missing minor details is acceptable, but missing major repeated themes should lower the score.

3. sentiment_alignment
- Does the summary match the intended sentiment bucket in the task prompt?
- For example, a negative summary should focus on complaints, not praise.

4. clarity
- Is the summary concise, readable, and non-redundant?

Scoring rubric:
- 5 = excellent, no meaningful issues
- 4 = good, only minor issues
- 3 = acceptable, but noticeable issues
- 2 = poor, major issues
- 1 = very poor or unusable

Important rules:
- Judge based mainly on the source reviews in the task prompt.
- Do not require any specific wording.
- Reward semantic correctness over surface similarity.
- Hallucinations should strongly reduce faithfulness and overall score.
- Overall score should reflect practical usefulness, with faithfulness weighted most heavily.

Return ONLY valid JSON in this exact format:
{{
  "faithfulness": 1,
  "coverage": 1,
  "sentiment_alignment": 1,
  "clarity": 1,
  "overall": 1.0
}}
""".strip()


def call_openai_judge(
    judge_model: str,
    prompt: str,
    prediction: str,
) -> dict[str, Any]:
    client = OpenAI()
    judge_prompt = build_judge_prompt(prompt, prediction)

    response = client.responses.create(
        model=judge_model,
        input=judge_prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "summary_judge_scores",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
                        "coverage": {"type": "integer", "minimum": 1, "maximum": 5},
                        "sentiment_alignment": {"type": "integer", "minimum": 1, "maximum": 5},
                        "clarity": {"type": "integer", "minimum": 1, "maximum": 5},
                        "overall": {"type": "number", "minimum": 1, "maximum": 5},
                    },
                    "required": [
                        "faithfulness",
                        "coverage",
                        "sentiment_alignment",
                        "clarity",
                        "overall",
                    ],
                    "additionalProperties": False,
                },
            }
        },
    )

    parsed = json.loads(response.output_text)
    return {
        "faithfulness": float(parsed["faithfulness"]),
        "coverage": float(parsed["coverage"]),
        "sentiment_alignment": float(parsed["sentiment_alignment"]),
        "clarity": float(parsed["clarity"]),
        "overall": float(parsed["overall"]),
    }


def run_judge_parallel(
    prompts: list[str],
    generated_answers: list[str],
    judge_model: str,
    max_workers: int,
) -> list[dict[str, Any]]:
    judge_rows: list[dict[str, Any] | None] = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                call_openai_judge,
                judge_model,
                prompts[idx],
                generated_answers[idx],
            ): idx
            for idx in range(len(prompts))
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            judge_rows[idx] = future.result()

    return judge_rows


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_path)

    prompts, ref_answers, generated_answers = [], [], []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = []

    for _, row in df.iterrows():
        prompt_used = safe_text(row["prompt"])
        ref_ans = safe_text(row["reference_answer"])
        generated_ans = safe_text(row["generated_answer"])

        prompts.append(prompt_used)
        ref_answers.append(ref_ans)
        generated_answers.append(generated_ans)

        rouge_result = scorer.score(ref_ans, generated_ans)
        rouge_l_scores.append(rouge_result["rougeL"].fmeasure)

    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    _, _, f1 = bertscore_score(
        generated_answers,
        ref_answers,
        lang="en",
        rescale_with_baseline=True,
    )
    bertscore_f1_scores = [float(x) for x in f1]
    avg_bertscore_f1 = sum(bertscore_f1_scores) / len(bertscore_f1_scores)

    result = {
        "count": len(df),
        "avg_rouge_l": avg_rouge_l,
        "avg_bertscore_f1": avg_bertscore_f1,
    }

    if args.run_judge:
        judge_rows = run_judge_parallel(
            prompts=prompts,
            generated_answers=generated_answers,
            judge_model=args.judge_model,
            max_workers=args.max_workers,
        )

        result["llm_judge"] = {
            "avg_faithfulness": sum(x["faithfulness"] for x in judge_rows) / len(judge_rows),
            "avg_coverage": sum(x["coverage"] for x in judge_rows) / len(judge_rows),
            "avg_sentiment_alignment": sum(x["sentiment_alignment"] for x in judge_rows)
            / len(judge_rows),
            "avg_clarity": sum(x["clarity"] for x in judge_rows) / len(judge_rows),
            "avg_overall": sum(x["overall"] for x in judge_rows) / len(judge_rows),
        }

        df["judge_faithfulness"] = [x["faithfulness"] for x in judge_rows]
        df["judge_coverage"] = [x["coverage"] for x in judge_rows]
        df["judge_sentiment_alignment"] = [x["sentiment_alignment"] for x in judge_rows]
        df["judge_clarity"] = [x["clarity"] for x in judge_rows]
        df["judge_overall"] = [x["overall"] for x in judge_rows]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # append JSONL
    with open(args.output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    per_sample_path = args.output_path.with_name("metrics_summary_model_per_sample.csv")
    df["rouge_l"] = rouge_l_scores
    df["bertscore_f1"] = bertscore_f1_scores
    df.to_csv(per_sample_path, index=False)

    print("Avg ROUGE-L:", avg_rouge_l)
    print("Avg BERTScore F1:", avg_bertscore_f1)
    if args.run_judge:
        print("LLM Judge:", result["llm_judge"])

    print(f"Appended aggregate metrics to: {args.output_path}")
    print(f"Saved per-sample metrics to: {per_sample_path}")


if __name__ == "__main__":
    main()
