# Evaluation and Benchmarking

This project evaluates two different things:

1. **model quality** — how well the sentiment and summarization models perform,
2. **serving behavior** — how the API, background jobs, cache, and model endpoint behave at runtime.

## 1. Evaluation overview

| Area | Script | Main outputs |
|---|---|---|
| Sentiment validation/test evaluation | `src/evaluation/evaluate_sentiment_baseline_model.py` | Classification metrics, report, confusion matrix, per-row predictions |
| Summary test-set generation | `scripts/summary_model_generation_testset.py` | Generated summaries and generation latency statistics |
| Summary quality evaluation | `src/evaluation/evaluate_summary_model.py` | ROUGE-L, BERTScore F1, optional LLM-as-a-Judge scores |
| Local serving benchmark | `scripts/benchmark_local_serving.py` | API latency, async job latency, cached-read latency, success rates |

## 2. Sentiment model evaluation

The sentiment baseline uses TF-IDF features and a Logistic Regression classifier. Evaluation is performed on the held-out test split.

**Script:**

```text
src/evaluation/evaluate_sentiment_baseline_model.py
```

### Run the evaluation

```bash
make eval-sentiment
```

To reproduce the complete sentiment path:

```bash
make sentiment-e2e
```

The equivalent DVC stage is:

```bash
.venv/bin/python -m dvc repro evaluate_sentiment_baseline
```

### Metrics

The evaluator reports:

| Metric | Why it is included |
|---|---|
| `accuracy` | Overall percentage of correct predictions |
| `precision_macro` | Mean precision across positive, neutral, and negative classes |
| `recall_macro` | Mean recall across the three classes |
| `f1_macro` | Treats every class equally, regardless of class frequency |
| `precision_weighted` | Precision weighted by class support |
| `recall_weighted` | Recall weighted by class support |
| `f1_weighted` | F1 weighted by class support |

The review dataset is imbalanced toward positive ratings. For that reason, `f1_macro` is more useful than accuracy alone.

### Outputs

The exact paths come from `params.yaml`. The evaluation stage produces files equivalent to:

```text
artifacts/reports/metrics.json
artifacts/reports/classification_report.txt
artifacts/reports/confusion_matrix.png
artifacts/reports/test_predictions.csv
artifacts/reports/evaluation_summary.json
```

The prediction file includes the original test rows, predicted sentiment, class probabilities when supported by the model, and prediction confidence.

### Recorded baseline result

One recorded test run in the repository produced:

| Metric | Value |
|---|---:|
| Accuracy | 0.8686 |
| Macro precision | 0.6512 |
| Macro recall | 0.7480 |
| Macro F1 | 0.6818 |
| Weighted F1 | 0.8861 |

## 3. Summary test-set generation

**Script:**

```text
scripts/summary_model_generation_testset.py
```

The script:

1. reads chat-formatted examples from the test JSONL,
2. extracts the user prompt and reference assistant answer,
3. sends each prompt to the OpenAI-compatible summary endpoint,
4. records the generated answer and request latency,
5. saves one row per test example.

### Prerequisite

The vLLM endpoint must be running and serving the adapter name configured in `params.yaml`.

Verify it first:

```bash
make wait-vllm
```

### Generate predictions

```bash
make summary-predict
```

The equivalent DVC stage is:

```bash
.venv/bin/python -m dvc repro generate_summary_predictions
```

### Prediction output

The generated table contains:

```text
prompt
reference_answer
generated_answer
latency_ms
messages
```

The output path is:

```text
artifacts/predictions/summary_predictions_testset.csv
```

The generation stage also writes latency statistics such as:

```text
count
mean_ms
min_ms
max_ms
p50_ms
p90_ms
p95_ms
p99_ms
```

A recorded run over 970 test examples produced a median generation latency of about 741 ms and a p95 latency of about 928 ms.


## 4. Summary quality evaluation

**Script:**

```text
src/evaluation/evaluate_summary_model.py
```

Run:

```bash
make summary-eval
```

The equivalent DVC stage is:

```bash
.venv/bin/python -m dvc repro evaluate_summary_model
```

When the prediction file is missing or stale, DVC can reproduce the required upstream generation stage first.

### Automatic metrics

The evaluator uses:

| Metric | What it measures |
|---|---|
| ROUGE-L F1 | Longest-subsequence overlap with the reference summary |
| BERTScore F1 | Semantic similarity between generated and reference summaries |

These metrics are useful for repeatable comparisons, but neither metric directly proves that a summary is faithful to the source reviews.

### Optional LLM-as-a-Judge

When enabled in `params.yaml`, the evaluator also scores each generated summary on:

```text
faithfulness
coverage
sentiment_alignment
clarity
overall
```

Scores are on a 1–5 scale.

The judge receives the original prompt, which already contains the source reviews, together with the generated summary. Faithfulness is treated as the most important criterion.

This mode requires an `OPENAI_API_KEY`, makes external API calls, and may incur cost.

### Outputs

The summary evaluation stage writes:

- one aggregate metrics file,
- one per-sample file containing the generated output and its scores.

The exact paths are configured under `summary_eval` in `params.yaml` and tracked by DVC.

### Recorded summary evaluation result

The summary model was evaluated on 970 held-out test examples.

| Metric | Average score |
|---|---:|
| ROUGE-L F1 | 0.4358 |
| BERTScore F1 | 0.5813 |
| LLM Judge — Faithfulness | 4.95 / 5 |
| LLM Judge — Coverage | 4.36 / 5 |
| LLM Judge — Sentiment alignment | 4.99 / 5 |
| LLM Judge — Clarity | 4.99 / 5 |
| LLM Judge — Overall | 4.82 / 5 |

### Notes:
- **ROUGE-L** checks lexical overlap with the synthetic reference.
- **BERTScore** gives a less wording-sensitive semantic comparison.
- **Faithfulness** checks whether claims are supported by the source reviews.
- **Coverage** checks whether recurring themes were included.
- **Sentiment alignment** checks whether the summary matches the requested positive, neutral, or negative bucket.
- **Clarity** checks conciseness and readability.

## 5. Full serving pipeline benchmark

**Script:**

```text
scripts/benchmark_local_serving.py
```

The benchmark covers the runtime path:

```text
FastAPI
→ Postgres / Redis
→ RQ worker
→ sentiment inference
→ vLLM summary generation
→ saved and cached insight
```

### Prerequisites

Before benchmarking:

```bash
make compose-up
make load-serving-data
make smoke
```

For real summary inference:

```bash
make restart-vllm
```

### Standard run

```bash
make benchmark-local
```

Default settings:

```text
10 products
100 requests per GET endpoint
10 async insight jobs
50 reviews maximum per job
3 representative reviews per sentiment group
```

Output:

```text
artifacts/benchmarks/local_serving_benchmark.json
```

### Cold-cache run

```bash
make benchmark-cold-cache
```

Output:

```text
artifacts/benchmarks/local_serving_benchmark_cold_cache.json
```

This target clears Redis before the run.

### What is measured

#### API reads

The script repeatedly calls:

```text
GET /health
GET /products?limit=5&offset=0
```

For each endpoint it reports:

```text
request count
success count
success rate
average latency
p50 latency
p95 latency
p99 latency
minimum and maximum latency
HTTP status counts
sample errors
```

#### Async insight jobs

For each selected product, the script submits:

```text
POST /products/{product_id}/insights/jobs
```

with `regenerate=true`, then polls:

```text
GET /jobs/{job_id}
```

until the job finishes or times out.

It records:

- job submission latency,
- end-to-end job completion latency,
- final status counts,
- finished-job success rate,
- sample job IDs and errors.

#### Cached insight reads

After preparing one saved insight, the script repeatedly calls:

```text
GET /products/{product_id}/insights
```

This measures the read path after the result is available and cached.

### Recorded local example

One recorded local benchmark produced:

| Operation | p50 | p95 |
|---|---:|---:|
| Health endpoint | 2.69 ms | 3.20 ms |
| Product list | 13.71 ms | 14.72 ms |
| Insight job submission | 10.29 ms | 27.31 ms |
| Insight job completion | 6.10 s | 10.11 s |
| Cached insight read | 5.79 ms | 9.78 ms |

## 6. MLflow registration

After evaluation, the project can register the model artifacts and attach their metrics to MLflow.

Summary LoRA:

```bash
make register-summary-lora
```

The registration script logs:

- LoRA adapter files,
- base model name,
- LoRA configuration,
- dataset and model version tags,
- selected evaluation metrics,
- a registered-model alias.

The sentiment baseline can be registered with:

```bash
.venv/bin/python -m scripts.register_sentiment_to_mlflow \
  --vectorizer-path artifacts/models/tfidf.joblib \
  --model-path artifacts/models/sentiment_model.joblib \
  --metrics-path artifacts/reports/metrics.json \
  --training-env local \
  --dataset-version appliances-local-v1 \
  --alias local-baseline
```
---

## 7. Recommended evaluation order

```bash
# Sentiment classification quality
make eval-sentiment

# Summary generation through vLLM
make summary-predict

# Summary quality metrics
make summary-eval

# Runtime latency and job behavior
make benchmark-local

```
