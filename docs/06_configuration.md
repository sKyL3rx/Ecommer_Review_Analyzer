# Configuration

This project uses two configuration layers:

- `.env` for runtime application settings and external API keys,
- `params.yaml` for the offline data and ML pipeline.

## 1. Runtime configuration with `.env`

The FastAPI application and RQ worker load settings from `.env` through `src/app/core/config.py`.

Create the local file from the template:

```bash
cp .env.example .env
```

### Application settings

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLAlchemy connection string for Postgres. |
| `REDIS_URL` | Redis connection used by RQ and the insight cache. |
| `VLLM_BASE_URL` | OpenAI-compatible vLLM endpoint. It must include `http://` or `https://` and ends with `/v1`. |
| `VLLM_API_KEY` | Bearer token sent to vLLM. |
| `SUMMARY_MODEL_NAME` | Model name sent in `/chat/completions` requests. For LoRA serving, this should match the adapter name exposed by vLLM. |
| `SUMMARY_MODEL_VERSION` | Version stored with generated product insights. |
| `USE_FAKE_SUMMARIZER` | Uses mock summaries instead of vLLM when set to `true`. |
| `USE_FAKE_SENTIMENT` | Uses the mock sentiment predictor instead of the trained sentiment model when set to `true`. |
| `AUTO_CREATE_TABLES` | Allows application-side table creation when enabled. Migrations are preferred for normal use. |
| `REDIS_CACHE_TTL_SECONDS` | Cache lifetime for generated insight payloads. |
| `GEMINI_API_KEY` | Required only when generating synthetic summaries with Gemini. |
| `OPENAI_API_KEY` | Required only when generating synthetic summaries with an OpenAI-compatible provider. |

The current application settings class expects the variable name:

```env
SUMMARY_MODEL_VERSION=summary-sft
```

## 2. Mock-mode configuration

Mock mode is useful for validating the API, Postgres, Redis, queue, worker, cache, and job flow without loading ML models.

When the API and worker run with Docker Compose:

```env
DATABASE_URL=postgresql+psycopg://ecom:ecom@postgres:5432/ecom_review
REDIS_URL=redis://redis:6379/0

USE_FAKE_SUMMARIZER=true
USE_FAKE_SENTIMENT=true

SUMMARY_MODEL_NAME=summary-sft
SUMMARY_MODEL_VERSION=summary-sft

AUTO_CREATE_TABLES=false
REDIS_CACHE_TTL_SECONDS=3600
```

Mock mode does not require a GPU, vLLM, Gemini, or OpenAI API access.

## 3. Real inference configuration

Real inference uses the trained sentiment artifact and the summary model served through vLLM.

```env
USE_FAKE_SUMMARIZER=false
USE_FAKE_SENTIMENT=false

VLLM_API_KEY=demo-key
SUMMARY_MODEL_NAME=summary-sft
SUMMARY_MODEL_VERSION=summary-sft
```

## 4. Model naming

The summary request uses:

```python
model=settings.summary_model_name
```

When vLLM serves the LoRA adapter with a name such as:

```bash
--lora-modules summary-sft=artifacts/models/summary_sft
```

the application should use:

```env
SUMMARY_MODEL_NAME=summary-sft
```

`SUMMARY_MODEL_VERSION` is separate from the request model name. It is used to identify and store generated insight versions in Postgres and Redis.

A simple local setup can use the same value for both:

```env
SUMMARY_MODEL_NAME=summary-sft
SUMMARY_MODEL_VERSION=summary-sft
```

## 5. Synthetic summary provider settings

The synthetic summary generation stage is implemented in:

```text
scripts/generate_synthetic_gold_summary_via_api.py
```

For Gemini:

```env
GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
```

Example:

```bash
python -m scripts.generate_synthetic_gold_summary_via_api \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --input_path data/processed/phase2/product_sentiment_groups_ranked.jsonl \
  --output_path data/processed/phase2/phase2_sft_summary_dataset.jsonl \
  --errors_path data/processed/phase2/phase2_sft_summary_errors.jsonl \
  --resume
```

For an OpenAI-compatible provider:

```env
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

## 6. Offline pipeline configuration with `params.yaml`

`params.yaml` stores reproducible data and model settings used by DVC stages.

The main parameter groups cover:

| Group | Examples |
|---|---|
| Data paths | Raw reviews, metadata, interim files, train/validation/test splits, serving parquet files. |
| Phase 1 preprocessing | Language filtering, minimum review length, split ratios, random seed, optional K-fold output. |
| Sentiment training | Text and target columns, TF-IDF settings, Logistic Regression settings, output paths. |
| Sentiment evaluation | Test path, model artifact paths, report and metric outputs. |
| Phase 2 preprocessing | Group size, representative review count, quality-score weights, embedding diversity settings. |
| Synthetic summary generation | Provider, model, top-k reviews, concurrency, resume behavior, output paths. |
| TRL/SFT data preparation | Input dataset, grouped split ratios, random seed, output directory. |
| SFT training | Base model, sequence length, LoRA/QLoRA settings, batch sizes, learning rate, epochs, checkpoints. |
| Summary generation and evaluation | vLLM URL, model name, generation settings, prediction and metric outputs. |
| Serving data | Raw serving files and final catalog/review parquet paths. |
| MLflow registration | Experiment, run, model name, version, alias, artifact and metric paths. |

`dvc.yaml` resolves references such as:

```yaml
${data.train_path}
${preprocess_phase1.random_state}
${sft.model_name}
```

from `params.yaml`.

After changing pipeline parameters, reproduce only the affected stage:

```bash
.venv/bin/python -m dvc repro <stage_name>
```

Examples:

```bash
.venv/bin/python -m dvc repro preprocess_phase1
.venv/bin/python -m dvc repro train_sentiment_baseline
.venv/bin/python -m dvc repro preprocess_phase2_review_groups
.venv/bin/python -m dvc repro train_summary_sft
```

Review parameter changes with:

```bash
.venv/bin/python -m dvc params diff
```

## 7. Validate the effective configuration

Check the Docker Compose configuration:

```bash
docker compose -f infra/compose/docker-compose.yml config >/dev/null
echo "compose configuration: OK"
```

After the services are running, inspect these application settings inside the worker:

```bash
docker compose -f infra/compose/docker-compose.yml exec worker python - <<'PY'
from src.app.core.config import settings

print("database_url:", settings.database_url)
print("redis_url:", settings.redis_url)
print("vllm_base_url:", settings.vllm_base_url)
print("summary_model_name:", settings.summary_model_name)
print("summary_model_version:", settings.summary_model_version)
print("use_fake_summarizer:", settings.use_fake_summarizer)
print("use_fake_sentiment:", settings.use_fake_sentiment)
print("redis_cache_ttl_seconds:", settings.redis_cache_ttl_seconds)
PY
```

When real inference is enabled, verify that the worker can reach vLLM:

```bash
docker compose -f infra/compose/docker-compose.yml exec worker python - <<'PY'
import os
import requests

base_url = os.environ["VLLM_BASE_URL"].rstrip("/")
api_key = os.environ.get("VLLM_API_KEY", "")

response = requests.get(
    f"{base_url}/models",
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=20,
)

print("status:", response.status_code)
response.raise_for_status()
print(response.json())
PY
```