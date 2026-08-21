# Running the Application

This document covers the runtime path of the project: starting the backend services, loading serving data, running either mock or real model inference, and testing the main API flow.

The offline preprocessing and training pipeline is documented separately in [`03_data_ml_pipeline.md`](03_data_ml_pipeline.md).

## 1. Runtime services

The local application uses the following services:

| Service | Purpose | Default port |
|---|---|---:|
| FastAPI | Product search, insight jobs, saved insights, and metrics | `8000` |
| RQ worker | Runs insight generation outside the API request | — |
| Postgres | Stores products, reviews, jobs, and generated insights | `5432` |
| Redis | RQ queue, job state, and insight cache | `6379` |
| MLflow | Local experiment and artifact tracking | `5000` |
| vLLM | OpenAI-compatible summary model serving | `8384` |



## 2. Before starting

Complete the installation and configuration steps first:

```text
05_local_installation.md
06_configuration.md
```

The runtime path also expects serving-ready data:

```text
data/serving/appliances_demo_catalog.parquet
data/serving/appliances_demo_reviews.parquet
```

For real inference, the project additionally expects:

```text
artifacts/models/tfidf.joblib
artifacts/models/sentiment_model.joblib
artifacts/models/summary_sft/
```

## 3. Choose an inference mode

### Mock mode

Mock mode is the fastest way to verify the application stack without a GPU or model server.

Set the following values in `.env`:

```env
USE_FAKE_SUMMARIZER=true
USE_FAKE_SENTIMENT=true
```

In this mode, the worker still reads from Postgres, processes an RQ job, saves the result, and writes to Redis. Only the model outputs are replaced with mocking test behavior.

### Real inference mode

Use real mode to run the trained sentiment model and the summary LoRA served through vLLM:

```env
USE_FAKE_SUMMARIZER=false
USE_FAKE_SENTIMENT=false

VLLM_API_KEY=demo-key
SUMMARY_MODEL_NAME=summary-sft
SUMMARY_MODEL_VERSION=summary-sft
```

Set `VLLM_BASE_URL` according to the deployment layout described in [`06_configuration.md`](06_configuration.md). 

After changing `.env`, recreate the API and worker so they receive the new values:

```bash
docker compose -f infra/compose/docker-compose.yml \
  up -d --build --force-recreate api worker
```

## 4. Start the application stack

From the repository root:

```bash
make compose-up
```

This command builds and starts the Docker Compose services and applies the Alembic migrations.

Check their status:

```bash
docker compose -f infra/compose/docker-compose.yml ps
```

The API, worker, Postgres, and Redis should be running. Postgres and Redis should report healthy status.

The API documentation is available at:

```text
http://localhost:8000/docs
```

MLflow is available at:

```text
http://localhost:5000
```

## 5. Load serving data into Postgres

On the first run with a new database volume:

```bash
make load-serving-data
```

The loader reads:

```text
data/serving/appliances_demo_catalog.parquet
data/serving/appliances_demo_reviews.parquet
```

and inserts the product and review records into Postgres.

Run the loader once for the database. Re-inserting an already populated database may produce duplicate primary-key errors.

To rebuild the local database from an empty volume:

```bash
make compose-reset
make compose-up
make load-serving-data
```

## 6. Verify the basic API path

Run the project smoke test:

```bash
make smoke
```

The equivalent requests are:

```bash
curl -fsS http://localhost:8000/health
curl -fsS "http://localhost:8000/products?limit=5&offset=0"
```

Expected health response:

```json
{"status":"ok"}
```

Search products by title, store, category, or other indexed text:

```bash
curl -fsS \
  "http://localhost:8000/products?query=coffee&limit=10&offset=0"
```

Read one product:

```bash
curl -fsS http://localhost:8000/products/B00005QTXI
```

A `404` response means that the product is not present in the loaded serving dataset.

## 7. Start vLLM for real inference

This step is not required in mock mode.

Start or restart the local vLLM server:

```bash
make restart-vllm
```

Follow the startup log:

```bash
tail -f artifacts/logs/vllm.log
```

Verify the OpenAI-compatible endpoint:

```bash
curl -fsS http://localhost:8384/v1/models \
  -H "Authorization: Bearer demo-key" \
  | python3 -m json.tool
```

The response should include the base model and/or the served LoRA name used by:

```env
SUMMARY_MODEL_NAME=summary-sft
```

Test connectivity from the worker:

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

## 8. Generate a product insight

Insight generation is asynchronous. The first request creates an RQ job instead of waiting for sentiment inference and LLM generation to finish.

Create a job:

```bash
curl -fsS -X POST \
  http://localhost:8000/products/B00005QTXI/insights/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "max_reviews": 100,
    "representative_k": 5,
    "regenerate": true
  }'
```

Example response:

```json
{
  "job_id": "aa03c67f-e341-4094-a9ab-977ffeb987b5",
  "status": "queued",
  "product_id": "B00005QTXI",
  "cached": false,
  "message": "Insight generation job enqueued."
}
```

Request fields:

| Field | Meaning |
|---|---|
| `max_reviews` | Maximum number of reviews selected for sentiment processing. |
| `representative_k` | Maximum representative reviews used per sentiment group. |
| `regenerate` | When `true`, enqueue a new generation job even if a saved insight already exists. |

## 9. Poll the job

Use the returned `job_id`:

```bash
curl -fsS \
  http://localhost:8000/jobs/aa03c67f-e341-4094-a9ab-977ffeb987b5 \
  | python3 -m json.tool
```

Common states are:

```text
queued
started
finished
failed
```

When the job finishes, `result` contains the generated insight. When it fails, `error` contains the worker traceback.

A successful result includes:

```text
product_info
sentiment_distribution
representative_reviews
prompts
summaries
selected_review_count
total_available_reviews
latency_ms
model_version
```

The summaries are generated separately for the positive, neutral, and negative review groups when enough reviews are available for those groups.

## 10. Read the saved insight

After a successful job:

```bash
curl -fsS \
  http://localhost:8000/products/B00005QTXI/insights \
  | python3 -m json.tool
```

The read path checks Redis first. On a cache miss, it loads the saved payload from Postgres and writes it back to Redis.

When `regenerate=false`, creating another job for the same product and model version may return:

```text
status=already_exists
```

This avoids unnecessary model inference. Use `regenerate=true` when a new result is required.

## 11. Check metrics

Prometheus-format metrics are exposed at:

```bash
curl -fsS http://localhost:8000/metrics
```

The application records API request counts and latency. It also reports insight job outcomes, job duration, queue depth, and sentiment-source counts.

## 12. Stop the application

Stop the Docker Compose services while keeping their volumes:

```bash
make compose-down
```

Stop vLLM separately:

```bash
make stop-vllm
```

Remove the local Postgres and Redis volumes:

```bash
make compose-reset
```

Use `compose-reset` only when a clean database and cache are required.

## 13. Recommended validation flows

### Backend validation without GPU

```bash
# .env:
# USE_FAKE_SUMMARIZER=true
# USE_FAKE_SENTIMENT=true

make compose-reset
make compose-up
make load-serving-data
make smoke
```

Then create an insight job, poll it, and read the saved result.

### Full application validation

```bash
# .env:
# USE_FAKE_SUMMARIZER=false
# USE_FAKE_SENTIMENT=false

make compose-reset
make compose-up
make load-serving-data
make restart-vllm
make smoke
```

Then:

```text
POST /products/{product_id}/insights/jobs
GET  /jobs/{job_id}
GET  /products/{product_id}/insights
GET  /metrics
```

This validates the complete runtime path:

```text
FastAPI
→ Redis/RQ
→ worker
→ Postgres reviews
→ sentiment model
→ vLLM summary model
→ Postgres persistence
→ Redis cache
→ API response
```
