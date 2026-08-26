# Installation

## 1. Prerequisites

Install the following tools before setting up the repository:

- Git
- Make
- Python 3.11
- Docker Engine or Docker Desktop
- DVC

For real local LLM inference, an NVIDIA GPU with a working CUDA driver is also required.

Verify the basic tools:

```bash
git --version
make --version
python3.11 --version
docker --version
docker compose version
dvc --version
```

If you want to run vLLM locally, also check the GPU:

```bash
nvidia-smi
```

## 2. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd Ecommer_Review_Analyzer
```

Create the local configuration file and expected directories:

```bash
cp .env.example .env

mkdir -p \
  data/raw \
  data/interim \
  data/processed \
  data/serving \
  artifacts/models/final/sentiment \
  artifacts/models/final/summary_sft \
  artifacts/reports \
  artifacts/predictions \
  artifacts/benchmarks \
  artifacts/logs
```

## 3. Create the data and training environment

The main environment is used for:

- data download and preprocessing,
- sentiment training and evaluation,
- representative review selection,
- synthetic summary generation,
- TRL/SFT dataset preparation,
- QLoRA/LoRA training,
- DVC commands,
- local tests and linting.

Create and activate the environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Verify the main ML dependencies:

```bash
python - <<'PY'
import dvc
import pandas
import sklearn
import torch
import transformers

print("training environment: OK")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

## 4. Create the serving environment

The serving environment contains the API, worker, Redis/RQ, PostgreSQL client, and vLLM dependencies.

```bash
python3.11 -m venv .venv-serving
source .venv-serving/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-serving.txt
```

Verify the backend dependencies:

```bash
python - <<'PY'
import fastapi
import pandas
import redis
import rq
import sqlalchemy

print("serving environment: OK")
PY
```

If vLLM and an NVIDIA GPU are available locally:

```bash
.venv-serving/bin/vllm --version
nvidia-smi
```

You can skip the GPU checks when using mock mode.

## 5. Configure environment variables

Start from the example configuration:

```bash
cp .env.example .env
```

For the first local run, the main values are:

```env
GEMINI_API_KEY=<YOUR_API_KEY>
OPENAI_API_KEY=<YOUR_API_KEY>

DATABASE_URL=postgresql+psycopg://ecom:ecom@localhost:5432/ecom_review
REDIS_URL=redis://localhost:6379/0

VLLM_API_KEY=demo-key

SUMMARY_MODEL_NAME=summary-sft
SUMMARY_MODEL_VERSION=summary-sft
MODEL_VERSION=summary-sft

USE_FAKE_SUMMARIZER=true
USE_FAKE_SENTIMENT=true

AUTO_CREATE_TABLES=false
REDIS_CACHE_TTL_SECONDS=3600
```

Mock mode is the easiest way to test the application first. It validates FastAPI, PostgreSQL, Redis, RQ workers, job handling, and caching without requiring a GPU or vLLM server.

API keys are only required for stages that call an external model provider, such as synthetic summary generation or LLM-based evaluation.

More configuration options are described in [`06_configuration.md`](06_configuration.md).

## 6. Download the pretrained models

The serving application uses two trained model artifacts:

1. a TF-IDF + Logistic Regression sentiment classifier,
2. a LoRA adapter fine-tuned from `Qwen/Qwen2.5-3B-Instruct` for review summarization.

The model files can also be restored using DVC in the next section. The Google Drive links are provided for users who want to run the trained application without reproducing the full training pipeline.

### 6.1 Sentiment model

The sentiment model contains:

```text
artifacts/models/final/sentiment/
├── tfidf.joblib
└── sentiment_model.joblib
```

Download the final sentiment model:

**[Download final sentiment model](SENTIMENT_GOOGLE_DRIVE_LINK)**

Extract the files into:

```text
artifacts/models/final/sentiment/
```

After extraction, the directory should contain:

```text
artifacts/models/final/sentiment/
├── tfidf.joblib
└── sentiment_model.joblib
```

If these files are already included after cloning the repository, this step can be skipped.

Verify them with:

```bash
ls -lh \
  artifacts/models/final/sentiment/tfidf.joblib \
  artifacts/models/final/sentiment/sentiment_model.joblib
```

### 6.2 Summary model

The summary model is a LoRA adapter fine-tuned from:

```text
Qwen/Qwen2.5-3B-Instruct
```

Download the final adapter:

**[Download final Qwen2.5-3B LoRA adapter](https://drive.google.com/drive/folders/1JEREyOcIqMnIYY6jbkBZmfgyx2_GNvKF?usp=drive_link)**

Extract it into:

```text
artifacts/models/final/summary_sft/
```

The directory should contain files such as:

```text
artifacts/models/final/summary_sft/
├── adapter_config.json
├── adapter_model.safetensors
└── ...
```

The download contains the fine-tuned LoRA adapter only.

The base model is:

```text
Qwen/Qwen2.5-3B-Instruct
```

When real summary inference is enabled, vLLM loads the base model and attaches the LoRA adapter.

### 6.3 Model artifact options

There are two ways to restore the trained models:

- use the Google Drive links above for a quick local setup,
- use DVC to restore the versioned project artifacts from the configured remote.

The Google Drive copies are provided for convenience. The project pipeline itself uses DVC for artifact versioning.

## 7. Restore DVC artifacts

The project uses DVC to version large datasets and model artifacts.

Activate the main environment:

```bash
source .venv/bin/activate
```

Check the current DVC state:

```bash
dvc status
```

After configuring access to the project DVC remote, restore the artifacts:

```bash
dvc pull
```

Expected serving datasets include:

```text
data/serving/appliances_demo_catalog.parquet
data/serving/appliances_demo_reviews.parquet
```

Final model artifacts are expected under:

```text
artifacts/models/final/sentiment/
artifacts/models/final/summary_sft/
```

For users without access to the configured DVC remote, use the public model downloads from the previous section.

The full datasets and model artifacts can also be reproduced from the pipeline described in [`03_data_ml_pipeline.md`](03_data_ml_pipeline.md).

## 8. Validate the installation

Check that the repository commands are available:

```bash
make help
```

Validate the Docker Compose configuration without starting the services:

```bash
docker compose -f infra/compose/docker-compose.yml config >/dev/null

echo "docker compose configuration: OK"
```

Run the unit tests:

```bash
source .venv/bin/activate

PYTHONPATH=. python -m pytest tests/unit -q
```

Run lint checks:

```bash
source .venv/bin/activate

ruff check src scripts tests
```

If these commands pass, the repository is ready to run.

## Next steps

Choose the workflow you want to use next:

- [`03_data_ml_pipeline.md`](03_data_ml_pipeline.md) — reproduce preprocessing, training, and evaluation.
- [`06_configuration.md`](06_configuration.md) — configure mock mode or real inference.
- [`07_running_the_application.md`](07_running_the_application.md) — start PostgreSQL, Redis, API, worker, MLflow, and model serving.
- [`09_gke_deployment.md`](09_gke_deployment.md) — deploy the application and GPU serving stack on GKE.