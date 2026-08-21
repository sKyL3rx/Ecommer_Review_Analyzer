# Installation

## 1. Prerequisites
Install the following tools before setting up the repository:

- Git
- Make
- Python 3.11
- Docker Engine or Docker Desktop
- DVC 

Verify the basic tools:

```bash
git --version
make --version
python3.11 --version
docker --version
docker compose version
```

## 2. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd Ecommer_Review_Analyzer
```

Create the local configuration file and expected artifact directories:

```bash
cp .env.example .env

mkdir -p \
  data/raw \
  data/interim \
  data/processed \
  data/serving \
  artifacts/models \
  artifacts/reports \
  artifacts/predictions \
  artifacts/benchmarks \
  artifacts/logs
```

## 3. Create the data and training environment
- data download and preprocessing,
- sentiment training and evaluation,
- synthetic summary dataset generation,
- TRL/SFT data preparation,
- LoRA training,
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
and
```bash
.venv-serving/bin/vllm --version
nvidia-smi
```

## 5. Configure environment variables

Start with the safe local defaults in `.env`:

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

Mock mode is the recommended first setup because it validates the API, database, Redis queue, worker, and cache without requiring a GPU or model server.

API keys are only needed for the stages that call an external model provider, such as synthetic summary generation or validating summary generation.

## 6. Restore DVC artifacts

The repository uses DVC for large datasets and model artifacts. After configuring access to the remote storage, check the repository state:

```bash
source .venv/bin/activate
dvc status
```

Pull the artifacts required by the current branch:

```bash
dvc pull
```

Expected serving files include:

```text
data/serving/appliances_demo_catalog.parquet
data/serving/appliances_demo_reviews.parquet
```

Expected summary model files are stored under:

```text
artifacts/models/summary_sft/
```

If remote access is not available, the same artifacts can be reproduced from the stages described in `03_data_ml_pipeline.md`.

## 7. Validate the installation

Check that the repository commands are available:

```bash
make help
```

Validate the Docker Compose file without starting services:

```bash
docker compose -f infra/compose/docker-compose.yml config >/dev/null
echo "docker compose configuration: OK"
```

Run the unit tests from the training environment:

```bash
source .venv/bin/activate
PYTHONPATH=. python -m pytest tests/unit -q
```

Run lint checks:

```bash
source .venv/bin/activate
ruff check src scripts tests
```

At this point, the repository is installed and ready for one of the next workflows:

- `03_data_ml_pipeline.md` — reproduce preprocessing, training, and evaluation,
- `06_configuration.md` — configure mock or real inference,
- `07_run_all_services.md` — start Postgres, Redis, API, worker, MLflow, and model serving.