PYTHONPATH_VALUE ?= .

TRAIN_VENV ?= .venv
SERVING_VENV ?= .venv-serving

TRAIN_PYTHON_BOOTSTRAP ?= python3
SERVING_PYTHON_BOOTSTRAP ?= python3.13

TRAIN_PYTHON ?= $(TRAIN_VENV)/bin/python
SERVING_PYTHON ?= $(SERVING_VENV)/bin/python

TRAIN_BIN ?= $(TRAIN_VENV)/bin
SERVING_BIN ?= $(SERVING_VENV)/bin

TRAIN_ENV = PATH=$(TRAIN_BIN):$$PATH PYTHONPATH=$(PYTHONPATH_VALUE)
SERVING_ENV = PATH=$(SERVING_BIN):$$PATH PYTHONPATH=$(PYTHONPATH_VALUE) VLLM_USE_FLASHINFER_SAMPLER=0

TRAIN_DVC = $(TRAIN_ENV) $(TRAIN_PYTHON) -m dvc
SERVING_DVC = PATH=$(SERVING_BIN):$(TRAIN_BIN):$$PATH PYTHONPATH=$(PYTHONPATH_VALUE) $(TRAIN_PYTHON) -m dvc

VLLM_BIN ?= $(SERVING_BIN)/vllm

COMPOSE_FILE ?= infra/compose/docker-compose.yml
API_BASE_URL ?= http://localhost:8000


LOCAL_DATABASE_URL ?= postgresql+psycopg://ecom:ecom@localhost:5432/ecom_review
LOCAL_REDIS_URL ?= redis://localhost:6379/0

CATALOG_PATH ?= data/serving/appliances_demo_catalog.parquet
REVIEWS_PATH ?= data/serving/appliances_demo_reviews.parquet

VLLM_HOST ?= 0.0.0.0
VLLM_PORT ?= 8384
VLLM_BASE_URL ?= http://localhost:8384/v1
VLLM_API_KEY ?= demo-key
VLLM_BASE_MODEL ?= Qwen/Qwen2.5-3B-Instruct
VLLM_LORA_NAME ?= summary-sft
VLLM_LORA_DIR ?= artifacts/models/summary_sft
VLLM_MAX_LORA_RANK ?= 16
VLLM_LOG ?= artifacts/logs/vllm.log
VLLM_PID_FILE ?= artifacts/logs/vllm.pid
STREAMLIT_APP ?= src/frontend/streamlit_app.py

.PHONY: help init init-train init-serving init-full doctor \
	dvc-dag dvc-status clean-repro \
	download-data preprocess-phase1 train-sentiment eval-sentiment sentiment-e2e \
	repro-serving \
	phase2-groups synthetic-summary trl-splits summary-data-e2e \
	train-summary-sft \
	serve-vllm serve-vllm-bg wait-vllm stop-vllm restart-vllm \
	summary-predict summary-eval summary-vllm-e2e register-summary-lora \
	compose-up compose-down compose-reset migrate load-serving-data smoke \
	benchmark-local benchmark-cold-cache \
	bootstrap-local bootstrap-full full-e2e \
	running_frontend \
	test-unit test-integration lint format ci-local 

help:
	@echo "Setup:"
	@echo "  make init                      Install full dependencies into .venv and .venv-serving"
	@echo "  make init-train                Create .venv and install requirements.txt + requirements-dev.txt"
	@echo "  make init-serving              Create .venv-serving with Python 3.13 and install requirements-serving.txt"
	@echo "  make doctor                    Check required tools and envs"
	@echo ""
	@echo "DVC graph:"
	@echo "  make dvc-dag                   Show DVC DAG"
	@echo "  make dvc-status                Show DVC status"
	@echo ""
	@echo "Phase 1 sentiment/data:"
	@echo "  make download-data             Run download_raw_sample"
	@echo "  make preprocess-phase1         Run preprocess_phase1"
	@echo "  make train-sentiment           Run train_sentiment_baseline"
	@echo "  make eval-sentiment            Run evaluate_sentiment_baseline"
	@echo "  make sentiment-e2e             Download -> preprocess -> train -> eval sentiment"
	@echo ""
	@echo "Serving artifacts/API:"
	@echo "  make repro-serving             Build serving parquet artifacts using .venv-serving"
	@echo "  make compose-up                Start API/Postgres/Redis and run migrations"
	@echo "  make load-serving-data         Load serving parquet into Postgres using .venv-serving"
	@echo "  make smoke                     API smoke test"
	@echo ""
	@echo "Phase 2 / synthetic / TRL:"
	@echo "  make phase2-groups             Run preprocess_phase2_review_groups"
	@echo "  make synthetic-summary         Run generate_synthetic_gold_summary"
	@echo "  make trl-splits                Run build_trl_sft_splits"
	@echo "  make summary-data-e2e          Phase2 -> synthetic summaries -> TRL splits"
	@echo ""
	@echo "SFT + vLLM:"
	@echo "  make train-summary-sft         Train SFT/QLoRA LoRA adapter using .venv"
	@echo "  make serve-vllm                Run vLLM foreground using .venv"
	@echo "  make serve-vllm-bg             Run vLLM background using .venv"
	@echo "  make wait-vllm                 Wait for vLLM /models"
	@echo "  make stop-vllm                 Stop background vLLM"
	@echo "  make restart-vllm              Restart background vLLM"
	@echo "  make summary-predict           Generate test predictions via vLLM"
	@echo "  make summary-eval              Evaluate generated summaries"
	@echo "  make summary-vllm-e2e          vLLM ready -> predict -> eval"
	@echo ""
	@echo "Full E2E:"
	@echo "  make bootstrap-local           TRUE FULL E2E including vLLM"
	@echo "  make bootstrap-full            Alias for bootstrap-local"
	@echo "  make full-e2e                  Alias for bootstrap-local"
	@echo ""
	@echo "Benchmark:"
	@echo "  make benchmark-local"
	@echo "  make benchmark-cold-cache"

init: init-full

init-train:
	$(TRAIN_PYTHON_BOOTSTRAP) -m venv $(TRAIN_VENV)
	$(TRAIN_PYTHON) -m pip install --upgrade pip setuptools wheel
	$(TRAIN_PYTHON) -m pip install -r requirements.txt
	$(TRAIN_PYTHON) -m pip install -r requirements-dev.txt

init-serving:
	$(SERVING_PYTHON_BOOTSTRAP) -m venv $(SERVING_VENV)
	$(SERVING_PYTHON) -m pip install --upgrade pip setuptools wheel
	$(SERVING_PYTHON) -m pip install -r requirements-serving.txt
	$(SERVING_PYTHON) -m pip install -r requirements-vllm.txt

init-full: init-train init-serving

doctor:
	@echo "Training/E2E env:"
	$(TRAIN_PYTHON) --version
	$(TRAIN_PYTHON) -m pip --version
	$(TRAIN_PYTHON) -m dvc --version
	@echo ""
	@echo "Serving env:"
	$(SERVING_PYTHON) --version
	$(SERVING_PYTHON) -m pip --version
	@echo ""
	@echo "vLLM:"
	@if [ -x "$(VLLM_BIN)" ]; then \
		$(VLLM_BIN) --version || true; \
	else \
		echo "vLLM binary not found at $(VLLM_BIN). Install it in $(TRAIN_VENV) if full vLLM E2E is needed."; \
	fi
	@echo ""
	git --version
	docker --version
	docker compose version

dvc-dag:
	$(TRAIN_DVC) dag

dvc-status:
	$(TRAIN_DVC) status

clean-repro:
	rm -rf data/raw data/interim data/processed data/serving data/cache
	rm -rf artifacts/models artifacts/reports artifacts/predictions artifacts/eval artifacts/benchmarks artifacts/logs
	rm -f dvc.lock

# ---------- DVC: Phase 1 sentiment/data using .venv ----------

download-data:
	$(TRAIN_DVC) repro download_raw_sample

preprocess-phase1:
	$(TRAIN_DVC) repro preprocess_phase1

train-sentiment:
	$(TRAIN_DVC) repro train_sentiment_baseline

eval-sentiment:
	$(TRAIN_DVC) repro evaluate_sentiment_baseline

sentiment-e2e:
	$(TRAIN_DVC) repro evaluate_sentiment_baseline

# ---------- DVC: serving parquet artifacts using .venv-serving for stage commands ----------

repro-serving:
	$(TRAIN_DVC) repro download_serving_data
	$(SERVING_DVC) repro build_serving_catalog
	$(SERVING_DVC) repro build_serving_reviews

# ---------- Local API serving ----------

compose-up:
	docker compose -f $(COMPOSE_FILE) up --build -d
	docker compose -f $(COMPOSE_FILE) run --rm migrate

compose-up-no-build:
	docker compose -f $(COMPOSE_FILE) up -d
	docker compose -f $(COMPOSE_FILE) run --rm migrate

compose-down:
	docker compose -f $(COMPOSE_FILE) down

compose-reset:
	docker compose -f $(COMPOSE_FILE) down -v

migrate:
	docker compose -f $(COMPOSE_FILE) run --rm migrate

load-serving-data:
	$(SERVING_ENV) DATABASE_URL="$(LOCAL_DATABASE_URL)" \
	$(SERVING_PYTHON) -m scripts.load_serving_data_to_postgres \
		--catalog_path $(CATALOG_PATH) \
		--reviews_path $(REVIEWS_PATH)

smoke:
	curl -f $(API_BASE_URL)/health
	curl -f "$(API_BASE_URL)/products?limit=5"

# ---------- DVC: Phase 2 / synthetic / TRL using .venv ----------

phase2-groups:
	$(TRAIN_DVC) repro preprocess_phase2_review_groups

synthetic-summary:
	$(TRAIN_DVC) repro generate_synthetic_gold_summary

trl-splits:
	$(TRAIN_DVC) repro build_trl_sft_splits

summary-data-e2e:
	$(TRAIN_DVC) repro build_trl_sft_splits

# ---------- DVC: train SFT using .venv ----------

train-summary-sft:
	$(TRAIN_DVC) repro train_summary_sft

# ---------- vLLM LoRA serving using .venv ----------

serve-vllm:
	$(SERVING_ENV) $(VLLM_BIN) serve $(VLLM_BASE_MODEL) \
		--host $(VLLM_HOST) \
		--port $(VLLM_PORT) \
		--enable-lora \
		--lora-modules $(VLLM_LORA_NAME)=$(VLLM_LORA_DIR) \
		--max-lora-rank $(VLLM_MAX_LORA_RANK) \
		--api-key $(VLLM_API_KEY)

serve-vllm-bg:
	mkdir -p artifacts/logs
	@if [ -f "$(VLLM_PID_FILE)" ] && kill -0 $$(cat $(VLLM_PID_FILE)) 2>/dev/null; then \
		echo "vLLM already running with PID $$(cat $(VLLM_PID_FILE))"; \
	else \
		echo "Starting vLLM in background..."; \
		$(SERVING_ENV) nohup $(VLLM_BIN) serve $(VLLM_BASE_MODEL) \
			--host $(VLLM_HOST) \
			--port $(VLLM_PORT) \
			--enable-lora \
			--lora-modules $(VLLM_LORA_NAME)=$(VLLM_LORA_DIR) \
			--max-lora-rank $(VLLM_MAX_LORA_RANK) \
			--api-key $(VLLM_API_KEY) \
			> $(VLLM_LOG) 2>&1 & echo $$! > $(VLLM_PID_FILE); \
		echo "vLLM PID: $$(cat $(VLLM_PID_FILE))"; \
		echo "Logs: $(VLLM_LOG)"; \
	fi

wait-vllm:
	@echo "Waiting for vLLM at $(VLLM_BASE_URL)/models ..."
	@for i in $$(seq 1 180); do \
		if curl -fsS -H "Authorization: Bearer $(VLLM_API_KEY)" "$(VLLM_BASE_URL)/models" >/dev/null 2>&1; then \
			echo "vLLM is ready."; \
			exit 0; \
		fi; \
		sleep 5; \
	done; \
	echo "vLLM did not become ready. Check logs:"; \
	echo "  tail -f $(VLLM_LOG)"; \
	exit 1

stop-vllm:
	@if [ -f "$(VLLM_PID_FILE)" ]; then \
		kill $$(cat $(VLLM_PID_FILE)) 2>/dev/null || true; \
		rm -f $(VLLM_PID_FILE); \
		echo "Stopped vLLM."; \
	else \
		echo "No vLLM PID file found."; \
	fi

restart-vllm: stop-vllm serve-vllm-bg wait-vllm

# ---------- DVC: vLLM prediction + summary evaluation using .venv ----------

summary-predict: wait-vllm
	$(TRAIN_DVC) repro generate_summary_predictions

summary-eval:
	$(TRAIN_DVC) repro evaluate_summary_model

summary-vllm-e2e: wait-vllm
	$(TRAIN_DVC) repro evaluate_summary_model

register-summary-lora:
	$(TRAIN_DVC) repro register_summary_lora_to_mlflow

# ---------- Benchmarks using .venv-serving ----------

benchmark-local:
	$(SERVING_ENV) $(SERVING_PYTHON) scripts/benchmark_local_serving.py \
		--api-base-url $(API_BASE_URL) \
		--product-limit 10 \
		--requests 100 \
		--jobs 10 \
		--max-reviews 50 \
		--representative-k 3 \
		--output artifacts/benchmarks/local_serving_benchmark.json

benchmark-cold-cache:
	$(SERVING_ENV) $(SERVING_PYTHON) scripts/benchmark_local_serving.py \
		--api-base-url $(API_BASE_URL) \
		--product-limit 10 \
		--requests 100 \
		--jobs 10 \
		--max-reviews 50 \
		--representative-k 3 \
		--flush-redis-before-run \
		--output artifacts/benchmarks/local_serving_benchmark_cold_cache.json

# ---------- FULL E2E ----------
# Includes:
# download -> preprocess -> train/eval sentiment
# serving parquet -> compose API -> load DB -> smoke
# phase2 -> synthetic -> TRL
# train SFT -> start vLLM -> predict via vLLM -> eval summary
# benchmark API

bootstrap-local: sentiment-e2e repro-serving compose-up load-serving-data smoke summary-data-e2e train-summary-sft restart-vllm summary-predict summary-eval benchmark-local
	@echo "TRUE FULL E2E completed."
	@echo "API docs: $(API_BASE_URL)/docs"
	@echo "vLLM endpoint: $(VLLM_BASE_URL)"
	@echo "vLLM logs: $(VLLM_LOG)"

bootstrap-full: bootstrap-local

full-e2e: bootstrap-local

running_frontend: 
	$(SERVING_ENV) $(SERVING_PYTHON) -m streamlit run $(STREAMLIT_APP)



# ---------- Tests / quality ----------

test-unit:
	$(TRAIN_ENV) $(TRAIN_PYTHON) -m pytest tests/unit -q

test-integration:
	$(SERVING_ENV) \
	DATABASE_URL="$(LOCAL_DATABASE_URL)" \
	REDIS_URL="$(LOCAL_REDIS_URL)" \
	USE_FAKE_SUMMARIZER=true \
	USE_FAKE_SENTIMENT=true \
	AUTO_CREATE_TABLES=false \
	$(SERVING_PYTHON) -m pytest tests/integration -q

lint:
	$(TRAIN_ENV) $(TRAIN_PYTHON) -m ruff check src scripts tests

format:
	$(TRAIN_ENV) $(TRAIN_PYTHON) -m ruff check src scripts tests --fix
	$(TRAIN_ENV) $(TRAIN_PYTHON) -m ruff format src scripts tests

ci-local: lint test-unit