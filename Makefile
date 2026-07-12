PYTHON ?= python
PYTHONPATH_VALUE ?= .
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

.PHONY: help init doctor \
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
	test-unit test-integration lint format ci-local

help:
	@echo "Setup:"
	@echo "  make init                     Install local dependencies"
	@echo "  make doctor                   Check required tools"
	@echo ""
	@echo "DVC graph:"
	@echo "  make dvc-dag                  Show DVC DAG"
	@echo "  make dvc-status               Show DVC status"
	@echo ""
	@echo "Phase 1 sentiment/data:"
	@echo "  make download-data            Run download_raw_sample"
	@echo "  make preprocess-phase1        Run preprocess_phase1"
	@echo "  make train-sentiment          Run train_sentiment_baseline"
	@echo "  make eval-sentiment           Run evaluate_sentiment_baseline"
	@echo "  make sentiment-e2e            Download -> preprocess -> train -> eval sentiment"
	@echo ""
	@echo "Serving artifacts/API:"
	@echo "  make repro-serving            Build/validate serving parquet artifacts"
	@echo "  make compose-up               Start API/Postgres/Redis and run migrations"
	@echo "  make load-serving-data        Load serving parquet into Postgres"
	@echo "  make smoke                    API smoke test"
	@echo ""
	@echo "Phase 2 / synthetic / TRL:"
	@echo "  make phase2-groups            Run preprocess_phase2_review_groups"
	@echo "  make synthetic-summary        Run generate_synthetic_gold_summary"
	@echo "  make trl-splits               Run build_trl_sft_splits"
	@echo "  make summary-data-e2e         Phase2 -> synthetic summaries -> TRL splits"
	@echo ""
	@echo "SFT + vLLM:"
	@echo "  make train-summary-sft        Train SFT/QLoRA LoRA adapter"
	@echo "  make serve-vllm               Run vLLM foreground"
	@echo "  make serve-vllm-bg            Run vLLM background"
	@echo "  make wait-vllm                Wait for vLLM /models"
	@echo "  make stop-vllm                Stop background vLLM"
	@echo "  make restart-vllm             Restart background vLLM"
	@echo "  make summary-predict          Generate test predictions via vLLM"
	@echo "  make summary-eval             Evaluate generated summaries"
	@echo "  make summary-vllm-e2e         vLLM ready -> predict -> eval"
	@echo ""
	@echo "Full E2E:"
	@echo "  make bootstrap-local          FULL E2E including vLLM"
	@echo "  make bootstrap-full           Alias for bootstrap-local"
	@echo "  make full-e2e                 Alias for bootstrap-local"
	@echo ""
	@echo "Benchmark:"
	@echo "  make benchmark-local"
	@echo "  make benchmark-cold-cache"

init:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-serving.txt -r requirements-dev.txt

doctor:
	$(PYTHON) --version
	git --version
	dvc --version
	docker --version
	docker compose version

dvc-dag:
	dvc dag

dvc-status:
	dvc status

clean-repro:
	rm -rf data/raw data/interim data/processed data/serving data/cache
	rm -rf artifacts/models artifacts/reports artifacts/predictions artifacts/eval artifacts/benchmarks artifacts/logs
	rm -f dvc.lock

# ---------- DVC: Phase 1 sentiment/data ----------

download-data:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro download_raw_sample

preprocess-phase1:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro preprocess_phase1

train-sentiment:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro train_sentiment_baseline

eval-sentiment:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro evaluate_sentiment_baseline

sentiment-e2e:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro evaluate_sentiment_baseline

# ---------- DVC: serving parquet artifacts ----------

repro-serving:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro build_serving_catalog
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro build_serving_reviews

# ---------- Local API serving ----------

compose-up:
	docker compose -f $(COMPOSE_FILE) up --build -d
	docker compose -f $(COMPOSE_FILE) run --rm migrate

compose-down:
	docker compose -f $(COMPOSE_FILE) down

compose-reset:
	docker compose -f $(COMPOSE_FILE) down -v

migrate:
	docker compose -f $(COMPOSE_FILE) run --rm migrate

load-serving-data:
	PYTHONPATH=$(PYTHONPATH_VALUE) DATABASE_URL="$(LOCAL_DATABASE_URL)" \
	$(PYTHON) -m scripts.load_serving_data_to_postgres \
		--catalog_path $(CATALOG_PATH) \
		--reviews_path $(REVIEWS_PATH)

smoke:
	curl -f $(API_BASE_URL)/health
	curl -f "$(API_BASE_URL)/products?limit=5"

# ---------- DVC: Phase 2 / synthetic / TRL ----------

phase2-groups:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro preprocess_phase2_review_groups

synthetic-summary:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro generate_synthetic_gold_summary

trl-splits:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro build_trl_sft_splits

summary-data-e2e:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro build_trl_sft_splits

# ---------- DVC: train SFT ----------

train-summary-sft:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro train_summary_sft

# ---------- vLLM LoRA serving ----------

serve-vllm:
	vllm serve $(VLLM_BASE_MODEL) \
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
		nohup vllm serve $(VLLM_BASE_MODEL) \
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

# ---------- DVC: vLLM prediction + summary evaluation ----------

summary-predict: wait-vllm
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro generate_summary_predictions

summary-eval:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro evaluate_summary_model

summary-vllm-e2e: wait-vllm
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro evaluate_summary_model

register-summary-lora:
	PYTHONPATH=$(PYTHONPATH_VALUE) dvc repro register_summary_lora_to_mlflow

# ---------- Benchmarks ----------

benchmark-local:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/benchmark_local_serving.py \
		--api-base-url $(API_BASE_URL) \
		--product-limit 10 \
		--requests 100 \
		--jobs 10 \
		--max-reviews 50 \
		--representative-k 3 \
		--output artifacts/benchmarks/local_serving_benchmark.json

benchmark-cold-cache:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) scripts/benchmark_local_serving.py \
		--api-base-url $(API_BASE_URL) \
		--product-limit 10 \
		--requests 100 \
		--jobs 10 \
		--max-reviews 50 \
		--representative-k 3 \
		--flush-redis-before-run \
		--output artifacts/benchmarks/local_serving_benchmark_cold_cache.json

# ---------- FULL E2E ----------

bootstrap-local: sentiment-e2e repro-serving compose-up load-serving-data smoke summary-data-e2e train-summary-sft restart-vllm summary-predict summary-eval benchmark-local
	@echo "TRUE FULL E2E completed."
	@echo "API docs: $(API_BASE_URL)/docs"
	@echo "vLLM endpoint: $(VLLM_BASE_URL)"
	@echo "vLLM logs: $(VLLM_LOG)

bootstrap-full: bootstrap-local

full-e2e: bootstrap-local

# ---------- Tests / quality ----------

test-unit:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m pytest tests/unit -q

test-integration:
	PYTHONPATH=$(PYTHONPATH_VALUE) \
	DATABASE_URL="$(LOCAL_DATABASE_URL)" \
	REDIS_URL="$(LOCAL_REDIS_URL)" \
	USE_FAKE_SUMMARIZER=true \
	USE_FAKE_SENTIMENT=true \
	AUTO_CREATE_TABLES=false \
	$(PYTHON) -m pytest tests/integration -q

lint:
	ruff check src scripts tests

format:
	ruff check src scripts tests --fix
	ruff format src scripts tests

ci-local: lint test-unit