COMPOSE_FILE=infra/compose/docker-compose.yml
LOCAL_DATABASE_URL=postgresql+psycopg://ecom:ecom@localhost:5432/ecom_review
LOCAL_REDIS_URL=redis://localhost:6379/0
PYTHONPATH_VALUE=.

.PHONY: compose-up compose-up-no-build compose-down compose-reset mlflow-up load-serving-data smoke lint format test-unit test-integration ci-local docker-build benchmark-local

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

mlflow-up:
	docker compose -f $(COMPOSE_FILE) up -d mlflow

load-serving-data:
	PYTHONPATH="$(PYTHONPATH_VALUE)" \
	DATABASE_URL="$(LOCAL_DATABASE_URL)" \
	python -m scripts.load_serving_data_to_postgres \
		--catalog_path ./data/serving/appliances_demo_catalog.parquet \
		--reviews_path ./data/serving/appliances_demo_reviews.parquet

smoke:
	curl -f http://localhost:8000/health
	curl -f "http://localhost:8000/products?limit=5"

test-unit:
	PYTHONPATH="$(PYTHONPATH_VALUE)" \
	python -m pytest tests/unit -q

test-integration:
	PYTHONPATH="$(PYTHONPATH_VALUE)" \
	DATABASE_URL="$(LOCAL_DATABASE_URL)" \
	REDIS_URL="$(LOCAL_REDIS_URL)" \
	USE_FAKE_SUMMARIZER=true \
	USE_FAKE_SENTIMENT=true \
	AUTO_CREATE_TABLES=false \
	python -m pytest tests/integration -q

lint:
	ruff check src scripts tests

format:
	ruff check src scripts tests --fix
	ruff format src scripts tests

docker-build:
	docker build -t ecommerce-review-analyzer:ci .

ci-local: lint test-unit test-integration docker-build

benchmark-local:
	PYTHONPATH="$(PYTHONPATH_VALUE)" \
	python scripts/benchmark_local_serving.py \
		--api-base-url http://localhost:8000 \
		--product-limit 10 \
		--requests 100 \
		--jobs 10 \
		--max-reviews 50 \
		--representative-k 3 \
		--output artifacts/benchmarks/local_serving_benchmark.json