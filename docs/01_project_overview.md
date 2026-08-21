# Project Overview

## What this project is

**Ecommerce Review Analyzer** is an end-to-end ML application for turning product reviews into structured product insights.

The project starts from raw ecommerce review data, builds sentiment and summarization datasets, trains/evaluates models, and serves the final results through a FastAPI application. The main feature is simple: given a `product_id`, the system groups reviews by sentiment and generates concise summaries of recurring positive, neutral, and negative themes.

# Main problem

Product pages often have hundreds or thousands of reviews. A user or seller may want to quickly understand:

- what customers repeatedly like,
- what customers repeatedly complain about,
- whether neutral reviews mention concrete trade-offs,
- and which themes are actually supported by the reviews.

## What the system does

For each product, the app can:

1. Search and return product metadata from Postgres.
2. Fetch reviews linked to a product.
3. Classify reviews into positive, neutral, and negative groups based on their reviews if they don't have provided senitments.
4. Select representative reviews from each sentiment group.
5. Send those selected reviews to a local vLLM OpenAI-compatible endpoint.
6. Generate short summaries using a summary LoRA model.
7. Store the generated insight in Postgres.
8. Cache the result in Redis for faster repeated reads.

## Repository structure

```text
.
├── src/
│   ├── app/          # FastAPI routes, services, RQ workers, DB access, clients
│   ├── data/         # Phase 1/2 preprocessing and TRL/SFT split creation
│   ├── training/     # Sentiment baseline and summary SFT training code
│   ├── evaluation/   # Sentiment and summary evaluation scripts
│   ├── serving/      # Builds serving-ready product/review parquet files
│   └── frontend/     # Streamlit Interfance
├── scripts/          # Data download, DB loading, synthetic summary generation, benchmarks
├── infra/
│   ├── compose/      # Local Docker Compose setup
│   └── k8s/          # Kubernetes deployment prototype
├── alembic/          # Postgres schema 
├── data/             # Raw/interim/processed/serving data artifacts
├── artifacts/        # Trained models, reports, predictions, benchmark outputs
├── docs/             # Project documentation
├── Makefile          # Common local commands
├── dvc.yaml          # Reproducible ML/data pipeline stages
├── params.yaml       # Pipeline and model configuration
└── requirements*.txt # Python dependencies for training and serving

