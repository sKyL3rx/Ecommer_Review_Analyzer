FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir "dvc[gs]"

COPY .dvc/config .dvc/config
COPY artifacts/models/summary_sft.dvc artifacts/models/summary_sft.dvc
