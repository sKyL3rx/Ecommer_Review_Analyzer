FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir "dvc[gs]"

COPY .dvc/config .dvc/config

COPY dvc.yaml dvc.yaml
COPY dvc.lock dvc.lock
COPY params.yaml params.yaml

COPY artifacts/models/final/sentiment.dvc artifacts/models/final/sentiment.dvc
COPY artifacts/models/final/summary_sft.dvc artifacts/models/final/summary_sft.dvc

RUN mkdir -p .dvc && \
    printf '\n[core]\n    no_scm = true\n' > .dvc/config.local