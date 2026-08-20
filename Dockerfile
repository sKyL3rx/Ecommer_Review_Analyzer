FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-serving.txt

COPY . .

RUN mkdir -p /app/artifacts/models

COPY artifacts/models/final/sentiment/tfidf.joblib \
     /app/artifacts/models/tfidf.joblib

COPY artifacts/models/final/sentiment/sentiment_model.joblib \
     /app/artifacts/models/sentiment_model.joblib

EXPOSE 8000