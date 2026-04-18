#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/Ecommer_Review_Analyzer
source .venv/bin/activate

echo "[1/4] Train model..."
accelerate launch src/training/train_sft.py

echo "[2/4] Track trained artifact with DVC..."
dvc add artifacts/models/summary_sft

echo "[3/4] Push artifact to DVC remote..."
dvc push

echo "[4/4] Push .dvc metadata to Git..."
git add artifacts/models/summary_sft.dvc
git commit -m "update trained artifact" || true
git push origin main

echo "[DONE]"