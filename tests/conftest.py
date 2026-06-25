from __future__ import annotations

import os

import pytest

os.environ.setdefault("USE_FAKE_SUMMARIZER", "true")
os.environ.setdefault("USE_FAKE_SENTIMENT", "true")
os.environ.setdefault("AUTO_CREATE_TABLES", "false")
os.environ.setdefault("SUMMARY_MODEL_VERSION", "summary-sft")


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from src.app.main import app

    with TestClient(app) as test_client:
        yield test_client
