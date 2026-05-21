from fastapi.testclient import TestClient

from src.app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_metrics_endpoint():
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert "api_request_latency_seconds" in response.text
    assert "insight_queue_depth" in response.text


def test_openapi_available():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200

    body = response.json()
    assert body["info"]["title"] == "Product Review Intelligence API"