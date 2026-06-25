from tests.fixtures.demo_data import DEMO_PRODUCT_ID


def test_list_products_returns_seeded_product(client, seeded_product):
    response = client.get(f"/products?query={DEMO_PRODUCT_ID}&limit=5&offset=0")

    assert response.status_code == 200

    body = response.json()
    assert body["total"] >= 1

    product_ids = {item["product_id"] for item in body["items"]}
    assert DEMO_PRODUCT_ID in product_ids


def test_list_products_respects_limit(client, seeded_product):
    response = client.get("/products?limit=1&offset=0")

    assert response.status_code == 200

    body = response.json()
    assert len(body["items"]) <= 1
    assert body["total"] >= len(body["items"])


def test_get_product_by_id(client, seeded_product):
    response = client.get(f"/products/{DEMO_PRODUCT_ID}")

    assert response.status_code == 200
    assert response.json()["product_id"] == DEMO_PRODUCT_ID


def test_get_product_not_found(client):
    response = client.get("/products/DOES_NOT_EXIST")

    assert response.status_code == 404
