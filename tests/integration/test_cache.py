from src.app.storage.cache import (
    delete_cache,
    get_json_cache,
    product_insight_cache_key,
    set_json_cache,
)


def test_product_insight_cache_roundtrip():
    key = product_insight_cache_key("DEMO_PRODUCT", "summary-sft")

    payload = {"product_id": "DEMO_PRODUCT", "summaries": {"positive": "Good."}}

    delete_cache(key)
    assert get_json_cache(key) is None

    set_json_cache(key, payload, ttl_seconds=60)

    assert get_json_cache(key) == payload

    delete_cache(key)
    assert get_json_cache(key) is None
