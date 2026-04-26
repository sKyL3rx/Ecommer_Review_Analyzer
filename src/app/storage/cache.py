import json
from typing import Any

from src.app.jobs.queue import redis_conn


def product_insight_cache_key(product_id: str, model_version: str) -> str:
    return f"product_insight:{product_id}:{model_version}"

def get_json_cache(key: str) -> dict[str, Any] | None:
    raw = redis_conn.get(key)

    if raw is None:
        return None

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    return json.loads(raw)


def set_json_cache(
    key: str,
    value: dict[str, Any],
    ttl_seconds: int = 3600,
) -> None:
    
    redis_conn.setex(
        key,
        ttl_seconds,
        json.dumps(value, ensure_ascii=False),
    )

def delete_cache(key: str) -> None:
    redis_conn.delete(key)


