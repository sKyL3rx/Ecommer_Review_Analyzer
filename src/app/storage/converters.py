from typing import Any

import pandas as pd

from src.app.storage.models import Product, Review


def product_to_info(product: Product) -> dict[str, Any]:
    return {
        "product_id": product.product_id,
        "product_title": product.product_title or "",
        "store": product.store or "",
        "main_category": product.main_category or "",
        "price": product.price or "",
        "average_rating": product.average_rating,
        "rating_number": product.rating_number or 0,
        "review_count": product.review_count or 0,
        "avg_review_rating": product.avg_review_rating or 0.0,
        "verified_review_count": product.verified_review_count or 0,
        "total_helpful_votes": product.total_helpful_votes or 0,
        "latest_review_ts": product.latest_review_ts,
        "image_url": product.image_url or "",
        "categories_list": product.categories_list or [],
        "features_text": product.features_text or "",
        "description_text": product.description_text or "",
        "search_text": product.search_text or "",
    }


def reviews_to_dataframe(reviews: list[Review]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_id": r.review_id,
                "product_id": r.product_id,
                "review_title": r.review_title or "",
                "review_text": r.review_text or "",
                "review_char_len": r.review_char_len or len(r.review_text or ""),
                "timestamp": r.timestamp,
                "review_datetime": r.review_datetime,
                "helpful_vote": r.helpful_vote or 0,
                "verified_purchase": bool(r.verified_purchase)
                if r.verified_purchase is not None
                else False,
                "has_review_image": bool(r.has_review_image)
                if r.has_review_image is not None
                else False,
            }
            for r in reviews
        ]
    )