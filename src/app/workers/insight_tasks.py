from src.app.clients.sentiment_client import InferenceSentiment
from src.app.clients.summarizer_client import VLLMSummaryGenerator
from src.app.core.config import settings
from src.app.services.generate_product_insights import ProductInsightsService
from src.app.storage.cache import product_insight_cache_key, set_json_cache
from src.app.storage.converters import product_to_info, reviews_to_dataframe
from src.app.storage.db import SessionLocal
from src.app.storage.repositories import (
    get_product_by_id,
    get_reviews_for_product,
    save_product_insight,
)


def generate_product_insight_task(
    product_id: str,
    max_reviews: int = 100,
    representative_k: int = 5,
    regenerate: bool = False,
) -> dict:
    
    sentiment_model = InferenceSentiment()

    summarizer_client = VLLMSummaryGenerator(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.summary_model_name,
    )

    service = ProductInsightsService(
        sentiment_predictor=sentiment_model,
        summary_generator=summarizer_client,
        model_version=settings.summary_model_version,
    )

    with SessionLocal() as session:
        product = get_product_by_id(
            session=session,
            product_id=product_id,
        )

        if product is None:
            raise ValueError(f"Product not found in Postgres: {product_id}")

        reviews = get_reviews_for_product(
            session=session,
            product_id=product_id,
            limit=10_000,
        )

        if not reviews:
            raise ValueError(f"No reviews found in Postgres for product_id={product_id}")
    
        product_info = product_to_info(product)

        reviews_df = reviews_to_dataframe(reviews)

    print(f"Loaded {len(reviews_df)} reviews from Postgres for product_id={product_id}")


    result = service.generate(
        product_id=product_id,
        max_reviews=max_reviews,
        representative_k=representative_k,
        regenerate=regenerate,
        product_info=product_info,
        reviews_df=reviews_df,
        use_file_cache=False,
    )

    with SessionLocal() as session:
        save_product_insight(
            session=session,
            product_id=product_id,
            model_version=settings.summary_model_version,
            payload=result,
        )

    cache_key = product_insight_cache_key(
    product_id=product_id,
    model_version=settings.summary_model_version,
    )

    set_json_cache(
        key=cache_key,
        value=result,
        ttl_seconds=3600,
    )
    
    return result