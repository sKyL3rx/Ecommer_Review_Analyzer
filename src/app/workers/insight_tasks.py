from src.app.clients.sentiment_client import InferenceSentiment
from src.app.clients.summarizer_client import VLLMSummaryGenerator
from src.app.core.config import settings
from src.app.services.generate_product_insights import ProductInsightsService

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
        model=settings.app_model_version,
    )

    service = ProductInsightsService(
        sentiment_predictor=sentiment_model,
        summary_generator=summarizer_client,
        model_version=settings.app_model_version,
    )

    return service.generate(
        product_id=product_id,
        max_reviews=max_reviews,
        representative_k=representative_k,
        regenerate=regenerate,
    )