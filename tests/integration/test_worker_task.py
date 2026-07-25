from src.app.workers import insight_tasks
from src.app.workers.insight_tasks import generate_product_insight_task
from tests.fixtures.demo_data import DEMO_PRODUCT_ID


def test_generate_product_insight_task_with_fake_summarizer(
    seeded_product,
    monkeypatch,
):
    monkeypatch.setattr(insight_tasks.settings, "use_fake_summarizer", True)

    result = generate_product_insight_task(
        product_id=DEMO_PRODUCT_ID,
        max_reviews=10,
        representative_k=2,
        regenerate=True,
    )

    assert result["product_id"] == DEMO_PRODUCT_ID
