from src.app.services.generate_product_insights import all_positive_sentiment_predictor
from src.app.workers import insight_tasks


def test_build_sentiment_predictor_fake_when_enabled(monkeypatch):
    monkeypatch.setattr(insight_tasks.settings, "use_fake_sentiment", True)

    predictor = insight_tasks.build_sentiment_predictor()

    assert predictor is all_positive_sentiment_predictor
