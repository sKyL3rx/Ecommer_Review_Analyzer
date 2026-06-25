import pandas as pd

from src.app.services.generate_product_insights import ProductInsightsService


def test_uses_precomputed_sentiment_without_model_call():
    calls = {"count": 0}

    def predictor(texts):
        calls["count"] += len(texts)
        return [{"sentiment_label": "negative", "sentiment_confidence": 0.5} for _ in texts]

    service = ProductInsightsService(sentiment_predictor=predictor)

    df = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "review_text": "Already labeled.",
                "predicted_sentiment": "positive",
                "sentiment_confidence": 0.91,
                "rating": None,
            }
        ]
    )

    result = service._predict_sentiment(df)

    assert calls["count"] == 0
    assert result.iloc[0]["sentiment_label"] == "positive"
    assert result.iloc[0]["sentiment_confidence"] == 0.91
    assert result.iloc[0]["sentiment_source"] == "precomputed"


def test_use_rating_without_model_call():
    calls = {"count": 0}

    def predictor(texts):
        calls["count"] += len(texts)
        return [{"sentiment_label": "negative", "sentiment_confidence": 0.5} for _ in texts]

    service = ProductInsightsService(sentiment_predictor=predictor)

    df = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "review_text": "Great.",
                "predicted_sentiment": None,
                "sentiment_confidence": None,
                "rating": 5,
            },
            {
                "review_id": "R2",
                "review_text": "Okay.",
                "predicted_sentiment": None,
                "sentiment_confidence": None,
                "rating": 3,
            },
            {
                "review_id": "R3",
                "review_text": "Bad.",
                "predicted_sentiment": None,
                "sentiment_confidence": None,
                "rating": 1,
            },
        ]
    )

    result = service._predict_sentiment(df)

    assert calls["count"] == 0
    assert result["sentiment_label"].tolist() == ["positive", "neutral", "negative"]
    assert result["sentiment_source"].tolist() == ["rating", "rating", "rating"]


def test_calls_model_only_for_missing_sentiment_rows():
    calls = {"texts": []}

    def predictor(texts):
        calls["texts"].extend(texts)
        return [{"sentiment_label": "negative", "sentiment_confidence": 0.88} for _ in texts]

    service = ProductInsightsService(sentiment_predictor=predictor)

    df = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "review_text": "Has rating.",
                "predicted_sentiment": None,
                "sentiment_confidence": None,
                "rating": 5,
            },
            {
                "review_id": "R2",
                "review_text": "Needs model.",
                "predicted_sentiment": None,
                "sentiment_confidence": None,
                "rating": None,
            },
        ]
    )

    result = service._predict_sentiment(df)

    assert calls["texts"] == ["Needs model."]
    assert result["sentiment_label"].tolist() == ["positive", "negative"]
    assert result["sentiment_source"].tolist() == ["rating", "model"]
