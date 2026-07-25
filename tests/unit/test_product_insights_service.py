import pandas as pd

from src.app.services.generate_product_insights import (
    add_quality_score,
    clean_text,
    compute_length_score,
    snippet,
)


def test_clean_text_handles_none_and_whitespace():
    assert clean_text(None) == ""
    assert clean_text("  hello  ") == "hello"


def test_snippet_truncates_long_text():
    text = "a" * 300

    result = snippet(text, max_len=20)

    assert len(result) <= 23
    assert result.endswith("...")


def test_compute_length_score_boundaries():
    assert compute_length_score(0) == 0.0
    assert compute_length_score(20) >= 0.0
    assert compute_length_score(100) >= 0.0


def test_add_quality_score_adds_quality_column():
    df = pd.DataFrame(
        [
            {
                "review_id": "R1",
                "review_text": "This is a useful product with decent build quality.",
                "helpful_vote": 5,
                "verified_purchase": True,
                "review_char_len": 60,
            }
        ]
    )

    scored = add_quality_score(df)

    assert "review_quality_score" in scored.columns
