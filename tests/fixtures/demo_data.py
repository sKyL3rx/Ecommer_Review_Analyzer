from __future__ import annotations

from src.app.storage.models import Product, Review

DEMO_PRODUCT_ID = "DEMO_AIR_FRYER_001"


def seed_demo_product(session):
    product = Product(
        product_id=DEMO_PRODUCT_ID,
        product_title="Demo Air Fryer",
        store="Demo Store",
        main_category="Appliances",
        price="$79.99",
        average_rating=4.5,
        rating_number=120,
        review_count=3,
        image_url=None,
        search_text="demo air fryer appliance kitchen",
    )
    session.merge(product)

    reviews = [
        Review(
            review_id="DEMO_R1",
            product_id=DEMO_PRODUCT_ID,
            review_title="Easy to use",
            review_text="This air fryer is easy to use and clean.",
            review_char_len=43,
            rating=5.0,
            helpful_vote=5,
            verified_purchase=True,
            predicted_sentiment="positive",
            sentiment_confidence=0.95,
        ),
        Review(
            review_id="DEMO_R2",
            product_id=DEMO_PRODUCT_ID,
            review_title="Good value",
            review_text="Good value for a small kitchen.",
            review_char_len=31,
            rating=5.0,
            helpful_vote=2,
            verified_purchase=True,
            predicted_sentiment="positive",
            sentiment_confidence=0.9,
        ),
        Review(
            review_id="DEMO_R3",
            product_id=DEMO_PRODUCT_ID,
            review_title="A bit loud",
            review_text="It works well but the fan is a bit loud.",
            review_char_len=41,
            rating=5.0,
            helpful_vote=1,
            verified_purchase=True,
            predicted_sentiment="neutral",
            sentiment_confidence=0.75,
        ),
    ]

    for review in reviews:
        session.merge(review)

    session.commit()
    return product
