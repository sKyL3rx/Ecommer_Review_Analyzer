from __future__ import annotations

import pytest
from sqlalchemy import text

from tests.fixtures.demo_data import seed_demo_product


@pytest.fixture()
def db_session():
    from src.app.storage.db import SessionLocal

    with SessionLocal() as session:
        yield session


@pytest.fixture()
def clean_test_rows(db_session):
    yield

    db_session.execute(text("DELETE FROM product_insights WHERE product_id LIKE 'DEMO_%'"))
    db_session.execute(text("DELETE FROM insight_jobs WHERE product_id LIKE 'DEMO_%'"))
    db_session.execute(text("DELETE FROM reviews WHERE product_id LIKE 'DEMO_%'"))
    db_session.execute(text("DELETE FROM products WHERE product_id LIKE 'DEMO_%'"))
    db_session.commit()


@pytest.fixture()
def seeded_product(db_session, clean_test_rows):
    return seed_demo_product(db_session)
