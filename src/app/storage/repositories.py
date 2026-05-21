from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.app.storage.models import Product, ProductInsight, Review


def search_products(
    session: Session,
    query: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[Product]:
    
    stmt = select(Product)
    if query:
        pattern = f"%{query}%"
        stmt = stmt.where(
            or_(
                Product.product_id.ilike(pattern),
                Product.product_title.ilike(pattern),
                Product.store.ilike(pattern),
                Product.main_category.ilike(pattern),
                Product.search_text.ilike(pattern),
            )
        )
    
    stmt = stmt.offset(offset).limit(limit)
    return list(session.execute(stmt).scalars().all())

def get_product_by_id(
    session: Session,
    product_id: str,
) -> Product | None:
    stmt = select(Product).where(Product.product_id == product_id)
    return session.execute(stmt).scalar_one_or_none()

def get_reviews_for_product(
    session: Session,
    product_id: str,
    limit: int = 10_000,
) -> list[Review]:
    stmt = (
        select(Review)
        .where(Review.product_id == product_id)
        .limit(limit)
    )
    return list(session.execute(stmt).scalars().all())
def save_product_insight(
    session: Session,
    product_id: str,
    model_version: str,
    payload: dict,
) -> None:
    stmt = insert(ProductInsight).values(
        product_id=product_id,
        model_version=model_version,
        payload=payload,
    )

    stmt = stmt.on_conflict_do_update(
        index_elements=["product_id", "model_version"],
        set_={
            "payload": payload,
        },
    )

    session.execute(stmt)
    session.commit()

def get_product_insight(
    session: Session,
    product_id: str,
    model_version: str,
) -> ProductInsight | None:
    stmt = select(ProductInsight).where(
        ProductInsight.product_id == product_id,
        ProductInsight.model_version == model_version,
    )
    return session.execute(stmt).scalar_one_or_none()
