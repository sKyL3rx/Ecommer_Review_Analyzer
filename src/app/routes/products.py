from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.app.schemas import ProductCard, ProductListResponse
from src.app.storage.db import get_session
from src.app.storage.repositories import count_products, get_product_by_id, search_products

router = APIRouter(tags=["products"])


def _to_product_card(product) -> ProductCard:
    return ProductCard(
        product_id=product.product_id,
        product_title=product.product_title or "",
        store=product.store,
        main_category=product.main_category,
        price=product.price,
        average_rating=product.average_rating,
        rating_number=product.rating_number,
        review_count=product.review_count,
        image_url=product.image_url,
    )


@router.get("/products", response_model=ProductListResponse)
def list_products(
    query: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> ProductListResponse:

    products = search_products(
        session=session,
        query=query,
        limit=limit,
        offset=offset,
    )

    total = count_products(
        session=session,
        query=query,
    )

    items = [_to_product_card(product) for product in products]

    return ProductListResponse(
        items=items,
        total=total,
    )


@router.get("/products/{product_id}", response_model=ProductCard)
def get_product(
    product_id: str,
    session: Session = Depends(get_session),
) -> ProductCard:
    product = get_product_by_id(
        session=session,
        product_id=product_id,
    )

    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")

    return _to_product_card(product)
