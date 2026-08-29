from __future__ import annotations

import ast
import time
from typing import Any

import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"


def api_get(
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
) -> Any:
    url = f"{base_url.rstrip('/')}{path}"

    resp = requests.get(
        url,
        params=params,
        timeout=60,
    )

    resp.raise_for_status()
    return resp.json()


def api_post(
    base_url: str,
    path: str,
    payload: dict[str, Any],
) -> Any:
    url = f"{base_url.rstrip('/')}{path}"

    resp = requests.post(
        url,
        json=payload,
        timeout=300,
    )

    resp.raise_for_status()
    return resp.json()


def wait_for_insight_job(
    base_url: str,
    job_id: str,
    timeout_seconds: int = 300,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        job = api_get(
            base_url,
            f"/jobs/{job_id}",
        )

        status = str(job.get("status", "")).lower()

        if status in {
            "finished",
            "completed",
            "success",
        }:
            return job

        if status in {
            "failed",
            "error",
        }:
            raise RuntimeError(
                job.get("error")
                or job.get("message")
                or f"Insight job {job_id} failed."
            )

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Insight job {job_id} did not finish within "
        f"{timeout_seconds} seconds."
    )


def normalize_image_url(value: Any) -> str | None:
    """
    Normalize product image data into a single URL.
    """

    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            return None

        first = value[0]

        if isinstance(first, str):
            first = first.strip()

            if first.startswith(("http://", "https://")):
                return first

        return None

    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value:
        return None

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list) and parsed:
                first = parsed[0]

                if isinstance(first, str):
                    first = first.strip()

                    if first.startswith(("http://", "https://")):
                        return first

        except (ValueError, SyntaxError):
            return None

    if value.startswith(("http://", "https://")):
        return value

    return None


def render_product_card(product: dict[str, Any]) -> None:
    title = product.get("product_title") or "Untitled product"
    store = product.get("store") or "Unknown store"
    category = product.get("main_category") or "Unknown category"
    price = product.get("price") or "N/A"

    review_count = product.get("review_count", 0)
    avg_rating = product.get("average_rating")
    rating_number = product.get("rating_number", 0)

    image_url = normalize_image_url(
        product.get("image_url")
    )

    with st.container(border=True):
        image_col, info_col = st.columns(
            [1, 3],
            gap="large",
        )

        with image_col:
            if image_url:
                st.image(
                    image_url,
                    width="stretch",
                )
            else:
                st.caption("No image available")

        with info_col:
            st.subheader(title)

            st.write(f"**Store:** {store}")
            st.write(f"**Category:** {category}")
            st.write(f"**Price:** {price}")

            if avg_rating is not None:
                st.write(
                    f"**Catalog rating:** "
                    f"{float(avg_rating):.2f} "
                    f"({rating_number} ratings)"
                )
            else:
                st.write("**Catalog rating:** N/A")

            st.write(
                f"**Review count:** {review_count}"
            )


def render_sentiment_distribution(
    dist: dict[str, Any],
) -> None:
    counts = dist.get("counts", {})
    ratios = dist.get("ratios", {})

    positive_count = counts.get("positive", 0)
    neutral_count = counts.get("neutral", 0)
    negative_count = counts.get("negative", 0)

    positive_ratio = float(
        ratios.get("positive", 0.0)
    )
    neutral_ratio = float(
        ratios.get("neutral", 0.0)
    )
    negative_ratio = float(
        ratios.get("negative", 0.0)
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Positive",
        positive_count,
    )
    c2.metric(
        "Neutral",
        neutral_count,
    )
    c3.metric(
        "Negative",
        negative_count,
    )

    st.markdown("**Sentiment ratios**")

    st.progress(
        positive_ratio,
        text=f"Positive: {positive_ratio:.2%}",
    )

    st.progress(
        neutral_ratio,
        text=f"Neutral: {neutral_ratio:.2%}",
    )

    st.progress(
        negative_ratio,
        text=f"Negative: {negative_ratio:.2%}",
    )


def render_summary_rows(
    summaries: dict[str, Any],
) -> None:
    for sentiment in ("positive", "neutral", "negative"):
        label_col, content_col = st.columns([1.4, 6], gap="large")

        with label_col:
            st.markdown(
                f"""
                <div style="
                    font-size: 1.15rem;
                    font-weight: 700;
                    white-space: nowrap;
                    padding-top: 0.25rem;
                ">
                    {sentiment.title()}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with content_col:
            st.write(
                summaries.get(sentiment)
                or f"No {sentiment} summary."
            )


def render_review_list(
    title: str,
    reviews: list[dict[str, Any]],
) -> None:
    st.markdown(f"### {title}")

    if not reviews:
        st.info(
            "No reviews in this sentiment bucket."
        )
        return

    for review in reviews:
        header = (
            review.get("review_title")
            or review.get("review_id")
            or "Review"
        )

        with st.expander(
            str(header),
            expanded=False,
        ):
            st.write(
                review.get("review_text", "")
            )

            meta_cols = st.columns(4)

            meta_cols[0].write(
                "**Helpful votes:** "
                f"{review.get('helpful_vote', 0)}"
            )

            meta_cols[1].write(
                "**Confidence:** "
                f"{float(review.get('sentiment_confidence', 0.0)):.2f}"
            )

            meta_cols[2].write(
                "**Quality:** "
                f"{float(review.get('review_quality_score', 0.0)):.2f}"
            )

            meta_cols[3].write(
                "**Verified:** "
                f"{review.get('verified_purchase', False)}"
            )



def main() -> None:
    st.set_page_config(
        page_title="Amazon Product Reviewer",
        page_icon="🛒",
        layout="wide",
    )

    st.title(
        "🛒 Amazon Product Reviewer"
    )

    st.caption(
        "Search products, inspect metadata, "
        "and generate review insights on demand."
    )

    with st.sidebar:
        st.header("Settings")

        api_base_url = st.text_input(
            "API base URL",
            value=DEFAULT_API_BASE,
        )

        query = st.text_input(
            "Search products",
            value="",
        )

        limit = st.slider(
            "Products per page",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
        )

        max_reviews = st.slider(
            "Max reviews for analysis",
            min_value=20,
            max_value=200,
            value=80,
            step=10,
        )

        representative_k = st.slider(
            "Representative reviews per sentiment",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
        )

        regenerate = st.checkbox(
            "Regenerate insights (ignore cache)",
            value=False,
        )


    if "selected_product_id" not in st.session_state:
        st.session_state.selected_product_id = None

    left, right = st.columns(
        [1.15, 1.85],
        gap="large",
    )

    with left:
        st.subheader("Product Catalog")

        try:
            products_payload = api_get(
                api_base_url,
                "/products",
                params={
                    "query": query,
                    "limit": limit,
                    "offset": 0,
                },
            )

        except Exception as exc:
            st.error(
                f"Failed to load products: {exc}"
            )
            return

        items = products_payload.get(
            "items",
            [],
        )

        total = products_payload.get(
            "total",
            0,
        )

        st.caption(
            f"Found {total} products"
        )

        if not items:
            st.warning(
                "No products found."
            )

        else:
            for product in items:
                product_id = product.get(
                    "product_id"
                )

                title = (
                    product.get("product_title")
                    or product_id
                    or "Untitled product"
                )

                button_label = (
                    f"Select: {str(title)[:70]}"
                )

                if st.button(
                    button_label,
                    key=f"select_{product_id}",
                    width="stretch",
                ):
                    st.session_state.selected_product_id = (
                        product_id
                    )

    with right:
        selected_product_id = (
            st.session_state.selected_product_id
        )

        if not selected_product_id:
            st.info(
                "Select a product from the catalog "
                "to inspect details and generate insights."
            )
            return

        try:
            product = api_get(
                api_base_url,
                f"/products/{selected_product_id}",
            )

        except Exception as exc:
            st.error(
                f"Failed to load product detail: {exc}"
            )
            return

        st.subheader("Selected Product")

        render_product_card(product)

        st.divider()

        analyze_col, info_col = st.columns(
            [1, 2]
        )

        with analyze_col:
            analyze_clicked = st.button(
                "Generate Insights",
                type="primary",
                width="stretch",
            )

        with info_col:
            st.caption(
                f"Product ID: {selected_product_id} | "
                f"Max reviews: {max_reviews} | "
                f"Representative per sentiment: "
                f"{representative_k}"
            )

        session_key = (
            f"insights_{selected_product_id}"
        )

        if (
            analyze_clicked
            or session_key in st.session_state
        ):
            if analyze_clicked:
                with st.spinner(
                    "Running sentiment analysis "
                    "and summarization..."
                ):
                    try:
                        job = api_post(
                            api_base_url,
                            (
                                f"/products/"
                                f"{selected_product_id}/"
                                "insights/jobs"
                            ),
                            payload={
                                "max_reviews": max_reviews,
                                "representative_k": representative_k,
                                "regenerate": regenerate,
                            },
                        )

                        job_id = job.get("job_id")

                        if not job_id:
                            raise RuntimeError(
                                "API did not return job_id."
                            )

                        # Wait for Redis/RQ worker to finish.
                        wait_for_insight_job(
                            api_base_url,
                            str(job_id),
                        )

                    
                        insights = api_get(
                            api_base_url,
                            (
                                f"/products/"
                                f"{selected_product_id}/"
                                "insights"
                            ),
                        )

                        st.session_state[
                            session_key
                        ] = insights

                    except Exception as exc:
                        st.error(
                            "Failed to generate insights: "
                            f"{exc}"
                        )
                        return

            else:
                insights = st.session_state[
                    session_key
                ]
                
            st.divider()
            st.subheader("Insights")

            meta_cols = st.columns(4)

            meta_cols[0].metric(
                "Selected reviews",
                insights.get(
                    "selected_review_count",
                    0,
                ),
            )

            meta_cols[1].metric(
                "Total available reviews",
                insights.get(
                    "total_available_reviews",
                    0,
                ),
            )

            meta_cols[2].metric(
                "Latency (ms)",
                insights.get(
                    "latency_ms",
                    0.0,
                ),
            )

            meta_cols[3].metric(
                "Model version",
                insights.get(
                    "model_version",
                    "N/A",
                ),
            )

            render_sentiment_distribution(
                insights.get(
                    "sentiment_distribution",
                    {},
                )
            )

            st.divider()
            st.subheader(
                "Summaries by Sentiment"
            )

            summaries = insights.get(
                "summaries",
                {},
            )

            render_summary_rows(
                summaries
            )

            st.divider()
            st.subheader(
                "Representative Reviews"
            )

            representative_reviews = (
                insights.get(
                    "representative_reviews",
                    {},
                )
            )

            tabs = st.tabs(
                [
                    "Positive",
                    "Neutral",
                    "Negative",
                ]
            )

            with tabs[0]:
                render_review_list(
                    "Positive Reviews",
                    representative_reviews.get(
                        "positive",
                        [],
                    ),
                )

            with tabs[1]:
                render_review_list(
                    "Neutral Reviews",
                    representative_reviews.get(
                        "neutral",
                        [],
                    ),
                )

            with tabs[2]:
                render_review_list(
                    "Negative Reviews",
                    representative_reviews.get(
                        "negative",
                        [],
                    ),
                )


            with st.expander(
                "Show raw prompts",
                expanded=False,
            ):
                prompts = insights.get(
                    "prompts",
                    {},
                )

                st.markdown(
                    "**Positive prompt**"
                )
                st.code(
                    prompts.get(
                        "positive",
                        "",
                    )
                )

                st.markdown(
                    "**Neutral prompt**"
                )
                st.code(
                    prompts.get(
                        "neutral",
                        "",
                    )
                )

                st.markdown(
                    "**Negative prompt**"
                )
                st.code(
                    prompts.get(
                        "negative",
                        "",
                    )
                )


if __name__ == "__main__":
    main()