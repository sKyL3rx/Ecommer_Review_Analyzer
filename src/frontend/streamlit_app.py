from __future__ import annotations

from typing import Any

import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"


def api_get(base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def api_post(base_url: str, path: str, payload: dict[str, Any]) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def render_product_card(product: dict[str, Any]) -> None:
    title = product.get("product_title") or "Untitled product"
    store = product.get("store") or "Unknown store"
    category = product.get("main_category") or "Unknown category"
    price = product.get("price") or "N/A"
    review_count = product.get("review_count", 0)
    avg_rating = product.get("average_rating")
    rating_number = product.get("rating_number", 0)
    image_url = product.get("image_url")

    with st.container(border=True):
        cols = st.columns([1, 3])

        with cols[0]:
            if image_url:
                st.image(image_url, use_container_width=True)
            else:
                st.caption("No image")

        with cols[1]:
            st.subheader(title)
            st.write(f"**Store:** {store}")
            st.write(f"**Category:** {category}")
            st.write(f"**Price:** {price}")

            if avg_rating is not None:
                st.write(f"**Catalog rating:** {avg_rating:.2f} ({rating_number} ratings)")
            else:
                st.write("**Catalog rating:** N/A")

            st.write(f"**Review count:** {review_count}")


def render_sentiment_distribution(dist: dict[str, Any]) -> None:
    counts = dist.get("counts", {})
    ratios = dist.get("ratios", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", counts.get("positive", 0))
    c2.metric("Neutral", counts.get("neutral", 0))
    c3.metric("Negative", counts.get("negative", 0))

    st.write("**Sentiment ratios**")
    st.progress(float(ratios.get("positive", 0.0)), text=f"Positive: {ratios.get('positive', 0.0):.2%}")
    st.progress(float(ratios.get("neutral", 0.0)), text=f"Neutral: {ratios.get('neutral', 0.0):.2%}")
    st.progress(float(ratios.get("negative", 0.0)), text=f"Negative: {ratios.get('negative', 0.0):.2%}")


def render_review_list(title: str, reviews: list[dict[str, Any]]) -> None:
    st.markdown(f"### {title}")

    if not reviews:
        st.info("No reviews in this sentiment bucket.")
        return

    for review in reviews:
        header = review.get("review_title") or review.get("review_id") or "Review"
        with st.expander(header, expanded=False):
            st.write(review.get("review_text", ""))
            meta_cols = st.columns(4)
            meta_cols[0].write(f"**Helpful votes:** {review.get('helpful_vote', 0)}")
            meta_cols[1].write(f"**Confidence:** {review.get('sentiment_confidence', 0.0):.2f}")
            meta_cols[2].write(f"**Quality:** {review.get('review_quality_score', 0.0):.2f}")
            meta_cols[3].write(f"**Verified:** {review.get('verified_purchase', False)}")


def main() -> None:
    st.set_page_config(
        page_title="Product Review Intelligence Console",
        page_icon="🛍️",
        layout="wide",
    )

    st.title("🛍️ Product Review Intelligence Console")
    st.caption("Search products, inspect metadata, and generate review insights on demand.")

    with st.sidebar:
        st.header("Settings")
        api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE)
        query = st.text_input("Search products", value="")
        limit = st.slider("Products per page", min_value=5, max_value=50, value=10, step=5)
        max_reviews = st.slider("Max reviews for analysis", min_value=20, max_value=200, value=80, step=10)
        representative_k = st.slider("Representative reviews per sentiment", min_value=1, max_value=10, value=5, step=1)
        regenerate = st.checkbox("Regenerate insights (ignore cache)", value=False)

    if "selected_product_id" not in st.session_state:
        st.session_state.selected_product_id = None

    left, right = st.columns([1.15, 1.85], gap="large")

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
        except Exception as e:
            st.error(f"Failed to load products: {e}")
            return

        items = products_payload.get("items", [])
        total = products_payload.get("total", 0)
        st.caption(f"Found {total} products")

        if not items:
            st.warning("No products found.")
        else:
            for product in items:
                title = product.get("product_title") or product.get("product_id")
                button_label = f"Select: {title[:70]}"
                if st.button(button_label, key=f"select_{product['product_id']}", use_container_width=True):
                    st.session_state.selected_product_id = product["product_id"]

    with right:
        selected_product_id = st.session_state.selected_product_id

        if not selected_product_id:
            st.info("Select a product from the catalog to inspect details and generate insights.")
            return

        try:
            product = api_get(api_base_url, f"/products/{selected_product_id}")
        except Exception as e:
            st.error(f"Failed to load product detail: {e}")
            return

        st.subheader("Selected Product")
        render_product_card(product)

        st.divider()

        analyze_col, info_col = st.columns([1, 2])
        with analyze_col:
            analyze_clicked = st.button("Generate Insights", type="primary", use_container_width=True)

        with info_col:
            st.caption(
                f"Product ID: {selected_product_id} | Max reviews: {max_reviews} | "
                f"Representative per sentiment: {representative_k}"
            )

        if analyze_clicked or f"insights_{selected_product_id}" in st.session_state:
            if analyze_clicked:
                with st.spinner("Running sentiment analysis and summarization..."):
                    try:
                        insights = api_post(
                            api_base_url,
                            f"/products/{selected_product_id}/insights",
                            payload={
                                "max_reviews": max_reviews,
                                "representative_k": representative_k,
                                "regenerate": regenerate,
                            },
                        )
                        st.session_state[f"insights_{selected_product_id}"] = insights
                    except Exception as e:
                        st.error(f"Failed to generate insights: {e}")
                        return
            else:
                insights = st.session_state[f"insights_{selected_product_id}"]

            st.divider()
            st.subheader("Insights")

            meta_cols = st.columns(4)
            meta_cols[0].metric("Selected reviews", insights.get("selected_review_count", 0))
            meta_cols[1].metric("Total available reviews", insights.get("total_available_reviews", 0))
            meta_cols[2].metric("Latency (ms)", insights.get("latency_ms", 0.0))
            meta_cols[3].metric("Model version", insights.get("model_version", "N/A"))

            render_sentiment_distribution(insights.get("sentiment_distribution", {}))

            st.divider()
            st.subheader("Summaries by Sentiment")

            summaries = insights.get("summaries", {})
            s1, s2, s3 = st.columns(3)

            with s1:
                st.markdown("#### Positive")
                st.write(summaries.get("positive") or "No positive summary.")

            with s2:
                st.markdown("#### Neutral")
                st.write(summaries.get("neutral") or "No neutral summary.")

            with s3:
                st.markdown("#### Negative")
                st.write(summaries.get("negative") or "No negative summary.")

            st.divider()
            st.subheader("Representative Reviews")

            rep_reviews = insights.get("representative_reviews", {})
            tabs = st.tabs(["Positive", "Neutral", "Negative"])

            with tabs[0]:
                render_review_list("Positive Reviews", rep_reviews.get("positive", []))
            with tabs[1]:
                render_review_list("Neutral Reviews", rep_reviews.get("neutral", []))
            with tabs[2]:
                render_review_list("Negative Reviews", rep_reviews.get("negative", []))

            with st.expander("Show raw prompts", expanded=False):
                prompts = insights.get("prompts", {})
                st.markdown("**Positive prompt**")
                st.code(prompts.get("positive", ""))
                st.markdown("**Neutral prompt**")
                st.code(prompts.get("neutral", ""))
                st.markdown("**Negative prompt**")
                st.code(prompts.get("negative", ""))


if __name__ == "__main__":
    main()