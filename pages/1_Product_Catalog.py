"""
Page 1 — Product Catalog
================================================================================
Browse the sampled product catalogue (5%/intention), filterable by the 10
shopping-intention segments discovered via LDA (thesis Chapter 4.2). Gives a
visual, tangible demonstration of the "intention discovery" results using
real product images.
================================================================================
"""

import os
import streamlit as st
from utils.data_loader import (
    download_data, load_articles, load_intention_labels, image_path,
)

st.set_page_config(page_title="Product Catalog", page_icon="🛍️", layout="wide")
download_data()

st.title("🛍️ Product Catalog by Shopping Intention")

articles = load_articles()
intention_labels = load_intention_labels()

st.caption(
    f"Showing {len(articles):,} products (5% stratified sample per intention — "
    "see the sampling methodology on the 'About Methodology' page)."
)

# ---- Filters ----
filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])

intention_options = ["All"] + [
    f"T{k} — {intention_labels[str(k)]['name']}" for k in range(10)
]
selected = filter_col1.selectbox("Filter by shopping-intention segment", intention_options)

sort_by = filter_col2.selectbox("Sort by", ["Default", "Price: low to high", "Price: high to low"])
n_show = filter_col3.slider("Number of products to show", 12, 96, 24, step=12)

filtered = articles.copy()
if selected != "All":
    k = int(selected.split("—")[0].strip()[1:])
    filtered = filtered[filtered["dominant_intention"] == k]

if sort_by == "Price: low to high" and "avg_price" in filtered.columns:
    filtered = filtered.sort_values("avg_price")
elif sort_by == "Price: high to low" and "avg_price" in filtered.columns:
    filtered = filtered.sort_values("avg_price", ascending=False)

filtered = filtered.head(n_show)

st.markdown(f"**{len(filtered)}** products shown")

# ---- Product grid ----
cols_per_row = 4
rows = [filtered.iloc[i:i + cols_per_row] for i in range(0, len(filtered), cols_per_row)]

for row_df in rows:
    cols = st.columns(cols_per_row)
    for col, (_, product) in zip(cols, row_df.iterrows()):
        with col:
            img_path = image_path(product["article_id"])
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.markdown(
                    "<div style='height:200px;background:#eee;display:flex;"
                    "align-items:center;justify-content:center;color:#999;'>"
                    "No image</div>", unsafe_allow_html=True
                )
            name = product.get("prod_name", "Unnamed product")
            st.markdown(f"**{name[:40]}**")
            k = int(product["dominant_intention"])
            st.caption(f"T{k} · {intention_labels[str(k)]['name']}")
            if "avg_price" in product and not str(product.get("avg_price", "")) == "nan":
                st.caption(f"💰 ~${float(product['avg_price']):.3f} (normalised)")

st.markdown("---")
with st.expander("ℹ️ Why only a 5% sample, not all 105,542 products?"):
    st.markdown(
        """
This is a demo build, using a **5% stratified sample per dominant
intention** — the exact same method and rate validated in the thesis
(Section 11, distribution deviation MAD ≤ 0.0036pp from the true
population). This keeps the dataset small enough to deploy for free on
Streamlit Community Cloud, while remaining scientifically representative of
each shopping-intention segment — see the full sampling report on the
"About Methodology" page.
"""
    )
