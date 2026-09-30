"""
pages/2_Personalized_Experience.py — Customer Validation
================================================================================
Redesigned per request: instead of a curated "persona", pick a REAL Customer
ID and validate Three-Tower vs Two-Tower against what that customer actually
bought (Script 03's customer_validation.csv / customer_purchases.csv) — a
proper ground-truth comparison, not a proxy intention-match rate. "All
customers" shows population-level customer/intention analysis instead of a
single customer's validation (there is no single ground truth to validate
against for "all" at once).

Layout intentionally mirrors the previous version: Step 1 (choose shopper) →
summary card → Step 2 (compare, two metric columns, two feeds side by side).
================================================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from utils.data_loader import (
    download_data, load_articles, load_feature_matrices, load_customer_validation,
    load_customer_purchases, load_intention_labels, model_paths, image_path,
)
from utils.models import load_models
from utils.recommender import score_catalog
from utils.theme import inject_global_css, intention_color, intention_lens, render_sidebar_chrome, thin_rule
from utils.thesis_data import segment_static, recommendation_text

st.set_page_config(page_title="Customer Validation", page_icon="✨", layout="wide")
inject_global_css()
download_data()
render_sidebar_chrome()

st.title("Customer Validation")
st.caption(
    "Instead of a curated persona, pick a real customer and compare Three-Tower "
    "vs Two-Tower against what that customer actually bought — this is a "
    "ground-truth check, not a proxy metric."
)

articles = load_articles()
validation = load_customer_validation()
purchases = load_customer_purchases()
intention_labels = load_intention_labels()
intention_cols = [f"intention_{k}" for k in range(10)]

if validation.empty:
    st.warning(
        "⚠️ customer_validation.csv not found yet — run Script 03 "
        "(03_customer_validation_export.py) on Kaggle and re-upload data_export.zip."
    )
    st.stop()

# ============================================================================
# Step 1 — Choose a shopper
# ============================================================================
st.header("Step 1 — Choose a customer")

cust_options = ["All customers"] + [
    f"{row.customer_id[:12]}… — {row.persona_label}" for row in validation.itertuples()
]
choice = st.selectbox("Customer", cust_options)

# ----------------------------------------------------------------------
# ALL CUSTOMERS — population-level analysis (Table 5.1), not a single
# customer's ground-truth validation (there's no one ground truth for "all")
# ----------------------------------------------------------------------
if choice == "All customers":
    st.info(
        "Showing population-level analysis across all 1.37M customers "
        "(Table 5.1) — not just the validation sample above. Select an "
        "individual customer to validate model accuracy against their real "
        "purchase history."
    )
    thin_rule()

    rows = []
    for k in range(10):
        s = segment_static(k)
        rows.append({
            "T": k, "name": intention_labels[str(k)]["name"], "users": s["users"],
            "share": s["user_share"], "confidence": s["confidence"], "gap": s["gap"],
        })
    seg_df = pd.DataFrame(rows)
    seg_df["label"] = seg_df.apply(lambda r: f"T{r['T']} — {r['name']}", axis=1)
    seg_df["color"] = seg_df["T"].apply(intention_color)

    st.subheader("How many customers per intention segment?")
    fig = px.bar(
        seg_df.sort_values("users"), x="users", y="label", orientation="h",
        color="label", color_discrete_map=dict(zip(seg_df["label"], seg_df["color"])),
        labels={"users": "Number of customers", "label": ""},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer count vs. average confidence, by segment")
    fig2 = px.scatter(
        seg_df, x="confidence", y="users", size="users", color="label",
        color_discrete_map=dict(zip(seg_df["label"], seg_df["color"])),
        labels={"confidence": "Avg. profile confidence", "users": "Customers"},
        size_max=60,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        seg_df[["label", "users", "share", "confidence", "gap"]].rename(columns={
            "label": "Segment", "users": "Customers", "share": "% of base",
            "confidence": "Avg. confidence", "gap": "Supply-demand gap (pp)",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"Validation sample available for individual lookup: "
        f"{len(validation)} real customers (10 per segment), with "
        f"{purchases['customer_id'].nunique()} of them having logged "
        f"transaction history in this demo."
    )
    st.stop()

# ----------------------------------------------------------------------
# SINGLE CUSTOMER — ground-truth validation
# ----------------------------------------------------------------------
row_idx = cust_options.index(choice) - 1
cust_row = validation.iloc[row_idx]
customer_id = cust_row["customer_id"]
own_k = int(cust_row["dominant_intention"])
accent = intention_color(own_k)

accent_html = (
    f"<div style='border-left:4px solid {accent}; padding:8px 14px; background:#F1EFE9;'>"
    f"Customer <code>{customer_id[:16]}…</code> · Primary segment: "
    f"<b>T{own_k} — {intention_labels[str(own_k)]['name']}</b> · "
    f"<i>{intention_lens(own_k)}</i><br>"
    f"Confidence {cust_row['confidence']:.2f} · {int(cust_row['n_purchases'])} real purchases on record</div>"
)
st.markdown(accent_html, unsafe_allow_html=True)

thin_rule()

# ---- Real purchase history (ground truth) ----
st.subheader("What this customer actually bought")
cust_purchases = purchases[purchases["customer_id"] == customer_id]
purchased_articles = articles[articles["article_id"].isin(cust_purchases["article_id"])].copy()

if purchased_articles.empty:
    st.warning("No transaction rows found for this customer in customer_purchases.csv.")
    st.stop()

cols = st.columns(min(5, len(purchased_articles)))
for i, (_, product) in enumerate(purchased_articles.iterrows()):
    with cols[i % len(cols)]:
        ip = image_path(product["article_id"])
        if os.path.exists(ip):
            st.image(ip, use_container_width=True)
        st.caption(str(product.get("prod_name", ""))[:26])

thin_rule()

# ============================================================================
# Step 2 — Validate model accuracy against these real purchases
# ============================================================================
st.header("Step 2 — Which model ranks these real purchases higher?")

if st.button("Run validation", type="primary"):
    three_path, two_path = model_paths()
    three_model, two_model = load_models(three_path, two_path)
    visual_feat, semantic_feat, art_feat_idx = load_feature_matrices()

    user_intention = cust_row[intention_cols].values.astype(np.float32)
    user_demo = np.array([
        cust_row.get("age", 30.0), cust_row.get("FN", 0.0), cust_row.get("Active", 0.0),
    ], dtype=np.float32)

    with st.spinner("Scoring the full candidate catalogue with both models..."):
        _, full_scored = score_catalog(
            three_model, two_model, visual_feat, semantic_feat, art_feat_idx,
            articles, user_intention, user_demo, top_n=10,
        )
    st.session_state["val_full"] = full_scored
    st.session_state["val_customer"] = customer_id

if st.session_state.get("val_customer") == customer_id and "val_full" in st.session_state:
    full_scored = st.session_state["val_full"]
    n_candidates = len(full_scored)

    full_scored = full_scored.copy()
    full_scored["three_rank"] = full_scored["three_tower_score"].rank(ascending=False, method="min")
    full_scored["two_rank"] = full_scored["two_tower_score"].rank(ascending=False, method="min")

    truth = full_scored[full_scored["article_id"].isin(cust_purchases["article_id"])].copy()
    truth["three_pct"] = truth["three_rank"] / n_candidates * 100
    truth["two_pct"] = truth["two_rank"] / n_candidates * 100

    three_avg_pct = truth["three_pct"].mean()
    two_avg_pct = truth["two_pct"].mean()
    winner = "Three-Tower" if three_avg_pct < two_avg_pct else "Two-Tower"

    thin_rule()
    m1, m2 = st.columns(2)
    m1.metric("Two-Tower — avg. percentile rank of real purchases", f"{two_avg_pct:.1f}%")
    m2.metric("Three-Tower — avg. percentile rank of real purchases", f"{three_avg_pct:.1f}%",
               f"{two_avg_pct - three_avg_pct:+.1f}pp better" if three_avg_pct < two_avg_pct
               else f"{three_avg_pct - two_avg_pct:+.1f}pp worse")
    st.success(f"**{winner}** ranked this customer's real purchases higher on average, out of {n_candidates:,} candidate products.")
    st.caption(
        "Lower percentile = better: it means the model ranked the product the "
        "customer actually bought closer to the top of the full candidate list, "
        "rather than buried near the bottom. This is a direct ground-truth "
        "check, not a proxy metric."
    )

    thin_rule()
    st.subheader("Rank of each real purchase, by model")
    display_truth = truth[["prod_name", "three_rank", "three_pct", "two_rank", "two_pct"]].rename(columns={
        "prod_name": "Product", "three_rank": "3T rank", "three_pct": "3T percentile",
        "two_rank": "2T rank", "two_pct": "2T percentile",
    })
    st.dataframe(display_truth, use_container_width=True, hide_index=True)
else:
    st.info("Click **Run validation** to score the full candidate catalogue for this customer.")

thin_rule()

# ============================================================================
# Customer recommendation summary
# ============================================================================
st.subheader("Customer recommendation summary")
avg_spend = cust_purchases["price"].mean() if len(cust_purchases) else None
price_line = f" Average spend per item: ~${avg_spend:.3f} (normalised)." if avg_spend else ""
st.markdown(
    f"This customer's purchases are dominated by **T{own_k} — "
    f"{intention_labels[str(own_k)]['name']}** (confidence {cust_row['confidence']:.2f})."
    f"{price_line} {recommendation_text(own_k)}"
)

thin_rule()
st.caption(
    "Ground-truth validation set: 10 real customers per intention segment "
    "(Script 03), each with 3-15 logged purchases, extended into the "
    "product-feature sample so every real purchase can be scored."
)
