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
    load_customer_purchases, load_intention_labels, load_monthly_trends, model_paths, image_path,
)
from utils.models import load_models
from utils.recommender import score_catalog
from utils.theme import (
    inject_global_css, intention_color, intention_lens, render_sidebar_chrome,
    render_period_selector, thin_rule, THREAD,
)
from utils.charts import intention_radar_chart
from utils.thesis_data import segment_static, recommendation_text
from utils.trends import filter_by_period

st.set_page_config(page_title="Customer Validation", page_icon="✨", layout="wide")
inject_global_css()
download_data()
render_sidebar_chrome()
granularity, period, compare_mode = render_period_selector(load_monthly_trends())

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
    st.plotly_chart(fig, use_container_width=True, key="custval_all_bar")

    st.subheader("Customer count vs. average confidence, by segment")
    fig2 = px.scatter(
        seg_df, x="confidence", y="users", size="users", color="label",
        color_discrete_map=dict(zip(seg_df["label"], seg_df["color"])),
        labels={"confidence": "Avg. profile confidence", "users": "Customers"},
        size_max=60,
    )
    st.plotly_chart(fig2, use_container_width=True, key="custval_all_scatter")

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

# ---- Real purchase history — loaded once, reused by every section below ----
cust_purchases_all = purchases[purchases["customer_id"] == customer_id].copy()
cust_purchases = filter_by_period(cust_purchases_all, "t_dat", granularity, period)
purchased_articles = articles[articles["article_id"].isin(cust_purchases["article_id"])].copy()

period_note = f"{granularity}: {period}" if period and granularity != "All time" else "All time"
st.caption(
    f"📅 Showing **{period_note}** — {len(cust_purchases)} of "
    f"{len(cust_purchases_all)} total real purchases on record for this period."
)

if purchased_articles.empty:
    st.warning(
        f"No real purchases for this customer in {period_note}. "
        "Try a different period (or 'All time') in the sidebar."
    )
    st.stop()

# ============================================================================
# Profile vs. Reality — with 50-100 real purchases now available, we can
# properly check whether the computed Bayesian profile (Section 3.4.5)
# actually matches what this customer bought, instead of just displaying it.
# ============================================================================
st.subheader("Profile vs. Reality")
st.caption("Does the computed intention profile match what this customer actually bought?")

profile_col, reality_col = st.columns(2)
with profile_col:
    st.markdown("**Computed profile** (Bayesian, Section 3.4.5)")
    vec = cust_row[intention_cols].values.astype(np.float32)
    st.plotly_chart(
        intention_radar_chart(vec, intention_labels), use_container_width=True,
        config={"displayModeBar": False}, key=f"custval_radar_{customer_id}",
    )

with reality_col:
    st.markdown(f"**Actual breakdown** of {len(purchased_articles)} real purchases")
    real_counts = purchased_articles["dominant_intention"].value_counts().reindex(range(10), fill_value=0)
    real_df = pd.DataFrame({"label": [f"T{i}" for i in range(10)], "count": real_counts.values})
    real_df["color"] = [intention_color(i) for i in range(10)]
    fig_real = px.bar(
        real_df, x="count", y="label", orientation="h",
        color="label", color_discrete_map=dict(zip(real_df["label"], real_df["color"])),
        labels={"count": "Real purchases", "label": ""},
    )
    fig_real.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_real, use_container_width=True, key=f"custval_realbar_{customer_id}")

dominant_real_k = int(real_counts.idxmax())
match_note = "✅ Matches" if dominant_real_k == own_k else "⚠️ Differs from"
st.caption(
    f"{match_note} the computed profile: most-purchased real segment is "
    f"**T{dominant_real_k} — {intention_labels[str(dominant_real_k)]['name']}** "
    f"({int(real_counts.max())} of {len(purchased_articles)} purchases), vs. "
    f"computed dominant **T{own_k}**."
)

thin_rule()

# ---- Purchase timeline — only meaningful now with 50-100 real data points ----
st.subheader("Purchase timeline")
timeline_df = cust_purchases.merge(
    purchased_articles[["article_id", "dominant_intention"]], on="article_id", how="left"
)
timeline_df["t_dat"] = pd.to_datetime(timeline_df["t_dat"], errors="coerce")
timeline_df["segment"] = timeline_df["dominant_intention"].apply(lambda i: f"T{int(i)}" if pd.notna(i) else "n/a")
color_map = {f"T{i}": intention_color(i) for i in range(10)}
fig_tl = px.scatter(
    timeline_df, x="t_dat", y="price", color="segment", color_discrete_map=color_map,
    labels={"t_dat": "Purchase date", "price": "Price (normalised)"},
)
fig_tl.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), legend_title="")
st.plotly_chart(fig_tl, use_container_width=True, key=f"custval_timeline_{customer_id}")
st.caption("Each point is one real transaction, coloured by that product's own segment.")

thin_rule()

# ---- Real purchase gallery (display capped; validation below uses ALL) ----
st.subheader("Sample of products actually bought")
GALLERY_DISPLAY_LIMIT = 20  # display only — validation below always uses the FULL purchase list
display_articles = purchased_articles.head(GALLERY_DISPLAY_LIMIT)
cols = st.columns(min(5, len(display_articles)))
for i, (_, product) in enumerate(display_articles.iterrows()):
    with cols[i % len(cols)]:
        ip = image_path(product["article_id"])
        if os.path.exists(ip):
            st.image(ip, use_container_width=True)
        st.caption(str(product.get("prod_name", ""))[:26])

if len(purchased_articles) > GALLERY_DISPLAY_LIMIT:
    st.caption(
        f"+ {len(purchased_articles) - GALLERY_DISPLAY_LIMIT} more real purchases "
        f"not shown here — all {len(purchased_articles)} are still used in the "
        f"validation below."
    )

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
    st.subheader("Distribution of percentile ranks — all real purchases")
    hist_df = truth[["three_pct", "two_pct"]].rename(
        columns={"three_pct": "Three-Tower", "two_pct": "Two-Tower"}
    ).melt(var_name="Model", value_name="Percentile")
    fig_hist = px.histogram(
        hist_df, x="Percentile", color="Model", barmode="overlay", opacity=0.65, nbins=20,
        color_discrete_map={"Three-Tower": THREAD, "Two-Tower": "#8A8578"},
    )
    fig_hist.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Percentile rank (%) — lower is better")
    st.plotly_chart(fig_hist, use_container_width=True, key=f"custval_hist_{customer_id}")
    st.caption(
        f"A distribution shifted left = that model consistently ranks this "
        f"customer's real purchases near the top, not just on average. "
        f"Based on all {len(truth)} of their real purchases found in the "
        f"candidate catalogue."
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
price_range = (
    f" Spend ranges from ~${cust_purchases['price'].min():.3f} to "
    f"~${cust_purchases['price'].max():.3f} per item (avg ~${avg_spend:.3f})."
    if avg_spend else ""
)
profile_note = (
    "" if dominant_real_k == own_k else
    f" Note: their actual top segment by purchase count is T{dominant_real_k}, "
    f"not the computed profile's T{own_k} — worth a closer look before acting "
    f"on this profile alone."
)
st.markdown(
    f"This customer's purchases are dominated by **T{own_k} — "
    f"{intention_labels[str(own_k)]['name']}** (confidence {cust_row['confidence']:.2f})."
    f"{price_range}{profile_note} {recommendation_text(own_k)}"
)

thin_rule()
st.caption(
    "Ground-truth validation set: 10 real customers per intention segment "
    "(Script 01), each with 50-100 logged purchases, extended into the "
    "product-feature sample so every real purchase can be scored."
)
