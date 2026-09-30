"""
app.py — Command Center (Segment Wall)
================================================================================
The admin's morning screen. Organised entirely around the 10 intention
segments — each card fuses real product imagery, a real business trend
(revenue share this period vs. the comparison period), and the model's own
measured performance for that segment (Table 4.7). Clicking "View detail"
drills into pages/1_Segment_Detail.py for the full picture (radar, trend,
product grid). Model QA lives separately in pages/2_Recommendation_Audit.py.
================================================================================
"""

import os
import streamlit as st

from utils.data_loader import (
    download_data, load_articles, load_intention_labels, load_monthly_trends, image_path,
)
from utils.theme import (
    inject_global_css, intention_color, render_sidebar_chrome, render_period_selector,
    status_badge, trend_arrow, thin_rule,
)
from utils.trends import with_period_columns, aggregate_period, segment_comparison
from utils.thesis_data import segment_static, model_improvement

st.set_page_config(page_title="Intention Console | H&M", page_icon="🧵", layout="wide")
inject_global_css()
download_data()

render_sidebar_chrome()
monthly_df = with_period_columns(load_monthly_trends())
granularity, period, compare_mode = render_period_selector(load_monthly_trends())

articles = load_articles()
intention_labels = load_intention_labels()

# ---- Header ----
st.markdown(
    "<div style='font-family:Libre Caslon Display, serif; font-size:2rem;'>"
    "Good morning, Merchandising &amp; Buying Team.</div>",
    unsafe_allow_html=True,
)
period_str = f"{granularity}: **{period}**" if period else "*(upload monthly_segment_trends.csv to enable period reporting)*"
st.markdown(f"Reporting period — {period_str}, vs. {compare_mode.lower()}.")
st.page_link(
    "pages/2_Personalized_Experience.py",
    label="✨ See the actual shopping experience Three-Tower creates — Personalized Experience →",
    icon="✨",
)
st.caption(
    "⚠️ Two independent metrics are shown per segment: the 🔴/🟡/🟢 status badge "
    "uses the **transaction-count-based** supply-demand gap (Table 5.2, static "
    "population figure). The ↑/↓ revenue trend uses **revenue-weighted** monthly "
    "share for the selected period. The two can move in different directions — "
    "e.g. a premium segment can have a small purchase-count share but a large "
    "revenue share. Do not read them as the same number over time."
)

comparison = None
if period:
    agg = aggregate_period(monthly_df, granularity)
    comparison = segment_comparison(agg, period, granularity, compare_mode)
    n_alerts = int((comparison["delta_pp"].abs() > 2).sum()) if comparison["delta_pp"].notna().any() else 0
    if n_alerts:
        st.markdown(f"🔴 **{n_alerts} segment(s) moved by more than 2pp this period** — see cards below.")

    # ---- Overall Three-Tower vs Two-Tower impact summary for this period ----
    valid = comparison[comparison["revenue"].notna()]
    if len(valid):
        total_revenue = valid["revenue"].sum()
        total_uplift = sum(
            row["revenue"] * model_improvement(idx) / 100 for idx, row in valid.iterrows()
        )
        st.markdown("##### Model impact summary — " + period)
        s1, s2, s3 = st.columns(3)
        s1.metric(f"Total revenue ({period})", f"{total_revenue:,.0f}")
        s2.metric("Estimated Three-Tower incremental revenue", f"+{total_uplift:,.0f}",
                   f"{total_uplift/total_revenue*100:.2f}% of period revenue")
        s3.metric("Segments with data this period", f"{len(valid)}/10")
        st.caption(
            "⚠️ Illustrative estimate: for each segment, real period revenue × "
            "that segment's measured AUC improvement (Table 4.7), summed across "
            "segments. This is a proxy translating measured model quality into "
            "a revenue figure — not a live A/B test result."
        )

thin_rule()

# ---- Segment Wall: 10 cards, 5 per row ----
cols_per_row = 5
for row_start in range(0, 10, cols_per_row):
    cols = st.columns(cols_per_row)
    for i, col in enumerate(cols):
        k = row_start + i
        with col:
            accent = intention_color(k)
            static = segment_static(k)
            emoji, status_label, status_color = status_badge(static["gap"])

            delta_str = "—"
            if comparison is not None and k in comparison.index:
                delta_str = trend_arrow(comparison.loc[k, "delta_pp"])

            # 3 representative product thumbnails
            seg_products = articles[articles["dominant_intention"] == k].head(3)
            thumbs = []
            for _, prod in seg_products.iterrows():
                ip = image_path(prod["article_id"])
                if os.path.exists(ip):
                    thumbs.append(ip)

            st.markdown(
                f"<div style='border-top:3px solid {accent}; padding-top:6px;'>"
                f"<b>T{k}</b> — {intention_labels[str(k)]['name']}</div>",
                unsafe_allow_html=True,
            )
            if thumbs:
                thumb_cols = st.columns(len(thumbs))
                for tc, tp in zip(thumb_cols, thumbs):
                    tc.image(tp, use_container_width=True)

            st.markdown(
                f"<div style='font-size:0.85rem; margin-top:4px;'>"
                f"{emoji} {status_label} <span style='color:#8A8578'>(txn-count gap, Table 5.2)</span><br>"
                f"Revenue {delta_str} <span style='color:#8A8578'>(this period)</span><br>"
                f"Model AUC gain: <b>+{model_improvement(k):.1f}%</b>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("View detail", key=f"detail_{k}", use_container_width=True):
                st.session_state["selected_segment"] = k
                st.switch_page("pages/1_Segment_Detail.py")

thin_rule()
st.caption(
    "Revenue trend computed from real transaction dates grouped by each "
    "product's dominant intention (Script 01 bonus step). Segment "
    "population figures (users, confidence, catalogue gap) are computed on "
    "the full 1.37M-customer population — Table 5.1/5.2. Model AUC gain — "
    "Table 4.7. Product thumbnails are drawn from the 5% demo sample."
)
