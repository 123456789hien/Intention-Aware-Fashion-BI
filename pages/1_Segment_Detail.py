"""
pages/1_Segment_Detail.py — Segment Detail
================================================================================
The drill-down for one intention segment: a representative persona radar,
a real revenue trend that RESPONDS to the sidebar period selector (month/
quarter/year + comparison), a rule-based recommendation, and the product
grid. The Recommendation Engine Audit has moved to its own page
(2_Recommendation_Audit.py) — this page is now purely business reporting,
not a model QA tool.
================================================================================
"""

import os
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import (
    download_data, load_articles, load_demo_personas,
    load_intention_labels, load_monthly_trends, image_path,
)
from utils.theme import (
    inject_global_css, intention_color, intention_lens, render_sidebar_chrome,
    render_period_selector, status_badge, thin_rule,
)
from utils.charts import intention_radar_chart
from utils.trends import (
    with_period_columns, aggregate_period, aggregate_units_share, segment_comparison, GRANULARITY_COL,
)
from utils.thesis_data import segment_static, model_improvement, recommendation_text

st.set_page_config(page_title="Segment Detail", page_icon="🔍", layout="wide")
inject_global_css()
download_data()

render_sidebar_chrome()
monthly_df_raw = load_monthly_trends()
monthly_df = with_period_columns(monthly_df_raw)
granularity, period, compare_mode = render_period_selector(monthly_df_raw)

articles = load_articles()
personas = load_demo_personas()
intention_labels = load_intention_labels()

# ---- Segment selector (kept in sync with the Command Center) ----
default_k = st.session_state.get("selected_segment", 0)
options = [f"T{k} — {intention_labels[str(k)]['name']}" for k in range(10)]
choice = st.selectbox("Segment", options, index=default_k)
k = int(choice.split("—")[0].strip()[1:])
st.session_state["selected_segment"] = k

accent = intention_color(k)
static = segment_static(k)
emoji, status_label, status_color = status_badge(static["gap"])

st.markdown(
    f"""
    <div style="border-top:4px solid {accent}; padding-top:10px;">
        <div style="font-family:'Libre Caslon Display',serif; font-size:1.9rem;">
            T{k} — {intention_labels[str(k)]['name']}
        </div>
        <div style="font-style:italic; color:#8A8578;">{intention_lens(k)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Period-aware summary — every number below now reflects the sidebar
# selection wherever the underlying data genuinely supports it.
# ============================================================================
agg = aggregate_period(monthly_df, granularity) if period else None
seg_comparison_row = None
if agg is not None:
    comp = segment_comparison(agg, period, granularity, compare_mode)
    if k in comp.index:
        seg_comparison_row = comp.loc[k]

# Period-specific customer count & supply-demand gap — both computed fresh
# from real transactions (monthly_segment_trends.csv), not the fixed
# all-time Table 5.1/5.2 figures.
units_agg = aggregate_units_share(monthly_df, granularity)
period_row = None
if not units_agg.empty:
    if granularity == "All time":
        match = units_agg[units_agg["intention"] == k]
    else:
        col = GRANULARITY_COL[granularity]
        match = units_agg[(units_agg[col] == period) & (units_agg["intention"] == k)]
    if len(match):
        period_row = match.iloc[0]

period_gap = (period_row["demand_share_pct"] - static["cat_share"]) if period_row is not None else None
period_customers = int(period_row["n_customers"]) if period_row is not None else None
period_emoji, period_status_label, _ = status_badge(period_gap) if period_gap is not None else (emoji, status_label, status_color)

m1, m2, m3, m4 = st.columns(4)
if period_customers is not None:
    period_label = period if granularity != "All time" else "all time"
    m1.metric(f"Customers active ({period_label})", f"{period_customers:,}")
else:
    m1.metric("Customers in segment", f"{static['users']:,}", f"{static['user_share']:.2f}% of base (all-time)")
m2.metric("Avg. confidence (all-time)", f"{static['confidence']:.4f}",
          help="Confidence is a per-customer, all-time Bayesian estimate (Section 3.4.5) — "
               "it cannot be recomputed for a shorter period without re-running that pipeline.")
if period_gap is not None:
    m3.metric(f"Supply-demand gap, txn-count ({period if granularity != 'All time' else 'all time'})",
              f"{period_gap:+.2f}pp", period_status_label)
else:
    m3.metric("Supply-demand gap (txn-count, Table 5.2)", f"{static['gap']:+.2f}pp", status_label)
if seg_comparison_row is not None and seg_comparison_row["revenue_share_pct"] is not None:
    delta = seg_comparison_row["delta_pp"]
    delta_str = f"{delta:+.2f}pp vs {compare_mode.lower()}" if delta is not None else "no prior period"
    m4.metric(f"Revenue share ({period})", f"{seg_comparison_row['revenue_share_pct']:.2f}%", delta_str)
else:
    m4.metric(f"Revenue share ({period or 'n/a'})", "—", "no data for this period")

st.caption(
    "Customers (m1) and the supply-demand gap (m3) are recomputed for the "
    f"**{granularity.lower()} you selected** from real transactions — they "
    "will change as you change the sidebar. Confidence (m2) is the one "
    "figure that cannot be time-sliced: it is a per-customer, all-time "
    "profile estimate, not something logged per transaction."
)

thin_rule()

# ============================================================================
# Radar (representative persona) + real revenue trend — period-aware
# ============================================================================
radar_col, trend_col = st.columns(2)

with radar_col:
    st.subheader("Representative intention profile")
    seg_personas = personas[personas["dominant_intention"] == k]
    if len(seg_personas):
        rep = seg_personas.sort_values("confidence", ascending=False).iloc[0]
        vec = rep[[f"intention_{i}" for i in range(10)]].values.astype(np.float32)
        st.plotly_chart(
            intention_radar_chart(vec, intention_labels),
            use_container_width=True, config={"displayModeBar": False},
            key=f"segdetail_radar_{k}",
        )
        st.caption(f"Persona confidence: {rep['confidence']:.2f} · {int(rep['n_purchases'])} historical purchases")
    else:
        st.info("No demo persona available for this segment.")

with trend_col:
    st.subheader(f"Revenue share trend — by {granularity.lower()}")
    if agg is None or agg.empty:
        st.info("monthly_segment_trends.csv not found yet — run Script 01.")
    elif granularity == "All time":
        seg_series = agg[agg["intention"] == k]
        if len(seg_series):
            row = seg_series.iloc[0]
            st.metric("Revenue share (all time)", f"{row['revenue_share_pct']:.2f}%")
            st.caption("Select Month, Quarter, or Year in the sidebar to see the trend line over time.")
        else:
            st.info("No data for this segment.")
    else:
        col = GRANULARITY_COL[granularity]
        seg_series = agg[agg["intention"] == k].sort_values(col)
        if len(seg_series):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=seg_series[col], y=seg_series["revenue_share_pct"],
                mode="lines+markers", line=dict(color=accent), marker=dict(color=accent),
            ))
            # Highlight the selected period so the chart visibly reflects the sidebar choice
            if period in seg_series[col].values:
                sel_row = seg_series[seg_series[col] == period].iloc[0]
                fig.add_trace(go.Scatter(
                    x=[period], y=[sel_row["revenue_share_pct"]], mode="markers",
                    marker=dict(color="#C23B5E", size=14, symbol="star"),
                    name="Selected period", showlegend=False,
                ))
            fig.update_layout(
                xaxis_title=granularity, yaxis_title="Revenue share (%)",
                margin=dict(l=10, r=10, t=10, b=10), height=300, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"segdetail_trend_{k}_{granularity}")
            st.caption(
                f"Aggregated to {granularity.lower()} level, matching the sidebar selector. "
                "The red star marks the period currently selected."
            )
        else:
            st.info("No data for this segment at the current granularity.")

thin_rule()

# ============================================================================
# Model impact estimate for the selected period (honest framing — see caption)
# ============================================================================
st.subheader("Estimated Three-Tower impact for this period")
if seg_comparison_row is not None and seg_comparison_row["revenue"] is not None:
    period_revenue = seg_comparison_row["revenue"]
    uplift_pct = model_improvement(k) / 100
    estimated_uplift = period_revenue * uplift_pct
    c1, c2 = st.columns(2)
    c1.metric(f"Segment revenue ({period})", f"{period_revenue:,.0f}")
    c2.metric("Estimated incremental revenue from Three-Tower", f"+{estimated_uplift:,.0f}",
              f"{model_improvement(k):.1f}% (Table 4.7)")
    st.caption(
        "⚠️ Illustrative estimate only: real period revenue × the measured "
        "AUC improvement for this segment (Table 4.7), applied as a proxy for "
        "revenue uplift. This is not a live A/B test result — no online "
        "experiment comparing the two models has been run."
    )
else:
    st.info("Select a period with available data to see this estimate.")

thin_rule()

# ============================================================================
# Business recommendation (rule-based, same logic as thesis Section 5.4)
# ============================================================================
st.subheader("Recommended action")
st.markdown(f"{emoji} **{recommendation_text(k)}**")

thin_rule()

# ============================================================================
# Product grid for this segment
# ============================================================================
st.subheader("Representative products")
seg_articles = articles[articles["dominant_intention"] == k].head(12)
cols = st.columns(6)
for i, (_, product) in enumerate(seg_articles.iterrows()):
    with cols[i % 6]:
        ip = image_path(product["article_id"])
        if os.path.exists(ip):
            st.image(ip, use_container_width=True)
        st.caption(str(product.get("prod_name", ""))[:28])

thin_rule()
link_col1, link_col2 = st.columns(2)
link_col1.page_link("pages/2_Personalized_Experience.py", label="See the shopping experience →", icon="✨")
link_col2.page_link("pages/3_Recommendation_Audit.py", label="Go to Recommendation Engine Audit →", icon="🧪")
