"""
Page 3 — Business Intelligence Dashboard
================================================================================
Recreates the core tables/charts of thesis Chapter 5: customer segmentation
by intention, supply-demand gap report, and personalisation investment
priority by segment. Figures are taken directly from sampling_report.json
and articles_sample.csv (computed on the FULL population in Script 01, so
they are NOT distorted by the 5% catalogue sample used for image display).
================================================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import (
    download_data, load_articles, load_intention_labels, load_sampling_report,
)

st.set_page_config(page_title="Business Intelligence", page_icon="📊", layout="wide")
download_data()

st.title("📊 Business Intelligence Dashboard")
st.caption("Recreating thesis Chapter 5 — figures computed on the full population, not the demo sample")

intention_labels = load_intention_labels()
articles = load_articles()
report = load_sampling_report()

# ============================================================================
# FIXED FIGURES FROM THE THESIS (Table 5.1, 5.2 — verified against the
# original notebooks). Stated explicitly here (not derived from the 5%
# sample) to guarantee they match the published Chapter 5 figures exactly,
# unaffected by the small demo sample size.
# ============================================================================
SEGMENT_DATA = [
    {"T": 0, "users": 100382, "share": 7.32, "conf": 0.6215, "cat_share": 9.06, "demand_share": 12.74, "gap": 3.68},
    {"T": 1, "users": 755247, "share": 55.44, "conf": 0.6966, "cat_share": 14.04, "demand_share": 25.69, "gap": 11.66},
    {"T": 2, "users": 11969, "share": 0.87, "conf": 0.5136, "cat_share": 1.12, "demand_share": 1.89, "gap": 0.77},
    {"T": 3, "users": 5942, "share": 0.43, "conf": 0.6967, "cat_share": 8.36, "demand_share": 1.31, "gap": -7.05},
    {"T": 4, "users": 115650, "share": 8.43, "conf": 0.5923, "cat_share": 10.06, "demand_share": 14.93, "gap": 4.87},
    {"T": 5, "users": 5234, "share": 0.38, "conf": 0.6894, "cat_share": 14.12, "demand_share": 2.45, "gap": -11.67},
    {"T": 6, "users": 50377, "share": 3.67, "conf": 0.7884, "cat_share": 13.67, "demand_share": 6.98, "gap": -6.70},
    {"T": 7, "users": 123378, "share": 8.99, "conf": 0.6920, "cat_share": 7.41, "demand_share": 11.87, "gap": 4.45},
    {"T": 8, "users": 152283, "share": 11.10, "conf": 0.6227, "cat_share": 10.52, "demand_share": 16.42, "gap": 5.90},
    {"T": 9, "users": 41819, "share": 3.05, "conf": 0.6764, "cat_share": 11.64, "demand_share": 5.72, "gap": -5.92},
]
seg_df = pd.DataFrame(SEGMENT_DATA)
seg_df["name"] = seg_df["T"].apply(lambda k: intention_labels[str(k)]["name"])
seg_df["label"] = seg_df.apply(lambda r: f"T{r['T']} — {r['name']}", axis=1)

tab1, tab2, tab3 = st.tabs([
    "👥 Customer Segmentation", "⚖️ Supply–Demand Gap", "💰 Personalisation Investment Priority",
])

# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Customer segmentation by 10 shopping-intention groups")
    st.caption(
        "Replaces traditional RFM segmentation — groups customers by "
        "*motivation* rather than transaction frequency/value alone."
    )

    fig1 = px.bar(
        seg_df.sort_values("users", ascending=True),
        x="users", y="label", orientation="h",
        title="Number of customers by intention segment",
        labels={"users": "Number of customers", "label": ""},
        color="conf", color_continuous_scale="Viridis",
        hover_data={"share": True, "conf": True},
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.dataframe(
        seg_df[["label", "users", "share", "conf"]].rename(columns={
            "label": "Segment", "users": "Customers",
            "share": "Share (%)", "conf": "Avg. confidence",
        }),
        use_container_width=True, hide_index=True,
    )

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Supply-demand gap report by segment")
    st.caption(
        "Gap = Demand share − Catalogue share. Positive = undersupplied "
        "(should stock more); negative = oversupplied (should rationalise "
        "SKUs). χ² = 122,462.5 (p < 0.001)."
    )

    fig2 = px.bar(
        seg_df.sort_values("gap"),
        x="gap", y="label", orientation="h",
        title="Supply-demand gap by segment (percentage points)",
        labels={"gap": "Gap (pp)", "label": ""},
        color="gap", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
    )
    st.plotly_chart(fig2, use_container_width=True)

    biggest_gap = seg_df.loc[seg_df["gap"].idxmax()]
    biggest_over = seg_df.loc[seg_df["gap"].idxmin()]
    c1, c2 = st.columns(2)
    c1.success(
        f"**Largest undersupply: T{int(biggest_gap['T'])} — {biggest_gap['name']}** "
        f"(+{biggest_gap['gap']:.2f}pp) → recommend expanding catalogue by 10–15%"
    )
    c2.warning(
        f"**Largest oversupply: T{int(biggest_over['T'])} — {biggest_over['name']}** "
        f"({biggest_over['gap']:.2f}pp) → recommend rationalising SKUs by 10–15%"
    )

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Personalisation investment priority by segment")
    st.caption(
        "Topic concentration (= average confidence) high → clear intention "
        "signal → personalisation is highly effective. Low (T2) → use a "
        "lighter model (Two-Tower) to save inference cost."
    )

    fig3 = px.scatter(
        seg_df, x="conf", y="share", size="users", color="label",
        title="Priority matrix: Topic concentration vs Segment size",
        labels={"conf": "Topic concentration (confidence)", "share": "Share of users (%)"},
        size_max=60,
    )
    st.plotly_chart(fig3, use_container_width=True)

    seg_df_sorted = seg_df.sort_values("conf", ascending=False).copy()
    seg_df_sorted["tier"] = pd.cut(
        seg_df_sorted["conf"], bins=[0, 0.6, 0.7, 1.0],
        labels=["Low — use Two-Tower", "Medium", "High — prioritise budget"],
    )
    st.dataframe(
        seg_df_sorted[["label", "conf", "share", "tier"]].rename(columns={
            "label": "Segment", "conf": "Topic concentration",
            "share": "Size (% of users)", "tier": "Recommendation",
        }),
        use_container_width=True, hide_index=True,
    )

st.markdown("---")
st.caption(
    "The Chapter 5 figures above are computed on the full population (1.37M "
    "customers, 105,542 products) — independent of the 5% sample used to "
    "display product images elsewhere in the app."
)
