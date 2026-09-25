"""
app.py — Home Page
================================================================================
Main entry point of the Streamlit multipage app. Presents the project
overview, core business value, and navigation to the 4 sub-pages.
================================================================================
"""

import streamlit as st
from utils.data_loader import download_data, load_sampling_report

st.set_page_config(
    page_title="Intention-Aware Fashion Recommender | H&M Demo",
    page_icon="🛍️",
    layout="wide",
)

download_data()  # download data once, on cold container start

st.title("🛍️ Deep Learning-Driven BI for Personalized Fashion Retail")
st.caption(
    "Interactive prototype for the Master's thesis — *Integrating Intention "
    "Analytics and Recommendation System* · Three-Tower Neural Network on H&M data"
)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
report = load_sampling_report()

col1.metric("Test AUC (Three-Tower)", "0.8199", "+3.52% vs Two-Tower")
col2.metric("Demo products", f"{report['total_sampled_articles']:,}",
            f"{report['overall_sample_rate_achieved']*100:.1f}% of catalogue")
col3.metric("Shopping-intention segments", "10", "LDA (K=10), ANOVA p<0.001")
col4.metric("Recall improvement", "+6.00%", "vs Two-Tower baseline")

st.markdown("---")

st.header("The business problem")
st.markdown(
    """
Today's fashion e-commerce recommender systems answer **"what does the
customer buy"** well (collaborative filtering) and **"who is the customer"**
(demographic features), but are largely blind to **"why does the customer
buy"** — the underlying psychological motivation behind the purchase.

This prototype demonstrates how a **Three-Tower Neural Network** architecture
(Visual + Semantic + **Intention Alignment**) answers all three questions at
once, and, more importantly, how that insight is turned into **concrete
business value**: more accurate recommendations, customer segmentation by
motivation rather than purchase behaviour alone, and a supply-demand gap
report for the buying team.
"""
)

st.header("Measured business value (from Chapter 4–5 of the thesis)")

biz_col1, biz_col2 = st.columns(2)
with biz_col1:
    st.subheader("📈 Model performance")
    st.markdown(
        """
- **AUC 0.8199** (+3.52% over the Two-Tower baseline, p < 0.001)
- **Recall +6.00%** — retrieves more of what customers actually want to buy,
  not just re-ranking items already surfaced
- **Largest gains** in segments with narrow, well-defined shopping
  intentions: T3 (Infant & Baby Care, +10.5%), T7 (Personal Comfort &
  Intimate Care, +10.2%), T9 (Professional Menswear, +9.2%)
- **User cold-start**: AUC 0.8449 for customers with no purchase history —
  still outperforms Two-Tower thanks to the Bayesian population prior
"""
    )
with biz_col2:
    st.subheader("💼 Business Intelligence")
    st.markdown(
        """
- **Intention-based customer segmentation** (10 groups) instead of
  traditional RFM alone — enables tailored marketing strategy per segment
- **Supply-demand gap report**: T1 (Everyday Wear) undersupplied by
  11.66pp, T5 (Kids' Wear) oversupplied by 11.67pp — a concrete basis for
  buying decisions
- **Personalisation investment prioritisation** by topic concentration per
  segment, avoiding wasted budget on segments with weak intention signal (T2)
"""
    )

st.markdown("---")

st.header("Explore the app")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
with nav_col1:
    st.page_link("pages/1_Product_Catalog.py", label="🛍️ Product Catalog", icon="🛍️")
    st.caption("Browse products by 10 shopping-intention segments, with real images")
with nav_col2:
    st.page_link("pages/2_Recommendation_Demo.py", label="🎯 Recommendation Demo", icon="🎯")
    st.caption("Direct comparison: Three-Tower vs Two-Tower for any persona")
with nav_col3:
    st.page_link("pages/3_Business_Intelligence.py", label="📊 BI Dashboard", icon="📊")
    st.caption("Segmentation, supply-demand gap, personalisation investment priority")
with nav_col4:
    st.page_link("pages/4_About_Methodology.py", label="ℹ️ Methodology", icon="ℹ️")
    st.caption("Model architecture, data sources, and demo sampling method")

st.markdown("---")
st.caption(
    "Demo data is a stratified random sample (5% per intention segment, same "
    "methodology as Section 11 of the thesis) drawn from the H&M "
    "Personalized Fashion Recommendations Dataset (Kaggle, 2022) — for "
    "academic/demo purposes only, not representative of H&M's full live "
    "catalogue."
)
