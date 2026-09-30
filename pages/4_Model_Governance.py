"""
pages/2_Model_Governance.py — Model Governance
================================================================================
The compliance/technical-audit page: architecture, data provenance, and the
5% sampling methodology. Deliberately positioned last in the navigation and
kept plain (no card decoration) — its job is legibility and trust for
whoever needs to check "how was this built", not day-to-day use.
================================================================================
"""

import streamlit as st
import pandas as pd

from utils.data_loader import download_data, load_sampling_report
from utils.theme import inject_global_css, render_sidebar_chrome, thin_rule

st.set_page_config(page_title="Model Governance", page_icon="🗂️", layout="wide")
inject_global_css()
download_data()
render_sidebar_chrome()

st.title("Model Governance")
st.caption("Architecture, data provenance, and sampling methodology — for technical review and compliance.")

st.header("1. Model architecture")
st.markdown(
    """
**Three-Tower Neural Network** — the thesis's central technical contribution:

| Tower | Input | Role |
|---|---|---|
| Tower 1 — Visual | ResNet-50 GAP, 2048-dim | *"What does this product look like?"* |
| Tower 2 — Semantic | BLIP + Sentence-BERT (384) + demographics (3) | *"Does the product's meaning match the customer's profile?"* |
| Tower 3 — Intention Alignment | Hadamard product (item ⊙ user), 10-dim | *"Does the product's purpose match why this customer usually buys?"* |

The **Two-Tower** baseline keeps Tower 1 + Tower 2, **drops Tower 3** — a
controlled experimental design ensuring any measured performance difference
is directly attributable to Tower 3 (thesis Section 3.4.4).

Result: **AUC 0.8199 vs 0.7921** (+3.52%, p < 0.001, Cohen's d = 0.321 —
medium effect), with Tower 3 accounting for only **2,784 of 1,685,729
parameters (0.2%)**.
"""
)

thin_rule()
st.header("2. Data source")
st.markdown(
    """
[H&M Personalized Fashion Recommendations Dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
(Kaggle, 2022): 105,542 products, 1,371,980 customers, 31.8 million
transactions (09/2018 – 09/2020).
"""
)

thin_rule()
st.header("3. Sampling method for this console's product imagery")
report = load_sampling_report()
st.markdown(
    f"""
Product thumbnails throughout this console use a **{report['sample_rate_target']*100:.0f}%
stratified sample per dominant intention** — the same method and rate
validated in thesis Section 11. **Revenue and customer figures are NOT
affected** — those are computed on the full population.
"""
)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total products", f"{report['total_population_articles']:,}")
col2.metric("Stratified 5% sample", f"{report['total_sampled_articles_stratified']:,}")
col3.metric("+ from real customer purchases", f"+{report['extra_articles_from_real_purchases']:,}")
col4.metric("Total in this console", f"{report['total_sampled_articles_final']:,}")
st.caption(
    "The stratified 5% covers general browsing; the extra articles ensure "
    "every real purchase in the Customer Validation page can be scored, "
    "even if it wasn't part of the original 5% draw."
)
col5, = st.columns(1)
col5.metric("Mean Absolute Deviation (sample vs. population)", f"{report['mean_absolute_deviation_pp']:.4f}pp")

with st.expander("Full sampling breakdown by segment"):
    per_intent_df = pd.DataFrame(report["per_intention"])
    st.dataframe(
        per_intent_df.rename(columns={
            "intention": "T", "name": "Segment name",
            "population_articles": "Products (population)", "population_share_pct": "Population %",
            "sampled_articles": "Products (sample)", "sample_share_pct": "Sample %",
            "deviation_pp": "Deviation (pp)",
        }),
        use_container_width=True, hide_index=True,
    )

thin_rule()
st.header("4. Known limitations")
st.markdown(
    """
- **Item cold-start** could not be fully evaluated on the current test
  split — see thesis Chapter 6.4.
- Revenue/customer figures on the Command Center are computed on the full
  population; **product imagery** is limited to the 5% demo sample.
- Results reflect **offline evaluation**; no live online A/B testing has
  been conducted.
- **Two non-interchangeable demand metrics are used in this console:**
  the supply-demand gap shown as a 🔴/🟡/🟢 badge (Table 5.2) is
  **transaction-count-based** (share of purchase events); the revenue
  trend shown on the Command Center and Segment Detail pages is
  **revenue-weighted** (share of monetary value), computed independently
  from transaction dates. The two differ systematically by segment —
  premium segments (e.g. T8) show a higher revenue share than
  transaction-count share, and low-price segments (e.g. T1) show the
  reverse — because one unit of a premium item contributes more revenue
  than one unit of a basic item. Both are legitimate business lenses; they
  should not be read as the same figure measured at two points in time.
"""
)

thin_rule()
st.caption("Source: FQW_DoThiHien_BABDS241 — Deep Learning-Driven Business Intelligence "
           "for Personalized Fashion Retail: Integrating Intention Analytics and "
           "Recommendation System.")
