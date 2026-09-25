"""
Page 4 — About / Methodology
================================================================================
A transparency page: explains the model architecture, data sources, and
in particular the 5% SAMPLING METHOD used for this demo — so a thesis
committee/viewer understands this is not "faked for looks" but follows the
same scientific procedure already used and validated in the thesis
(Section 11).
================================================================================
"""

import streamlit as st
import pandas as pd

from utils.data_loader import download_data, load_sampling_report

st.set_page_config(page_title="About & Methodology", page_icon="ℹ️", layout="wide")
download_data()

st.title("ℹ️ Methodology & Data Transparency")

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
is directly attributable to Tower 3's contribution (thesis Section 3.4.4).

Result: **AUC 0.8199 vs 0.7921** (+3.52%, p < 0.001, Cohen's d = 0.321 —
medium effect), with Tower 3 accounting for only **2,784 of 1,685,729
parameters (0.2%)**.
"""
)

st.header("2. Data source")
st.markdown(
    """
[H&M Personalized Fashion Recommendations Dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
(Kaggle, 2022): 105,542 products, 1,371,980 customers, 31.8 million
transactions (09/2018 – 09/2020). Used for research/academic purposes.
"""
)

st.header("3. Sampling method for this demo")
report = load_sampling_report()

st.markdown(
    f"""
This demo **does not use all 105,542 products** (due to deployment size
constraints on the free tier), but a **{report['sample_rate_target']*100:.0f}%
stratified sample per dominant product intention** — the **same method and
rate** already used and validated in thesis Section 11 (not a new, arbitrary
choice).
"""
)

col1, col2, col3 = st.columns(3)
col1.metric("Total original products", f"{report['total_population_articles']:,}")
col2.metric("Products in this demo", f"{report['total_sampled_articles']:,}")
col3.metric("Achieved sample rate", f"{report['overall_sample_rate_achieved']*100:.2f}%")

st.metric(
    "Mean Absolute Deviation (distribution deviation vs. population)",
    f"{report['mean_absolute_deviation_pp']:.4f} pp",
    help="Closer to 0 means better representativeness of the true "
         "population. The acceptance threshold used in the thesis "
         "(Section 11) is under 0.5pp.",
)

st.subheader("Sampling breakdown by shopping-intention segment")
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

st.info(
    "The Business Intelligence page (📊) uses figures computed on the "
    "**full population**, unaffected by the 5% sampling above — only the "
    "product catalogue images and the recommendation demo use this 5% sample."
)

st.header("4. Known limitations")
st.markdown(
    """
- **Item cold-start** (entirely new products) could not be fully evaluated
  on the current test split, due to the negative-sampling design in
  Section 3.3.2. See thesis Chapter 6.4 for details and the proposed fix.
- The app uses **real personas** (sampled from user_intention_weights.csv)
  rather than all 1.37M customers, to keep the deployment lightweight — this
  does not affect model correctness, only the number of selectable personas
  in the demo.
- Results reflect **offline evaluation** on historical data; no live online
  A/B testing has been conducted yet.
"""
)

st.markdown("---")
st.caption("Source: FQW_DoThiHien_BABDS241 — Deep Learning-Driven Business Intelligence "
           "for Personalized Fashion Retail: Integrating Intention Analytics and "
           "Recommendation System.")
