"""
pages/2_Recommendation_Audit.py — Recommendation Engine Audit
================================================================================
Moved out of Segment Detail into its own page: this is a model QA tool
(inspect exactly what the Three-Tower model recommends and why for a real
persona), not part of the business reporting flow. Reuses the same
score_catalog / explain_recommendation / tower_contribution_chart logic.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from utils.data_loader import (
    download_data, load_articles, load_feature_matrices, load_demo_personas,
    load_intention_labels, model_paths, image_path,
)
from utils.models import load_models
from utils.recommender import score_catalog, explain_recommendation
from utils.theme import inject_global_css, intention_color, render_sidebar_chrome, thin_rule, alignment_badge, ALIGNMENT_STRONG_THRESHOLD, THREAD
from utils.charts import tower_contribution_chart

st.set_page_config(page_title="Recommendation Audit", page_icon="🧪", layout="wide")
inject_global_css()
download_data()
render_sidebar_chrome()

st.title("Recommendation Engine Audit")
st.caption(
    "Pick a real persona and inspect exactly what the Three-Tower model "
    "recommends and why — a QA check on the engine's behaviour, independent "
    "of the period-based business reports on the other pages."
)

articles = load_articles()
personas = load_demo_personas()
intention_labels = load_intention_labels()

# ---- Segment + persona selection ----
default_k = st.session_state.get("selected_segment", 0)
options = [f"T{k} — {intention_labels[str(k)]['name']}" for k in range(10)]
choice = st.selectbox("Segment", options, index=default_k)
k = int(choice.split("—")[0].strip()[1:])
accent = intention_color(k)

seg_personas = personas[personas["dominant_intention"] == k].reset_index(drop=True)
if len(seg_personas) == 0:
    st.warning("No persona available for this segment to audit.")
    st.stop()

# customer_id suffix guarantees a unique label even when confidence/n_purchases
# coincide across different real customers (common in small segments) — the
# raw persona_label alone was not always unique, which silently broke the
# dropdown->row lookup below.
seg_personas["display_label"] = seg_personas.apply(
    lambda r: f"{r['persona_label']} · id {r['customer_id'][-6:]}", axis=1
)
persona_label = st.selectbox("Persona", seg_personas["display_label"].tolist(), key=f"audit_persona_{k}")
persona_row = seg_personas[seg_personas["display_label"] == persona_label].iloc[0]

top_n = st.slider("Number of recommendations to inspect", 4, 16, 8, step=4)

if st.button("Run audit", type="primary"):
    three_path, two_path = model_paths()
    three_model, two_model = load_models(three_path, two_path)
    visual_feat, semantic_feat, art_feat_idx = load_feature_matrices()

    user_intention = persona_row[[f"intention_{i}" for i in range(10)]].values.astype(np.float32)
    user_demo = np.array([
        persona_row.get("age", 30.0), persona_row.get("FN", 0.0), persona_row.get("Active", 0.0),
    ], dtype=np.float32)

    with st.spinner("Scoring catalogue with both models..."):
        top, _ = score_catalog(
            three_model, two_model, visual_feat, semantic_feat, art_feat_idx,
            articles, user_intention, user_demo, top_n=top_n,
        )
    st.session_state[f"audit_result_{k}"] = top

result_key = f"audit_result_{k}"
if result_key in st.session_state:
    top = st.session_state[result_key]

    thin_rule()
    n_strong = int((top["top_alignment_value"] >= ALIGNMENT_STRONG_THRESHOLD).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg. Three-Tower score", f"{top['three_tower_score'].mean():.3f}")
    m2.metric("Avg. Two-Tower score", f"{top['two_tower_score'].mean():.3f}")
    m3.metric("Average delta", f"{top['score_delta'].mean():+.3f}")
    m4.metric("Strong intention matches", f"{n_strong}/{len(top)}")

    with st.expander("ℹ️ Why can individual deltas be negative?"):
        st.caption(
            "Three-Tower wins on average (+3.52% AUC, thesis Table 4.6), but "
            "individual items near the score ceiling (0.90+) can flip sign — "
            "consistent with McNemar's test (Table 4.9): Two-Tower is actually "
            "correct in 35.5% of disagreements. **Strong intention match** = "
            "Hadamard alignment ≥ 0.10 — Tower 3 only activates strongly for "
            "products that truly match this shopper's motivation.\n\n"
            "Note: the two models are trained separately, so a strong match "
            "doesn't guarantee Three-Tower wins that item — Tower 3's value "
            "shows up as a rescue for items Tower 1+2 rate low, not a bonus "
            "stacked on items Tower 1+2 already rate high."
        )

    # ---- Overview charts: the "shape" of this audit, before the detail list ----
    st.markdown("##### At a glance")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        cmp_df = top[["prod_name", "three_tower_score", "two_tower_score"]].copy()
        cmp_df["prod_name"] = cmp_df["prod_name"].astype(str).str[:22]
        cmp_long = cmp_df.melt(id_vars="prod_name", var_name="Model", value_name="Score")
        cmp_long["Model"] = cmp_long["Model"].map({"three_tower_score": "Three-Tower", "two_tower_score": "Two-Tower"})
        fig_cmp = px.bar(
            cmp_long, x="Score", y="prod_name", color="Model", orientation="h", barmode="group",
            color_discrete_map={"Three-Tower": THREAD, "Two-Tower": "#8A8578"},
            title="Score by product",
        )
        fig_cmp.update_layout(height=max(220, 32 * len(top)), margin=dict(l=10, r=10, t=30, b=10),
                               yaxis_title="", legend_title="")
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        avg_t1, avg_t2, avg_t3 = top["tower1_mag"].mean(), top["tower2_mag"].mean(), top["tower3_mag"].mean()
        fig_avg = tower_contribution_chart(avg_t1, avg_t2, avg_t3)
        fig_avg.update_layout(title="Average tower contribution (this feed)", height=220,
                               margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_avg, use_container_width=True, config={"displayModeBar": False})

        match_counts = pd.DataFrame({
            "Type": ["🎯 Strong intention match", "🖼️ Visual/semantic only"],
            "Count": [n_strong, len(top) - n_strong],
        })
        fig_match = px.bar(
            match_counts, x="Count", y="Type", orientation="h",
            color="Type", color_discrete_map={
                "🎯 Strong intention match": "#5B7065", "🖼️ Visual/semantic only": "#8A8578",
            },
        )
        fig_match.update_layout(showlegend=False, height=140, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="")
        st.plotly_chart(fig_match, use_container_width=True, config={"displayModeBar": False})

    sort_by_alignment = st.checkbox("Sort by intention-alignment strength (instead of score)", value=False)
    display_df = top.sort_values("top_alignment_value", ascending=False) if sort_by_alignment else top

    thin_rule()
    st.markdown("##### Item-by-item detail")
    for _, product in display_df.iterrows():
        badge_label, badge_color = alignment_badge(product["top_alignment_value"])
        img_col, text_col, chart_col = st.columns([1, 2, 2])
        with img_col:
            ip = image_path(product["article_id"])
            if os.path.exists(ip):
                st.image(ip, use_container_width=True)
        with text_col:
            st.markdown(
                f"<div style='border-left:3px solid {badge_color}; padding-left:8px;'>"
                f"<b>{str(product.get('prod_name', 'Product'))[:38]}</b><br>"
                f"<span style='color:{badge_color}; font-size:0.82rem;'>{badge_label} "
                f"({product['top_alignment_value']:.3f})</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"3T {product['three_tower_score']:.3f} · 2T {product['two_tower_score']:.3f}")
            st.markdown(explain_recommendation(product, intention_labels))
        with chart_col:
            tf = tower_contribution_chart(product["tower1_mag"], product["tower2_mag"], product["tower3_mag"])
            st.plotly_chart(tf, use_container_width=True, config={"displayModeBar": False})
        thin_rule()
else:
    st.info("Choose a persona above, then click **Run audit**.")
