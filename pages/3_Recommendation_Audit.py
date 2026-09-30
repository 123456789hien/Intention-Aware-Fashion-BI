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
import streamlit as st

from utils.data_loader import (
    download_data, load_articles, load_feature_matrices, load_demo_personas,
    load_intention_labels, model_paths, image_path,
)
from utils.models import load_models
from utils.recommender import score_catalog, explain_recommendation
from utils.theme import inject_global_css, intention_color, render_sidebar_chrome, thin_rule, alignment_badge, ALIGNMENT_STRONG_THRESHOLD
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

seg_personas = personas[personas["dominant_intention"] == k]
if len(seg_personas) == 0:
    st.warning("No persona available for this segment to audit.")
    st.stop()

persona_label = st.selectbox("Persona", seg_personas["persona_label"].tolist(), key=f"audit_persona_{k}")
persona_row = seg_personas[seg_personas["persona_label"] == persona_label].iloc[0]

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
    st.caption(
        "⚠️ Individual items can have a negative delta even though Three-Tower "
        "wins on average (+3.52% AUC over 544,178 test interactions, thesis "
        "Table 4.6). With a small sample (a handful of top-ranked items, all "
        "already scoring 0.90+), differences shrink toward the score ceiling "
        "and can flip sign — this matches McNemar's test (Table 4.9): in "
        "35.5% of cases where the two models disagree, Two-Tower is actually "
        "correct. Increase the slider above or try a different persona to see "
        "the aggregate advantage emerge more clearly.\n\n"
        "🎯 **Strong intention match** = Hadamard alignment score ≥ 0.10 — the "
        "product's own dominant intention genuinely overlaps this persona's "
        "profile. Below that, the recommendation is driven mainly by Tower 1 "
        "(visual) and Tower 2 (semantic), with Tower 3 close to neutral — "
        "this is expected: Tower 3 only activates strongly for products that "
        "truly match the shopper's motivation, not every item in the feed."
    )

    sort_by_alignment = st.checkbox("Sort by intention-alignment strength (instead of score)", value=False)
    display_df = top.sort_values("top_alignment_value", ascending=False) if sort_by_alignment else top

    thin_rule()
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
