"""
pages/3_Recommendation_Audit.py — Recommendation Engine Audit
================================================================================
Segment-level QA: audits the Three-Tower model across EVERY real persona in
a chosen segment at once, not one customer at a time — averaging across
~10 real personas gives a far more statistically robust read than any single
persona (individual-item deltas are known to be noisy — see the note below).
Individual-customer, ground-truth validation lives on the Customer
Validation page; this page is purely about the model's behaviour.
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
    "Audits the Three-Tower model across every real persona in a segment at "
    "once — a QA check on the engine's behaviour, independent of the "
    "period-based business reports on the other pages."
)

articles = load_articles()
personas = load_demo_personas()
intention_labels = load_intention_labels()

# ---- Segment selection only — no persona filter needed ----
default_k = st.session_state.get("selected_segment", 0)
options = [f"T{k} — {intention_labels[str(k)]['name']}" for k in range(10)]
choice = st.selectbox("Segment", options, index=default_k)
k = int(choice.split("—")[0].strip()[1:])
accent = intention_color(k)

seg_personas = personas[personas["dominant_intention"] == k].reset_index(drop=True)
if len(seg_personas) == 0:
    st.warning("No personas available for this segment to audit.")
    st.stop()

top_n = st.slider("Recommendations to inspect per persona", 4, 16, 8, step=4)
st.caption(f"Will run for all {len(seg_personas)} real personas in this segment.")

if st.button("Run audit", type="primary"):
    three_path, two_path = model_paths()
    three_model, two_model = load_models(three_path, two_path)
    visual_feat, semantic_feat, art_feat_idx = load_feature_matrices()

    all_results = []
    progress = st.progress(0.0, text="Scoring personas...")
    for i, (_, p_row) in enumerate(seg_personas.iterrows()):
        user_intention = p_row[[f"intention_{j}" for j in range(10)]].values.astype(np.float32)
        user_demo = np.array(
            [p_row.get("age", 30.0), p_row.get("FN", 0.0), p_row.get("Active", 0.0)], dtype=np.float32
        )
        top, _ = score_catalog(
            three_model, two_model, visual_feat, semantic_feat, art_feat_idx,
            articles, user_intention, user_demo, top_n=top_n,
        )
        top = top.copy()
        top["persona_label"] = p_row["persona_label"]
        top["customer_id"] = p_row["customer_id"]
        all_results.append(top)
        progress.progress((i + 1) / len(seg_personas), text=f"Scored {i+1}/{len(seg_personas)} personas...")
    progress.empty()

    st.session_state[f"audit_result_{k}"] = pd.concat(all_results, ignore_index=True)

result_key = f"audit_result_{k}"
if result_key in st.session_state:
    pooled = st.session_state[result_key]
    n_strong = int((pooled["top_alignment_value"] >= ALIGNMENT_STRONG_THRESHOLD).sum())

    thin_rule()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg. Three-Tower score", f"{pooled['three_tower_score'].mean():.3f}")
    m2.metric("Avg. Two-Tower score", f"{pooled['two_tower_score'].mean():.3f}")
    m3.metric("Average delta", f"{pooled['score_delta'].mean():+.3f}")
    m4.metric("Strong intention matches", f"{n_strong}/{len(pooled)}")

    with st.expander("ℹ️ Why can individual deltas be negative?"):
        st.caption(
            "Three-Tower wins on average (+3.52% AUC, thesis Table 4.6), but "
            "individual items near the score ceiling (0.90+) can flip sign — "
            "consistent with McNemar's test (Table 4.9): Two-Tower is actually "
            "correct in 35.5% of disagreements. **Strong intention match** = "
            "Hadamard alignment ≥ 0.10 — Tower 3 only activates strongly for "
            "products that truly match a shopper's motivation.\n\n"
            "Note: the two models are trained separately, so a strong match "
            "doesn't guarantee Three-Tower wins that item — Tower 3's value "
            "shows up as a rescue for items Tower 1+2 rate low, not a bonus "
            "stacked on items Tower 1+2 already rate high."
        )

    # ---- At a glance ----
    st.markdown("##### At a glance")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        persona_summary = pooled.groupby("persona_label").agg(
            three=("three_tower_score", "mean"), two=("two_tower_score", "mean"),
        ).reset_index()
        persona_summary["persona_short"] = persona_summary["persona_label"].str.slice(0, 28)
        cmp_long = persona_summary.melt(
            id_vars="persona_short", value_vars=["three", "two"], var_name="Model", value_name="Score"
        )
        cmp_long["Model"] = cmp_long["Model"].map({"three": "Three-Tower", "two": "Two-Tower"})
        fig_cmp = px.bar(
            cmp_long, x="Score", y="persona_short", color="Model", orientation="h", barmode="group",
            color_discrete_map={"Three-Tower": THREAD, "Two-Tower": "#8A8578"},
            title="Avg. score by persona",
        )
        fig_cmp.update_layout(height=max(220, 32 * len(persona_summary)), margin=dict(l=10, r=10, t=30, b=10),
                               yaxis_title="", legend_title="")
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False}, key=f"audit_cmp_{k}")

    with chart_col2:
        avg_t1, avg_t2, avg_t3 = pooled["tower1_mag"].mean(), pooled["tower2_mag"].mean(), pooled["tower3_mag"].mean()
        fig_avg = tower_contribution_chart(avg_t1, avg_t2, avg_t3)
        fig_avg.update_layout(title="Average tower contribution (whole segment)", height=220,
                               margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_avg, use_container_width=True, config={"displayModeBar": False}, key=f"audit_avg_{k}")

        match_counts = pd.DataFrame({
            "Type": ["🎯 Strong intention match", "🖼️ Visual/semantic only"],
            "Count": [n_strong, len(pooled) - n_strong],
        })
        fig_match = px.bar(
            match_counts, x="Count", y="Type", orientation="h",
            color="Type", color_discrete_map={
                "🎯 Strong intention match": "#5B7065", "🖼️ Visual/semantic only": "#8A8578",
            },
        )
        fig_match.update_layout(showlegend=False, height=140, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="")
        st.plotly_chart(fig_match, use_container_width=True, config={"displayModeBar": False}, key=f"audit_match_{k}")

    thin_rule()
    st.markdown("##### Per-persona summary")
    summary_table = pooled.groupby("persona_label").agg(
        avg_3T=("three_tower_score", "mean"),
        avg_2T=("two_tower_score", "mean"),
        avg_delta=("score_delta", "mean"),
        strong_matches=("top_alignment_value", lambda s: int((s >= ALIGNMENT_STRONG_THRESHOLD).sum())),
        n_items=("article_id", "count"),
    ).reset_index().rename(columns={"persona_label": "Persona"})
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    thin_rule()
    st.markdown("##### Item-by-item detail (per persona)")
    sort_by_alignment = st.checkbox("Sort by intention-alignment strength (instead of score)", value=False)

    for customer_id_val, group in pooled.groupby("customer_id", sort=False):
        persona_display = f"{group['persona_label'].iloc[0]} · id {customer_id_val[-6:]}"
        with st.expander(f"{persona_display} ({len(group)} items)"):
            display_df = group.sort_values("top_alignment_value", ascending=False) if sort_by_alignment else group
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
                    st.plotly_chart(
                        tf, use_container_width=True, config={"displayModeBar": False},
                        key=f"audit_tower_{customer_id_val}_{product['article_id']}",
                    )
                thin_rule()
else:
    st.info(f"Click **Run audit** to score all {len(seg_personas)} personas in this segment.")
