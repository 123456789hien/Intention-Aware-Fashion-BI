"""
Page 2 — Recommendation Demo
================================================================================
The core interactive feature of the app: pick an existing "persona" (a real
user from demo_personas.csv) OR simulate a brand-new cold-start user, then
run BOTH models (Three-Tower & Two-Tower) over the full sampled catalogue,
showing the Top-N recommendations along with a Hadamard-alignment
explanation (the Explainable Intention Layer).

This is a live, interactive demonstration of the thesis's central finding
(Chapter 4): the Three-Tower model delivers a measurable improvement over
Two-Tower, especially pronounced for personas in narrow-intention segments
(T3, T7, T9).
================================================================================
"""

import os
import numpy as np
import streamlit as st
import plotly.express as px

from utils.data_loader import (
    download_data, load_articles, load_article_intention_profiles,
    load_feature_matrices, load_demo_personas, load_intention_labels,
    load_sampling_report, model_paths, image_path,
)
from utils.models import load_models
from utils.recommender import score_catalog, explain_recommendation

st.set_page_config(page_title="Recommendation Demo", page_icon="🎯", layout="wide")
download_data()

st.title("🎯 Recommendation Demo: Three-Tower vs Two-Tower")

articles = load_articles()
visual_feat, semantic_feat, art_feat_idx = load_feature_matrices()
personas = load_demo_personas()
intention_labels = load_intention_labels()
report = load_sampling_report()
three_path, two_path = model_paths()
three_model, two_model = load_models(three_path, two_path)

intention_cols = [f"intention_{k}" for k in range(10)]
articles_full = articles

# ============================================================================
# CHOOSE A USER
# ============================================================================
st.header("Step 1 — Choose a user to generate recommendations for")

mode = st.radio(
    "Mode", ["Real persona (from trained data)", "New customer (cold-start)"],
    horizontal=True,
)

if mode.startswith("Real"):
    persona_choice = st.selectbox("Choose a persona", personas["persona_label"].tolist())
    persona_row = personas[personas["persona_label"] == persona_choice].iloc[0]
    user_intention = persona_row[intention_cols].values.astype(np.float32)
    age = float(persona_row.get("age", 30.0)) if "age" in persona_row else 30.0
    fn = float(persona_row.get("FN", 0.0)) if "FN" in persona_row else 0.0
    active = float(persona_row.get("Active", 0.0)) if "Active" in persona_row else 0.0
    st.info(
        f"A real persona with **{int(persona_row['n_purchases'])}** historical purchases, "
        f"confidence = **{persona_row['confidence']:.2f}**. This is NOT synthetic data — "
        f"it's a profile computed via real Bayesian smoothing (Section 3.4.5)."
    )
else:
    st.markdown(
        "Simulating a brand-new customer with no purchase history — the system "
        "falls back to the **Bayesian global prior** (Section 3.4.5) as the "
        "default profile, exactly as the real system handles cold-start."
    )
    archetype_choice = st.selectbox(
        "Which segment does this customer most identify with? (optional, "
        "helps with initial personalisation)",
        ["No selection — use the global prior only"] +
        [f"T{k} — {intention_labels[str(k)]['name']}" for k in range(10)],
    )
    global_prior = np.array(
        [report["global_prior"][f"intention_{k}"] for k in range(10)], dtype=np.float32
    )
    if archetype_choice.startswith("No selection"):
        user_intention = global_prior
    else:
        k = int(archetype_choice.split("—")[0].strip()[1:])
        # Blend slightly toward the chosen archetype (alpha=0.3) — simulates
        # a weak initial signal (e.g. from an onboarding quiz), while still
        # keeping the global prior as the dominant component
        boosted = global_prior.copy()
        boosted[k] += 0.3
        user_intention = boosted / boosted.sum()

    age = st.slider("Age", 16, 80, 30)
    fn = st.selectbox("Subscribed to fashion newsletter (FN)?", [0, 1], index=0)
    active = st.selectbox("Active account (Active)?", [0, 1], index=1)

# ============================================================================
# RUN THE MODELS
# ============================================================================
st.header("Step 2 — Compare recommendations")

top_n = st.slider("Number of recommendations", 4, 24, 12, step=4)

if st.button("🔮 Generate recommendations", type="primary"):
    with st.spinner("Scoring the full sampled catalogue with both models..."):
        user_demo = np.array([age, fn, active], dtype=np.float32)
        top, full_scored = score_catalog(
            three_model, two_model, visual_feat, semantic_feat, art_feat_idx,
            articles_full, user_intention, user_demo, top_n=top_n,
        )
    st.session_state["last_top"] = top
    st.session_state["last_full"] = full_scored

if "last_top" in st.session_state:
    top = st.session_state["last_top"]
    full_scored = st.session_state["last_full"]

    # ---- Comparison summary ----
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg. Three-Tower score (Top-N)", f"{top['three_tower_score'].mean():.3f}")
    m2.metric("Avg. Two-Tower score (Top-N)", f"{top['two_tower_score'].mean():.3f}")
    m3.metric("Average delta", f"{top['score_delta'].mean():+.3f}")

    fig = px.bar(
        top.sort_values("three_tower_score"),
        x="three_tower_score",
        y="prod_name" if "prod_name" in top.columns else "article_id",
        orientation="h",
        title="Predicted purchase probability — Three-Tower (Top-N)",
        labels={"three_tower_score": "Purchase probability", "prod_name": "Product"},
        color="score_delta", color_continuous_scale="RdYlGn",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top recommendations (Three-Tower) with explanations")
    cols_per_row = 4
    rows = [top.iloc[i:i + cols_per_row] for i in range(0, len(top), cols_per_row)]
    for row_df in rows:
        cols = st.columns(cols_per_row)
        for col, (_, product) in zip(cols, row_df.iterrows()):
            with col:
                img_path = image_path(product["article_id"])
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                st.markdown(f"**{str(product.get('prod_name', 'Product'))[:35]}**")
                st.caption(
                    f"3T: {product['three_tower_score']:.3f} · "
                    f"2T: {product['two_tower_score']:.3f} · "
                    f"Δ {product['score_delta']:+.3f}"
                )
                with st.popover("Why this recommendation?"):
                    st.markdown(explain_recommendation(product, intention_labels))
else:
    st.info("Choose a user in Step 1, then click **Generate recommendations** to see results.")
