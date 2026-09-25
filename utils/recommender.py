"""
recommender.py
================================================================================
Generates recommendations for a user (an existing persona OR a simulated
cold-start user) using both the Three-Tower and Two-Tower models, so the app
can compare them directly — a live illustration of the thesis's central
finding (Chapter 4).

The "Explainable Intention Layer" feature (proposed as a future extension in
Chapter 6 of the thesis) is implemented here: for each Three-Tower
recommendation, the code computes which intention dimension of the Hadamard
vector (item ⊙ user) contributed most, then turns that into a natural-
language explanation for non-technical users/merchandisers.
================================================================================
"""

import numpy as np
import torch
import pandas as pd

DEFAULT_DEMO = np.array([30.0, 0.0, 0.0], dtype=np.float32)


def build_user_vector(intention_probs: np.ndarray, age: float, fn: float, active: float):
    demo = np.array([age, fn, active], dtype=np.float32)
    return intention_probs.astype(np.float32), demo


def score_catalog(
    three_model, two_model,
    visual_feat, semantic_feat, art_feat_idx,
    articles_df: pd.DataFrame,
    user_intention: np.ndarray, user_demo: np.ndarray,
    top_n: int = 12,
):
    """
    Scores purchase probability for the entire sampled catalogue (the 5%
    sample) with both models, returning the top-N by Three-Tower score
    alongside the Two-Tower score for comparison, plus the top-1
    contributing intention (Hadamard alignment) for each item.
    """
    n = len(articles_df)
    vis_batch = np.zeros((n, visual_feat.shape[1]), dtype=np.float32)
    sem_batch = np.zeros((n, semantic_feat.shape[1]), dtype=np.float32)
    item_int_batch = np.zeros((n, 10), dtype=np.float32)

    intention_cols = [f"intention_{k}" for k in range(10)]
    item_int_matrix = articles_df[intention_cols].values.astype(np.float32)

    for i, aid in enumerate(articles_df["article_id"].values):
        fi = art_feat_idx.get(aid, -1)
        if fi >= 0:
            vis_batch[i] = visual_feat[fi]
            sem_batch[i] = semantic_feat[fi]
        item_int_batch[i] = item_int_matrix[i]

    demo_batch = np.tile(user_demo, (n, 1)).astype(np.float32)
    user_int_batch = np.tile(user_intention, (n, 1)).astype(np.float32)

    with torch.no_grad():
        vis_t = torch.from_numpy(vis_batch)
        sem_t = torch.from_numpy(sem_batch)
        demo_t = torch.from_numpy(demo_batch)
        iint_t = torch.from_numpy(item_int_batch)
        uint_t = torch.from_numpy(user_int_batch)

        three_scores, alignment = three_model(vis_t, sem_t, demo_t, iint_t, uint_t)
        two_scores = two_model(vis_t, sem_t, demo_t, iint_t, uint_t)

    three_scores = three_scores.numpy()
    two_scores = two_scores.numpy()
    alignment = alignment.numpy()  # (n, 10) Hadamard item ⊙ user per intention

    result = articles_df.copy().reset_index(drop=True)
    result["three_tower_score"] = three_scores
    result["two_tower_score"] = two_scores
    result["score_delta"] = three_scores - two_scores
    result["top_alignment_intention"] = np.argmax(alignment, axis=1)
    result["top_alignment_value"] = np.max(alignment, axis=1)

    top = result.sort_values("three_tower_score", ascending=False).head(top_n).reset_index(drop=True)
    return top, result


def explain_recommendation(row, intention_labels: dict) -> str:
    """Generates a natural-language explanation for one recommended item
    (the Explainable Intention Layer — a business feature proposed in
    Chapter 6 of the thesis)."""
    k = int(row["top_alignment_intention"])
    name = intention_labels.get(str(k), {}).get("name", f"T{k}")
    val = row["top_alignment_value"]
    delta = row["score_delta"]
    direction = "higher than" if delta > 0 else "lower than"
    return (
        f"Recommended mainly because both you and this product score highly "
        f"on the shopping-intention segment **\"{name}\"** (Hadamard "
        f"alignment score = {val:.3f}). The Three-Tower model's prediction "
        f"for this item is {direction} the Two-Tower model's by {abs(delta):.3f}."
    )
