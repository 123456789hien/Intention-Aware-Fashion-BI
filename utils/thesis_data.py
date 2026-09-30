"""
thesis_data.py
================================================================================
Verified static figures from thesis Chapter 4 (Table 4.7) and Chapter 5
(Table 5.1, 5.2) — computed once from the real notebooks earlier in this
project and reused everywhere, instead of being duplicated per-page.

These are NOT re-derivable from the 5% demo sample (they're population-
level figures), so they are stated explicitly here rather than computed at
runtime — exactly as documented on the Model Governance page.
================================================================================
"""

# Table 5.1 / 5.2 — per-segment population figures
SEGMENT_STATIC = {
    0: {"users": 100382, "user_share": 7.32, "confidence": 0.6215, "cat_share": 9.06, "demand_share": 12.74, "gap": 3.68},
    1: {"users": 755247, "user_share": 55.44, "confidence": 0.6966, "cat_share": 14.04, "demand_share": 25.69, "gap": 11.66},
    2: {"users": 11969, "user_share": 0.87, "confidence": 0.5136, "cat_share": 1.12, "demand_share": 1.89, "gap": 0.77},
    3: {"users": 5942, "user_share": 0.43, "confidence": 0.6967, "cat_share": 8.36, "demand_share": 1.31, "gap": -7.05},
    4: {"users": 115650, "user_share": 8.43, "confidence": 0.5923, "cat_share": 10.06, "demand_share": 14.93, "gap": 4.87},
    5: {"users": 5234, "user_share": 0.38, "confidence": 0.6894, "cat_share": 14.12, "demand_share": 2.45, "gap": -11.67},
    6: {"users": 50377, "user_share": 3.67, "confidence": 0.7884, "cat_share": 13.67, "demand_share": 6.98, "gap": -6.70},
    7: {"users": 123378, "user_share": 8.99, "confidence": 0.6920, "cat_share": 7.41, "demand_share": 11.87, "gap": 4.45},
    8: {"users": 152283, "user_share": 11.10, "confidence": 0.6227, "cat_share": 10.52, "demand_share": 16.42, "gap": 5.90},
    9: {"users": 41819, "user_share": 3.05, "confidence": 0.6764, "cat_share": 11.64, "demand_share": 5.72, "gap": -5.92},
}

# Table 4.7 — per-intention AUC improvement (Three-Tower vs Two-Tower),
# relative %, from the real Part D notebook output verified earlier.
MODEL_IMPROVEMENT_PCT = {
    0: 8.83, 1: 2.92, 2: 7.61, 3: 10.47, 4: 5.48,
    5: 5.88, 6: 6.84, 7: 10.17, 8: 4.55, 9: 9.24,
}


def segment_static(k: int) -> dict:
    return SEGMENT_STATIC.get(int(k), {})


def model_improvement(k: int) -> float:
    return MODEL_IMPROVEMENT_PCT.get(int(k), 0.0)


def recommendation_text(k: int) -> str:
    """Rule-based recommendation text — same logic already used in thesis
    Section 5.4, applied programmatically per segment rather than written
    as four static paragraphs."""
    s = segment_static(k)
    gap = s["gap"]
    conf = s["confidence"]
    share = s["user_share"]

    if gap > 5:
        supply_msg = f"Undersupplied by {gap:+.2f}pp — increase SKU allocation by 10-15%."
    elif gap < -5:
        supply_msg = f"Oversupplied by {gap:.2f}pp — rationalise SKUs by 10-15%."
    else:
        supply_msg = f"Supply roughly matches demand ({gap:+.2f}pp) — maintain current allocation."

    if conf > 0.65 and share < 2:
        invest_msg = "High intention confidence in a small segment — strong candidate for automated, low-risk lifecycle marketing."
    elif conf < 0.55:
        invest_msg = "Weak intention signal — prefer the lighter Two-Tower model here to save inference cost; personalisation ROI is likely low."
    else:
        invest_msg = "Moderate signal strength — standard personalisation investment is appropriate."

    return f"{supply_msg} {invest_msg}"
