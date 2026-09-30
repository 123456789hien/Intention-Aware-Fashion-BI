"""
theme.py
================================================================================
Design system for the app: a single source of truth for colors, type, and
the small set of custom HTML/CSS components (chips, cards, stat strip) used
across pages. Grounded in the thesis's own subject matter rather than a
generic dashboard kit:

- The 10 intention segments already have names and a natural narrative
  (Chapter 4.2) — they become the app's categorical color system, used
  consistently everywhere (chips, radar charts, card borders) instead of a
  default Plotly colorway.
- User/item intention vectors are 10-dimensional — a genuine fit for a
  radar/spider chart (utils/charts.py), not a generic bar chart.
================================================================================
"""
from __future__ import annotations

import streamlit as st

INK = "#181A20"
CANVAS = "#FAFAF8"
CANVAS_RAISED = "#F1EFE9"
THREAD = "#3B3873"       # primary brand — "indigo dye"
SIGNAL = "#C9932E"       # single accent, used sparingly
MUTED = "#8A8578"

# One consistent colour per intention segment (T0-T9), used everywhere:
# chips, radar chart traces, card accents, BI charts.
INTENTION_COLORS = {
    0: "#A65B8C",  # T0 — Ladieswear Full Body: Special Occasion Dressing
    1: "#3B3873",  # T1 — Ladieswear Upper Body: Everyday Workwear Comfort (largest segment -> brand primary)
    2: "#2B2B2E",  # T2 — Unisex Dark Basics: Utilitarian Necessity Purchase
    3: "#C9932E",  # T3 — Baby Full Body: Infant & Nurturing Care
    4: "#5B7065",  # T4 — Unisex Lower Body: Functional Versatility Seeking
    5: "#4F8FA6",  # T5 — Children's Upper Body: Trendy & Casual Provisioning
    6: "#C23B5E",  # T6 — Ladies Accessories & Footwear: Hedonic Purchase
    7: "#B98CA6",  # T7 — Ladieswear Underwear: Intimate Self-Care
    8: "#8A6D3B",  # T8 — Ladieswear Knitwear: Premium Quality Investment
    9: "#35506B",  # T9 — Menswear Shirts: Professional Identity Expression
}

# A one-line "psychological lens" per segment — descriptive framing already
# implicit in the segment names chosen in thesis Chapter 4, surfaced here as
# design content rather than decoration.
INTENTION_LENS = {
    0: "Occasion-driven self-presentation",
    1: "Habitual comfort — low-friction repeat purchase",
    2: "Utilitarian, necessity-driven purchase",
    3: "Caregiving & attachment-driven purchase",
    4: "Functional utility over aesthetics",
    5: "Parental provisioning — trendy, casual purchases for children",
    6: "Hedonic reward & impulse purchase",
    7: "Private, intimate self-care",
    8: "Aspirational, quality-driven investment",
    9: "Professional identity construction",
}


def intention_color(k: int) -> str:
    return INTENTION_COLORS.get(int(k), MUTED)


def intention_lens(k: int) -> str:
    return INTENTION_LENS.get(int(k), "")


def inject_global_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Libre+Caslon+Display&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans', sans-serif;
        }}
        .stApp {{
            background-color: {CANVAS};
            color: {INK};
        }}
        h1, h2, h3 {{
            font-family: 'Libre Caslon Display', serif !important;
            color: {INK} !important;
            letter-spacing: -0.01em;
        }}
        p, li {{
            color: {INK};
        }}

        /* ---- Buttons: single brand color, no default red ---- */
        .stButton > button[kind="primary"] {{
            background-color: {THREAD};
            border: none;
            border-radius: 4px;
            font-weight: 600;
            color: white !important;
        }}
        .stButton > button[kind="primary"] * {{
            color: white !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: #2C2A56;
        }}
        .stButton > button[kind="primary"]:hover * {{
            color: white !important;
        }}

        /* ---- Intention chip (Home hero taxonomy strip) ---- */
        .chip-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 28px 0; }}
        .chip {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 14px; border-radius: 100px;
            font-size: 0.85rem; font-weight: 500; color: white;
            white-space: nowrap;
        }}
        .chip .dot {{
            width: 7px; height: 7px; border-radius: 50%;
            background: rgba(255,255,255,0.85);
        }}

        /* ---- Editorial stat strip (Home) ---- */
        .stat-strip {{
            display: flex; flex-wrap: wrap; gap: 0;
            border-top: 1px solid {INK}22; border-bottom: 1px solid {INK}22;
            margin: 20px 0 32px 0;
        }}
        .stat-item {{
            flex: 1; min-width: 160px; padding: 18px 20px;
            border-right: 1px solid {INK}22;
        }}
        .stat-item:last-child {{ border-right: none; }}
        .stat-number {{
            font-family: 'Libre Caslon Display', serif;
            font-size: 2.1rem; color: {THREAD}; line-height: 1.1;
        }}
        .stat-label {{ font-size: 0.85rem; color: {MUTED}; margin-top: 4px; }}

        /* ---- Lookbook product card (Product Catalog) ---- */
        .lookbook-card {{
            border-left: 4px solid var(--card-accent, {THREAD});
            background: white; border-radius: 2px;
            padding: 10px 12px 12px 12px; margin-bottom: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        .lookbook-name {{ font-weight: 600; font-size: 0.92rem; margin: 8px 0 2px 0; }}
        .lookbook-meta {{ font-size: 0.78rem; color: {MUTED}; }}

        /* ---- Persona identity card (Recommendation Demo) ---- */
        .persona-card {{
            background: {CANVAS_RAISED};
            border-radius: 4px; padding: 20px 22px;
            border-top: 3px solid var(--persona-accent, {THREAD});
            margin-bottom: 12px;
        }}
        .persona-title {{ font-family: 'Libre Caslon Display', serif; font-size: 1.3rem; }}
        .persona-lens {{ font-style: italic; color: {MUTED}; font-size: 0.9rem; margin-top: 2px; }}

        /* ---- Section divider ---- */
        .thin-rule {{ border: none; border-top: 1px solid {INK}22; margin: 28px 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def chip_row_html(intention_labels: dict) -> str:
    chips = "".join(
        f'<span class="chip" style="background:{intention_color(k)}">'
        f'<span class="dot"></span>T{k} — {intention_labels[str(k)]["name"]}</span>'
        for k in range(10)
    )
    return f'<div class="chip-row">{chips}</div>'


def stat_strip_html(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="stat-item"><div class="stat-number">{value}</div>'
        f'<div class="stat-label">{label}</div></div>'
        for value, label in items
    )
    return f'<div class="stat-strip">{cells}</div>'


def thin_rule():
    st.markdown('<hr class="thin-rule">', unsafe_allow_html=True)


# ============================================================================
# Admin-console chrome: sidebar identity + shared period selector
# ============================================================================

def render_sidebar_chrome():
    """Renders the fixed sidebar identity block. Call once per page."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="font-family:'Libre Caslon Display',serif; font-size:1.3rem; color:{THREAD};">
                🧵 H&M — Intention Console
            </div>
            <div style="font-size:0.82rem; color:{MUTED}; margin-bottom:16px;">
                Signed in as: Merchandising &amp; Buying Team
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_period_selector(monthly_df) -> tuple[str, str, str]:
    """Renders granularity / period / compare-mode controls in the sidebar
    and returns (granularity, selected_period, compare_mode). Persists the
    selection in st.session_state so it stays consistent as the user moves
    between Command Center and Segment Detail. Granularity "All time"
    collapses the entire date range into one view (no period picker, no
    comparison — there's nothing to compare "all time" against)."""
    from utils.trends import with_period_columns, period_options

    df = with_period_columns(monthly_df)
    with st.sidebar:
        st.markdown("**Reporting period**")
        granularity = st.radio("Granularity", ["Month", "Quarter", "Year", "All time"], horizontal=True, key="granularity")

        if granularity == "All time":
            st.caption("Showing totals across the entire dataset (Sep 2018 – Sep 2020).")
            st.markdown("<hr style='margin:14px 0; border-color:#181A2022;'>", unsafe_allow_html=True)
            return granularity, "All", "N/A — all time has no prior period"

        options = period_options(df, granularity)
        if not options:
            st.caption("⚠️ monthly_segment_trends.csv not found yet — re-run the Script 01 bonus step and re-upload data_export.zip to Drive.")
            return granularity, None, "Previous period"
        default_idx = len(options) - 1
        period = st.selectbox("Period", options, index=default_idx, key=f"period_{granularity}")
        compare_options = ["Previous period"] + (["Same period last year"] if granularity != "Year" else [])
        compare_mode = st.radio("Compare against", compare_options, key="compare_mode")
        st.markdown("<hr style='margin:14px 0; border-color:#181A2022;'>", unsafe_allow_html=True)
    return granularity, period, compare_mode


def status_badge(gap_pp: float) -> tuple[str, str, str]:
    """Simple, reproducible traffic-light rule from the supply-demand gap:
    returns (emoji, label, hex_color)."""
    if gap_pp is None:
        return "⚪", "No data", MUTED
    if gap_pp > 5:
        return "🔴", "Undersupplied", "#C23B5E"
    if gap_pp < -5:
        return "🟡", "Oversupplied", "#C9932E"
    return "🟢", "Balanced", "#5B7065"


def trend_arrow(delta_pp) -> str:
    if delta_pp is None:
        return "→ n/a"
    if delta_pp > 0.05:
        return f"↑ +{delta_pp:.2f}pp"
    if delta_pp < -0.05:
        return f"↓ {delta_pp:.2f}pp"
    return "→ flat"


# Threshold separating a genuine intention-driven match from a coincidental
# one. Chosen from observed score clustering (Section on Recommendation
# Audit): true within-segment matches score ~0.11-0.19, unrelated items
# score ~0.02-0.07 — 0.10 sits cleanly between the two clusters.
ALIGNMENT_STRONG_THRESHOLD = 0.10


def alignment_badge(value: float) -> tuple[str, str]:
    """Returns (label, hex_color) for a Hadamard alignment score."""
    if value >= ALIGNMENT_STRONG_THRESHOLD:
        return "🎯 Strong intention match", "#5B7065"
    return "🖼️ Visual/semantic match only", "#8A8578"
