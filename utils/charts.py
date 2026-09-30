"""
charts.py
================================================================================
Two chart types grounded directly in the thesis's data structures, rather
than generic bar/line charts:

1. Radar chart — a user's or item's 10-dimensional intention vector is a
   natural fit for a spider/radar plot (one axis per T0-T9), immediately
   showing concentration vs. diffusion of shopping intention — the same
   property the thesis calls "confidence" (Section 3.4.5).

2. Tower contribution breakdown — decomposes one recommendation's score
   into the relative magnitude of its three input signals (visual,
   semantic, intention-alignment), a lightweight visual companion to the
   Explainable Intention Layer's text explanation.
================================================================================
"""

import numpy as np
import plotly.graph_objects as go

from utils.theme import intention_color, THREAD


def intention_radar_chart(vector: np.ndarray, intention_labels: dict, title: str = "") -> go.Figure:
    labels = [f"T{k}" for k in range(10)]
    values = list(vector) + [vector[0]]  # close the polygon
    axis_labels = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=axis_labels, fill="toself",
        line=dict(color=THREAD, width=2),
        fillcolor="rgba(59,56,115,0.18)",
        hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(0.35, float(vector.max()) * 1.15)], showticklabels=False),
            angularaxis=dict(rotation=90, direction="clockwise"),
        ),
        showlegend=False,
        title=title,
        margin=dict(l=30, r=30, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def tower_contribution_chart(visual_mag: float, semantic_mag: float, intention_mag: float) -> go.Figure:
    total = max(visual_mag + semantic_mag + intention_mag, 1e-9)
    shares = [visual_mag / total, semantic_mag / total, intention_mag / total]
    names = ["Tower 1 — Visual", "Tower 2 — Semantic", "Tower 3 — Intention"]
    colors = ["#8A8578", "#5B7065", THREAD]

    fig = go.Figure(go.Bar(
        x=shares, y=names, orientation="h",
        marker=dict(color=colors),
        text=[f"{s*100:.0f}%" for s in shares], textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(autorange="reversed"),
        height=150, margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
