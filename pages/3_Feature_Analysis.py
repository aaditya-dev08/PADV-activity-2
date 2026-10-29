# coding: utf-8
"""pages/3_Feature_Analysis.py -- Audio feature deep-dive with pill tabs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from src.utils import (
    load_data, apply_theme, inject_css,
    page_header, divider, section_title, insight_box, COLORS, FEATURES,
)
from src.ui import top_nav, page_footer

st.set_page_config(
    page_title="Feature Analysis · Spotify Intelligence",
    page_icon="🎚️", layout="wide",
)
inject_css()
top_nav("Feature Analysis")
page_header("Feature Analysis", "Deep-dive into the audio DNA of 89,741 tracks", "🎚️")

df = load_data()

# ── Pill-tab navigation (st.radio styled as pills via CSS) ──────────
active_tab = st.radio(
    "Tab",
    ["🔗  Relationships", "🌡️  Correlation", "📦  Distributions", "📈  Trends"],
    horizontal=True,
    label_visibility="collapsed",
)
st.html('<div class="custom-divider" style="margin:0 0 1.5rem;"></div>')

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Relationships
# ══════════════════════════════════════════════════════════════════
if active_tab == "🔗  Relationships":
    sample = df.sample(5000, random_state=42)

    fig = px.scatter(
        sample, x="energy", y="danceability", color="popularity",
        size="popularity", size_max=11,
        color_continuous_scale=[[0, "#0A0E14"], [0.4, "#1A3A2A"], [0.7, "#1DB954"], [1, "#1ED760"]],
        labels={"energy": "Energy", "danceability": "Danceability", "popularity": "Popularity"},
        opacity=0.65,
    )
    apply_theme(fig, height=500, title="Energy vs Danceability — 5,000 track sample, size & color = Popularity")
    fig.update_layout(coloraxis_colorbar=dict(
        title="Popularity",
        tickfont=dict(color="#A0A8B8"),
        font=dict(color="#A0A8B8"),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Feature vs popularity scatter grid (mini)
    divider()
    section_title("Each Feature vs Popularity")
    cols_feats = ["danceability", "energy", "valence", "acousticness"]
    fc1, fc2, fc3, fc4 = st.columns(4)
    for col_widget, feat in zip([fc1, fc2, fc3, fc4], cols_feats):
        samp2 = df[[feat, "popularity"]].sample(2000, random_state=7)
        f2 = px.scatter(samp2, x=feat, y="popularity", opacity=0.4,
                        color_discrete_sequence=["#1DB954"], trendline="lowess")
        apply_theme(f2, height=260, title=feat.replace("_", " ").title())
        f2.update_layout(margin=dict(t=40, b=30, l=30, r=10), showlegend=False)
        f2.update_traces(marker_size=4)
        with col_widget:
            st.plotly_chart(f2, use_container_width=True)

    insight_box(
        "Popular tracks cluster in the <strong style='color:#1DB954;'>high-energy, "
        "high-danceability zone</strong>. Acousticness shows the strongest negative relationship "
        "with popularity — acoustic tracks are significantly less likely to be hits in this dataset."
    )

# ══════════════════════════════════════════════════════════════════
# TAB 2 — Correlation
# ══════════════════════════════════════════════════════════════════
elif active_tab == "🌡️  Correlation":
    corr_cols = FEATURES + ["popularity", "is_hit"]
    corr      = df[corr_cols].corr()
    labels    = [c.replace("_", " ").title() for c in corr_cols]

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[
            [0.0, "#C62828"], [0.3, "#E53935"],
            [0.5, "rgba(18,24,32,0.9)"],
            [0.7, "#1DB954"], [1.0, "#1ED760"],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(size=8, color="#FFFFFF"),
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#A0A8B8"),
            title=dict(text="r", font=dict(color="#A0A8B8")),
        ),
    ))
    apply_theme(fig, height=560, title="Audio Feature Correlation Matrix")
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickfont=dict(size=10, color="#A0A8B8")),
        yaxis=dict(tickfont=dict(size=10, color="#A0A8B8")),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Correlation with popularity mini cards
    divider()
    section_title("Pearson r with Popularity")
    pop_corr = (df[FEATURES + ["popularity"]].corr()["popularity"]
                  .drop("popularity")
                  .sort_values(key=abs, ascending=False))
    mini_cols = st.columns(len(FEATURES))
    for col_w, (feat, val) in zip(mini_cols, pop_corr.items()):
        color = "#1DB954" if val > 0 else "#F037A5"
        with col_w:
            st.html(f"""
            <div class="glass-card" style="text-align:center; padding:0.8rem 0.5rem; border-left:3px solid {color};">
                <div style="font-size:0.68rem; color:#A0A8B8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">
                    {feat.replace("_","<br>")}
                </div>
                <div style="font-size:1.1rem; font-weight:800; color:{color};">{val:+.3f}</div>
            </div>""")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Distributions
# ══════════════════════════════════════════════════════════════════
elif active_tab == "📦  Distributions":
    feats6 = {
        "danceability": "Danceability",
        "energy":       "Energy",
        "valence":      "Valence",
        "acousticness": "Acousticness",
        "speechiness":  "Speechiness",
        "instrumentalness": "Instrumentalness",
    }
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(feats6.values()),
        vertical_spacing=0.14, horizontal_spacing=0.08,
    )
    seg_colors = {"Low": "#F037A5", "Medium": "#FFA42B", "High": "#1DB954"}
    df_plot    = df.dropna(subset=["popularity_segment"])

    for idx, (feat, _) in enumerate(feats6.items()):
        r, c = divmod(idx, 3)
        for seg, color in seg_colors.items():
            vals = df_plot[df_plot["popularity_segment"] == seg][feat]
            fig.add_trace(
                go.Box(
                    y=vals, name=seg, marker_color=color,
                    line_color=color, fillcolor=f"{color}20",
                    showlegend=(idx == 0), boxmean=True,
                    whiskerwidth=0.5,
                ),
                row=r + 1, col=c + 1,
            )

    apply_theme(fig, height=560, title="Audio Feature Distributions by Popularity Segment (Low / Medium / High)")
    for ax in fig.layout:
        if ax.startswith("xaxis"):
            fig.layout[ax].showticklabels = False
    st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "<strong style='color:#1DB954;'>Key finding:</strong> Instrumentalness and acousticness "
        "are markedly <em>lower</em> in High-popularity tracks. Danceability and energy skew "
        "<em>higher</em> for hits. Valence shows little separation — mood alone doesn't predict success."
    )

# ══════════════════════════════════════════════════════════════════
# TAB 4 — Trends
# ══════════════════════════════════════════════════════════════════
elif active_tab == "📈  Trends":
    trend_feats = ["danceability", "energy", "valence", "acousticness", "instrumentalness"]
    df_tr  = df.dropna(subset=["popularity_segment"])
    trend  = (df_tr.groupby("popularity_segment")[trend_feats]
                .mean().reindex(["Low", "Medium", "High"]))

    fig = go.Figure()
    for feat, color in zip(trend_feats, COLORS):
        fig.add_trace(go.Scatter(
            x=["Low", "Medium", "High"], y=trend[feat],
            mode="lines+markers",
            name=feat.replace("_", " ").title(),
            line=dict(color=color, width=2.5),
            marker=dict(size=9, color=color, line=dict(color="#0A0E14", width=2)),
        ))
    apply_theme(fig, height=460,
                title="Audio Feature Trends Across Popularity Segments")
    fig.update_layout(
        xaxis_title="Popularity Segment",
        yaxis_title="Average Value",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    divider()
    section_title("Segment-Level Averages")
    stat_df = trend.T.reset_index()
    stat_df.columns = ["Feature", "Low Popularity", "Medium Popularity", "High Popularity"]
    stat_df = stat_df.round(4)
    st.dataframe(stat_df, use_container_width=True, hide_index=True)

    insight_box(
        "<strong style='color:#1DB954;'>Acousticness</strong> drops steeply from Low → High popularity. "
        "<strong style='color:#1ED760;'>Danceability</strong> and "
        "<strong style='color:#FFA42B;'>Energy</strong> rise moderately. "
        "<strong style='color:#F037A5;'>Instrumentalness</strong> collapses to near zero in hits — "
        "vocal tracks dominate the charts."
    )
page_footer()
