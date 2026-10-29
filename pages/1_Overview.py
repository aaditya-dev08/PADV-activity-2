# coding: utf-8
"""pages/1_Overview.py -- Executive Overview dashboard."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.utils import (
    load_data, apply_theme, kpi_card_html, inject_css,
    page_header, divider, section_title, insight_box, COLORS,
)
from src.ui import top_nav, page_footer

st.set_page_config(
    page_title="Overview · Spotify Intelligence",
    page_icon="📊", layout="wide",
)
inject_css()
top_nav("Executive Overview")
page_header("Executive Overview", "Music intelligence at a glance", "📊")

df = load_data()

# ── KPI values ─────────────────────────────────────────────────────
total       = len(df)
hit_rate    = df["is_hit"].mean() * 100
genres      = df["track_genre"].nunique()
avg_pop     = df["popularity"].mean()
avg_energy  = df["energy"].mean()
avg_dance   = df["danceability"].mean()

# ── Row 1 ──────────────────────────────────────────────────────────
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    st.html(kpi_card_html("Total Tracks", f"{total:,}", "🎵", "Cleaned dataset"))
with r1c2:
    st.html(kpi_card_html("Hit Rate", f"{hit_rate:.1f}%", "🏆", "Popularity > 50"))
with r1c3:
    st.html(kpi_card_html("Unique Genres", f"{genres}", "🎸", "Music categories"))

# ── Row 2 ──────────────────────────────────────────────────────────
r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    st.html(kpi_card_html("Avg Popularity", f"{avg_pop:.1f}", "⭐", "Out of 100"))
with r2c2:
    st.html(kpi_card_html("Avg Energy", f"{avg_energy:.2f}", "⚡", "0–1 scale"))
with r2c3:
    st.html(kpi_card_html("Avg Danceability", f"{avg_dance:.2f}", "🕺", "0–1 scale"))

divider()

# ── Chart row 1 ────────────────────────────────────────────────────
section_title("Popularity & Hit Analysis")
ch1, ch2 = st.columns(2)

with ch1:
    fig = px.histogram(df, x="popularity", nbins=50, color_discrete_sequence=["#1DB954"])
    fig.add_vline(x=50, line_dash="dash", line_color="#F037A5",
                  annotation_text=" Hit threshold", annotation_font_color="#F037A5")
    fig.update_traces(marker_line_width=0)
    apply_theme(fig, height=380, title="Popularity Distribution")
    st.plotly_chart(fig, use_container_width=True)

with ch2:
    gp = (df.groupby("track_genre")["popularity"].mean()
            .nlargest(10).reset_index()
            .sort_values("popularity"))
    gp.columns = ["Genre", "Avg Popularity"]
    fig = go.Figure(go.Bar(
        y=gp["Genre"], x=gp["Avg Popularity"], orientation="h",
        marker=dict(
            color=gp["Avg Popularity"],
            colorscale=[[0, "#0E5C2C"], [0.5, "#1DB954"], [1, "#1ED760"]],
            showscale=False,
        ),
        text=[f"{v:.1f}" for v in gp["Avg Popularity"]],
        textposition="outside", textfont=dict(color="#A0A8B8", size=11),
    ))
    apply_theme(fig, height=380, title="Top 10 Genres by Avg Popularity")
    st.plotly_chart(fig, use_container_width=True)

# ── Chart row 2 ────────────────────────────────────────────────────
ch3, ch4 = st.columns(2)

with ch3:
    counts = df["is_hit"].value_counts().reset_index()
    counts.columns = ["is_hit", "count"]
    counts["label"] = counts["is_hit"].map({1: "Hit", 0: "Not Hit"})
    fig = go.Figure(go.Pie(
        labels=counts["label"], values=counts["count"], hole=0.62,
        marker_colors=["#1DB954", "#F037A5"],
        textinfo="percent+label",
        textfont=dict(color="#FFFFFF", size=13),
    ))
    fig.add_annotation(
        text=f"<b>{hit_rate:.1f}%</b><br><span style='font-size:11px'>Hit Rate</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="#1DB954"),
    )
    apply_theme(fig, height=380, title="Hit vs Not-Hit Split")
    st.plotly_chart(fig, use_container_width=True)

with ch4:
    seg_order = ["Low", "Medium", "High"]
    seg = (df["popularity_segment"].value_counts()
             .reindex(seg_order).fillna(0).reset_index())
    seg.columns = ["Segment", "Count"]
    fig = px.bar(
        seg, x="Segment", y="Count", color="Segment",
        color_discrete_map={"Low": "#F037A5", "Medium": "#FFA42B", "High": "#1DB954"},
        text="Count",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", textfont_color="#A0A8B8")
    apply_theme(fig, height=380, title="Tracks by Popularity Segment")
    st.plotly_chart(fig, use_container_width=True)

divider()

# ── Audio feature snapshot ─────────────────────────────────────────
section_title("Audio Feature Snapshot")
feat_cols = ["danceability", "energy", "valence", "acousticness",
             "speechiness", "instrumentalness", "liveness"]
feat_means = df[feat_cols].mean()

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=list(feat_means) + [feat_means.iloc[0]],
    theta=[f.replace("_", " ").title() for f in feat_cols] + [feat_cols[0].replace("_", " ").title()],
    fill="toself",
    fillcolor="rgba(29,185,84,0.18)",
    line=dict(color="#1DB954", width=2.5),
    name="Dataset Average",
))
fig_radar.update_layout(
    polar=dict(
        bgcolor="rgba(18,24,32,0.6)",
        radialaxis=dict(visible=True, range=[0, 1],
                        gridcolor="rgba(255,255,255,0.07)",
                        tickfont=dict(color="#A0A8B8", size=9)),
        angularaxis=dict(gridcolor="rgba(255,255,255,0.07)",
                         tickfont=dict(color="#A0A8B8", size=11)),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FFFFFF", family="Inter, sans-serif"),
    height=380,
    margin=dict(t=50, b=40, l=40, r=40),
    showlegend=False,
)
rc1, rc2, rc3 = st.columns([1, 2, 1])
with rc2:
    st.plotly_chart(fig_radar, use_container_width=True)

divider()

insight_box(
    f"Only <strong style='color:#1DB954;'>{hit_rate:.1f}%</strong> of {total:,} tracks cross the "
    "hit threshold (popularity&nbsp;&gt;&nbsp;50). "
    "Audio features alone explain roughly <strong>10–15% of popularity variance</strong>. "
    "Marketing spend, artist fame, playlist placement, and release timing drive the rest."
)
page_footer()
