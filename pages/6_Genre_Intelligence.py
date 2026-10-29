# coding: utf-8
"""pages/6_Genre_Intelligence.py -- Genre-level market map, audio fingerprints, hit rates."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.utils import (
    load_data, apply_theme, inject_css,
    page_header, divider, section_title, insight_box, COLORS, FEATURES,
)
from src.ui import top_nav, page_footer

st.set_page_config(
    page_title="Genre Intelligence · Spotify Intelligence",
    page_icon="🎸", layout="wide",
)
inject_css()
top_nav("Genre Intelligence")
page_header("Genre Intelligence", "Market maps, audio fingerprints, and hit rates across 114 genres", "🎸")

df = load_data()

# ── Precompute genre stats ─────────────────────────────────────────
genre_stats = (
    df.groupby("track_genre")
    .agg(
        track_count  = ("track_name",     "count"),
        avg_pop      = ("popularity",      "mean"),
        hit_rate     = ("is_hit",          "mean"),
        avg_energy   = ("energy",          "mean"),
        avg_dance    = ("danceability",    "mean"),
        avg_valence  = ("valence",         "mean"),
        avg_acoustic = ("acousticness",    "mean"),
        avg_tempo    = ("tempo",           "mean"),
    )
    .reset_index()
)
genre_stats["hit_rate_pct"] = genre_stats["hit_rate"] * 100
genre_stats = genre_stats[genre_stats["track_count"] >= 5]  # remove tiny genres

# ── Tab selector ───────────────────────────────────────────────────
active_tab = st.radio(
    "Genre View",
    ["🌍  Market Map", "🏆  Hit Rate Rankings", "🎵  Audio Fingerprints", "📊  Genre Deep-Dive"],
    horizontal=True,
    label_visibility="collapsed",
)
st.html('<div class="custom-divider" style="margin:0 0 1.5rem;"></div>')

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Market Map
# ══════════════════════════════════════════════════════════════════
if active_tab == "🌍  Market Map":
    section_title("Genre Market Map — Popularity vs Energy (bubble size = track count)")

    fig = px.scatter(
        genre_stats, x="avg_energy", y="avg_pop",
        size="track_count", color="hit_rate_pct",
        hover_name="track_genre",
        size_max=55,
        color_continuous_scale=[
            [0.0, "#C62828"], [0.4, "#E53935"],
            [0.6, "#FFA42B"], [0.8, "#1DB954"], [1.0, "#1ED760"],
        ],
        labels={
            "avg_energy":    "Avg Energy",
            "avg_pop":       "Avg Popularity",
            "hit_rate_pct":  "Hit Rate %",
            "track_count":   "Track Count",
        },
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color="rgba(255,255,255,0.2)")))
    apply_theme(fig, height=560, title="Genre Market Map")
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text="Hit Rate %", font=dict(color="#A0A8B8")),
            tickfont=dict(color="#A0A8B8"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    divider()

    # Top 10 table
    section_title("Top 10 Genres by Avg Popularity")
    top10 = genre_stats.nlargest(10, "avg_pop").reset_index(drop=True)
    for i, row in top10.iterrows():
        bar_w = f"{row['avg_pop']:.0f}%"
        st.html(f"""
        <div class="glass-card" style="padding:0.7rem 1rem; margin-bottom:6px;">
          <div style="display:flex; align-items:center; gap:12px;">
            <div class="rank-badge" style="width:28px; height:28px; font-size:0.75rem;">{i+1}</div>
            <div style="flex:1; min-width:0;">
              <div style="font-weight:600; font-size:0.9rem; color:#FFF;
                          overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                {row['track_genre']}
              </div>
              <div style="background:rgba(255,255,255,0.06); border-radius:50px; height:5px; margin-top:5px;">
                <div style="width:{bar_w}; height:100%; background:#1DB954; border-radius:50px;"></div>
              </div>
            </div>
            <div style="text-align:right; flex-shrink:0;">
              <span class="pop-badge">{row['avg_pop']:.1f}</span>
              <div style="font-size:0.72rem; color:#A0A8B8; margin-top:4px;">
                {row['hit_rate_pct']:.1f}% hits · {int(row['track_count']):,} tracks
              </div>
            </div>
          </div>
        </div>""")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — Hit Rate Rankings
# ══════════════════════════════════════════════════════════════════
elif active_tab == "🏆  Hit Rate Rankings":
    top_n = st.slider("Show top / bottom N genres", 5, 30, 15, 1)
    top_hit   = genre_stats.nlargest(top_n, "hit_rate_pct")
    bot_hit   = genre_stats.nsmallest(top_n, "hit_rate_pct")

    t_col, b_col = st.columns(2, gap="large")

    with t_col:
        section_title(f"Top {top_n} by Hit Rate")
        tdf = top_hit.sort_values("hit_rate_pct", ascending=True)
        fig = go.Figure(go.Bar(
            x=tdf["hit_rate_pct"], y=tdf["track_genre"], orientation="h",
            marker=dict(
                color=tdf["hit_rate_pct"],
                colorscale=[[0, "#0E5C2C"], [0.5, "#1DB954"], [1, "#1ED760"]],
                showscale=False,
            ),
            text=[f"{v:.1f}%" for v in tdf["hit_rate_pct"]],
            textposition="outside", textfont=dict(color="#A0A8B8", size=10),
        ))
        apply_theme(fig, height=max(300, top_n * 26), title=f"Top {top_n} Hit-Rate Genres")
        fig.update_layout(xaxis_title="Hit Rate %", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with b_col:
        section_title(f"Bottom {top_n} by Hit Rate")
        bdf = bot_hit.sort_values("hit_rate_pct", ascending=True)
        fig2 = go.Figure(go.Bar(
            x=bdf["hit_rate_pct"], y=bdf["track_genre"], orientation="h",
            marker=dict(
                color=bdf["hit_rate_pct"],
                colorscale=[[0, "#C62828"], [0.5, "#F037A5"], [1, "#E53935"]],
                showscale=False,
            ),
            text=[f"{v:.1f}%" for v in bdf["hit_rate_pct"]],
            textposition="outside", textfont=dict(color="#A0A8B8", size=10),
        ))
        apply_theme(fig2, height=max(300, top_n * 26), title=f"Bottom {top_n} Hit-Rate Genres")
        fig2.update_layout(xaxis_title="Hit Rate %", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    divider()

    # Hit vs not-hit volume stacked
    section_title("Hit vs Not-Hit Volume — Top 20 Genres by Track Count")
    top20 = genre_stats.nlargest(20, "track_count")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="Hits", x=top20["track_genre"],
        y=(top20["hit_rate"] * top20["track_count"]).round(),
        marker_color="#1DB954", opacity=0.9,
    ))
    fig3.add_trace(go.Bar(
        name="Not Hits", x=top20["track_genre"],
        y=((1 - top20["hit_rate"]) * top20["track_count"]).round(),
        marker_color="#F037A5", opacity=0.75,
    ))
    apply_theme(fig3, height=400, title="Hit vs Not-Hit Track Counts — Top 20 Genres by Volume")
    fig3.update_layout(
        barmode="stack",
        xaxis_tickangle=-35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Audio Fingerprints
# ══════════════════════════════════════════════════════════════════
elif active_tab == "🎵  Audio Fingerprints":
    all_genres_sorted = sorted(genre_stats["track_genre"].tolist())
    sel_genres = st.multiselect(
        "Compare genres (pick 2–6)",
        all_genres_sorted,
        default=all_genres_sorted[:4] if len(all_genres_sorted) >= 4 else all_genres_sorted,
        max_selections=6,
    )

    if len(sel_genres) < 2:
        st.warning("Select at least 2 genres to compare.")
    else:
        radar_feats = ["danceability", "energy", "valence", "acousticness", "speechiness", "instrumentalness", "liveness"]
        fig = go.Figure()
        for i, genre in enumerate(sel_genres):
            gdf = df[df["track_genre"] == genre]
            vals = [gdf[f].mean() for f in radar_feats]
            vals_closed = vals + [vals[0]]
            labels_closed = [f.replace("_", " ").title() for f in radar_feats] + [radar_feats[0].replace("_", " ").title()]
            fig.add_trace(go.Scatterpolar(
                r=vals_closed, theta=labels_closed,
                fill="toself",
                fillcolor=f"{COLORS[i % len(COLORS)]}25",
                line=dict(color=COLORS[i % len(COLORS)], width=2.2),
                name=genre,
                marker=dict(size=5),
            ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(18,24,32,0.6)",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="rgba(255,255,255,0.07)",
                                tickfont=dict(color="#A0A8B8", size=8)),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.07)",
                                 tickfont=dict(color="#A0A8B8", size=11)),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF", family="Inter, sans-serif"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#A0A8B8"),
                        orientation="h", yanchor="bottom", y=-0.15),
            height=520,
            margin=dict(t=40, b=60, l=40, r=40),
            title=dict(text="Genre Audio Fingerprint Comparison",
                       font=dict(color="#FFFFFF", size=15), x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

        divider()

        # Side-by-side feature bars
        section_title("Side-by-Side Feature Comparison")
        comp_data = {
            genre: {f: df[df["track_genre"] == genre][f].mean() for f in radar_feats}
            for genre in sel_genres
        }
        comp_df2 = pd.DataFrame(comp_data).T.reset_index()
        comp_df2.columns = ["Genre"] + radar_feats

        fig2 = go.Figure()
        for i, feat in enumerate(radar_feats):
            fig2.add_trace(go.Bar(
                name=feat.replace("_", " ").title(),
                x=comp_df2["Genre"], y=comp_df2[feat],
                marker_color=COLORS[i % len(COLORS)], opacity=0.85,
                text=[f"{v:.2f}" for v in comp_df2[feat]],
                textposition="outside", textfont=dict(size=9, color="#A0A8B8"),
            ))
        apply_theme(fig2, height=400, title="Average Audio Features by Genre")
        fig2.update_layout(barmode="group", xaxis_tickangle=-20,
                           legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — Genre Deep-Dive
# ══════════════════════════════════════════════════════════════════
elif active_tab == "📊  Genre Deep-Dive":
    all_g = sorted(df["track_genre"].dropna().unique())
    sel_g = st.selectbox("Select a genre to explore", all_g)
    gdf   = df[df["track_genre"] == sel_g]
    gstat = genre_stats[genre_stats["track_genre"] == sel_g].iloc[0]

    # KPI row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.html(f"""
        <div class="kpi-card" style="border-left-color:#1DB954;">
          <div class="kpi-label">Track Count</div>
          <div class="kpi-value glow-metric" style="color:#1DB954;">{int(gstat['track_count']):,}</div>
          <div class="kpi-sub">Tracks in dataset</div>
        </div>""")
    with m2:
        st.html(f"""
        <div class="kpi-card" style="border-left-color:#FFA42B;">
          <div class="kpi-label">Avg Popularity</div>
          <div class="kpi-value glow-metric" style="color:#FFA42B;">{gstat['avg_pop']:.1f}</div>
          <div class="kpi-sub">Out of 100</div>
        </div>""")
    with m3:
        st.html(f"""
        <div class="kpi-card" style="border-left-color:#F037A5;">
          <div class="kpi-label">Hit Rate</div>
          <div class="kpi-value glow-metric" style="color:#F037A5;">{gstat['hit_rate_pct']:.1f}%</div>
          <div class="kpi-sub">Tracks with pop &gt; 50</div>
        </div>""")
    with m4:
        st.html(f"""
        <div class="kpi-card" style="border-left-color:#4B8BFF;">
          <div class="kpi-label">Avg Tempo</div>
          <div class="kpi-value glow-metric" style="color:#4B8BFF;">{gstat['avg_tempo']:.0f} BPM</div>
          <div class="kpi-sub">Beats per minute</div>
        </div>""")

    divider()

    ch1, ch2 = st.columns(2)
    with ch1:
        fig = px.histogram(gdf, x="popularity", nbins=30, color_discrete_sequence=["#1DB954"])
        fig.add_vline(x=50, line_dash="dash", line_color="#F037A5",
                      annotation_text=" Hit threshold", annotation_font_color="#F037A5")
        apply_theme(fig, height=320, title=f"{sel_g} — Popularity Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        feat_avgs = {f.replace("_", " ").title(): gdf[f].mean()
                     for f in ["danceability", "energy", "valence", "acousticness", "speechiness"]}
        fig2 = go.Figure(go.Bar(
            x=list(feat_avgs.keys()), y=list(feat_avgs.values()),
            marker=dict(
                color=list(feat_avgs.values()),
                colorscale=[[0, "#0E5C2C"], [0.5, "#1DB954"], [1, "#1ED760"]],
            ),
            text=[f"{v:.3f}" for v in feat_avgs.values()],
            textposition="outside", textfont=dict(color="#A0A8B8", size=11),
        ))
        apply_theme(fig2, height=320, title=f"{sel_g} — Key Audio Features")
        fig2.update_layout(yaxis_range=[0, 1.2], xaxis_title="Feature", yaxis_title="Avg Value")
        st.plotly_chart(fig2, use_container_width=True)

    divider()

    section_title(f"Top 10 {sel_g} Tracks by Popularity")
    top10_g = gdf.nlargest(10, "popularity")[["track_name", "artists", "popularity", "is_hit"]]
    for i, (_, row) in enumerate(top10_g.iterrows(), 1):
        hit_badge = '<span class="badge">HIT</span>' if row["is_hit"] else '<span class="badge badge-miss">MISS</span>'
        st.html(f"""
        <div class="glass-card" style="padding:0.6rem 1rem; margin-bottom:5px;
             display:flex; align-items:center; gap:12px;">
          <div class="rank-badge" style="width:26px; height:26px; font-size:0.75rem;">{i}</div>
          <div style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
            <span style="color:#FFF; font-weight:600; font-size:0.88rem;">{row['track_name']}</span>
            <span style="color:#A0A8B8; font-size:0.78rem;"> &middot; {row['artists']}</span>
          </div>
          <div style="display:flex; gap:6px; align-items:center; flex-shrink:0;">
            {hit_badge}
            <span class="pop-badge">{int(row['popularity'])}</span>
          </div>
        </div>""")

    divider()
    insight_box(
        f"<strong style='color:#1DB954;'>{sel_g}</strong> has a hit rate of "
        f"<strong>{gstat['hit_rate_pct']:.1f}%</strong> with an average popularity of "
        f"<strong>{gstat['avg_pop']:.1f}</strong>. "
        f"Average energy is <strong>{gstat['avg_energy']:.2f}</strong> and danceability is "
        f"<strong>{gstat['avg_dance']:.2f}</strong>. "
        "Use these baselines to benchmark individual tracks against their genre peers."
    )
page_footer()
