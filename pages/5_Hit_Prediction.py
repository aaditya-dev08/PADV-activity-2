# coding: utf-8
"""pages/5_Hit_Prediction.py -- Real-time hit predictor with gauge, radar, similar tracks."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from src.utils import (
    load_data, load_model, apply_theme, inject_css,
    page_header, divider, section_title, insight_box, FEATURES, esc,
)
from src.ui import top_nav, page_footer

st.set_page_config(
    page_title="Hit Prediction · Spotify Intelligence",
    page_icon="⚡", layout="wide",
)
inject_css()
top_nav("Hit Prediction Studio")
page_header(
    "Hit Prediction Studio",
    "Tune your track's audio DNA — get an instant ML-powered hit probability score",
    "⚡",
)

model, scaler = load_model()
df = load_data()

# ── Two-column layout ───────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.25], gap="large")

# ── LEFT: 9 audio sliders ──────────────────────────────────────────
with left_col:
    st.html('<div class="glass-card">')
    section_title("🎛️  Audio Controls")
    dance    = st.slider("💃  Danceability",         0.0, 1.0,   0.60, 0.01)
    energy   = st.slider("⚡  Energy",               0.0, 1.0,   0.70, 0.01)
    loudness = st.slider("🔊  Loudness (dB)",       -30.0, 0.0,  -5.0, 0.5)
    valence  = st.slider("😊  Valence (mood)",        0.0, 1.0,   0.50, 0.01)
    tempo    = st.slider("🥁  Tempo (BPM)",          50.0, 200.0, 120.0, 1.0)
    acoustic = st.slider("🎸  Acousticness",          0.0, 1.0,   0.20, 0.01)
    instrum  = st.slider("🎹  Instrumentalness",      0.0, 1.0,   0.00, 0.01)
    speech   = st.slider("🗣️  Speechiness",           0.0, 1.0,   0.05, 0.01)
    liveness = st.slider("🏟️  Liveness",              0.0, 1.0,   0.15, 0.01)
    st.html("</div>")

# ── RIGHT: prediction + gauge ──────────────────────────────────────
with right_col:
    user_arr   = np.array([[dance, energy, loudness, valence,
                             tempo, acoustic, instrum, speech, liveness]])
    user_sc    = scaler.transform(user_arr)
    proba      = float(model.predict_proba(user_sc)[0][1])
    is_hit_p   = proba >= 0.5
    pct        = proba * 100
    bar_color  = "#1DB954" if is_hit_p else "#F037A5"
    card_class = "prediction-card-hit" if is_hit_p else "prediction-card-miss"
    emoji      = "✅" if is_hit_p else "❌"
    verdict    = "LIKELY A HIT" if is_hit_p else "LIKELY NOT A HIT"

    st.html(f"""
    <div class="{card_class}">
      <div style="font-size:3rem; margin-bottom:6px;">{emoji}</div>
      <div style="font-family:'Poppins',sans-serif; font-size:1.7rem; font-weight:800;
                  color:{'#1DB954' if is_hit_p else '#F037A5'}; margin-bottom:4px;">
        {verdict}
      </div>
      <div style="font-size:3.2rem; font-weight:900; color:#FFF; line-height:1;">{pct:.1f}%</div>
      <div style="color:#A0A8B8; font-size:0.85rem; margin-bottom:1.2rem;">hit probability</div>
      <div style="background:rgba(255,255,255,0.1); border-radius:50px; height:9px; overflow:hidden;">
        <div style="width:{pct:.0f}%; height:100%;
                    background:linear-gradient(90deg,{bar_color},{bar_color}AA);
                    border-radius:50px; transition:width 0.5s ease;"></div>
      </div>
    </div>""")

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Hit Probability", "font": {"color": "#A0A8B8", "size": 13}},
        number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 34}},
        gauge={
            "axis": {"range": [0, 100],
                     "tickcolor": "#A0A8B8",
                     "tickfont": {"color": "#A0A8B8"}},
            "bar": {"color": bar_color, "thickness": 0.26},
            "bgcolor": "rgba(18,24,32,0.7)",
            "borderwidth": 1,
            "bordercolor": "rgba(29,185,84,0.2)",
            "steps": [
                {"range": [0, 50],  "color": "rgba(240,55,165,0.1)"},
                {"range": [50, 100], "color": "rgba(29,185,84,0.1)"},
            ],
            "threshold": {
                "line": {"color": "#FFFFFF", "width": 2},
                "thickness": 0.78, "value": 50,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#FFFFFF"),
        height=260, margin=dict(t=40, b=10, l=20, r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

divider()

# ── Radar + Similar tracks ─────────────────────────────────────────
radar_col, sim_col = st.columns(2, gap="large")

with radar_col:
    section_title("Audio Fingerprint Radar")
    feat_labels = [
        "Danceability", "Energy", "Loudness\n(norm)",
        "Valence", "Tempo\n(norm)", "Acousticness",
        "Instrumentalness", "Speechiness", "Liveness",
    ]
    vals_norm = [
        dance, energy,
        (loudness + 30) / 30.0,
        valence,
        (tempo - 50) / 150.0,
        acoustic, instrum, speech, liveness,
    ]
    global_avg = [
        df["danceability"].mean(), df["energy"].mean(),
        (df["loudness"].mean() + 30) / 30.0,
        df["valence"].mean(),
        (df["tempo"].mean() - 50) / 150.0,
        df["acousticness"].mean(), df["instrumentalness"].mean(),
        df["speechiness"].mean(), df["liveness"].mean(),
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_norm + [vals_norm[0]],
        theta=feat_labels + [feat_labels[0]],
        fill="toself",
        fillcolor="rgba(29,185,84,0.22)",
        line=dict(color="#1DB954", width=2.5),
        name="Your Track",
        marker=dict(size=6),
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=global_avg + [global_avg[0]],
        theta=feat_labels + [feat_labels[0]],
        fill="toself",
        fillcolor="rgba(255,164,43,0.10)",
        line=dict(color="#FFA42B", width=1.5, dash="dot"),
        name="Dataset Avg",
        marker=dict(size=4),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(18,24,32,0.6)",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="rgba(255,255,255,0.07)",
                            tickfont=dict(color="#A0A8B8", size=8)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.07)",
                             tickfont=dict(color="#A0A8B8", size=9)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF", family="Inter, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#A0A8B8")),
        height=400,
        margin=dict(t=50, b=30, l=30, r=30),
        title=dict(text="Your Track vs Dataset Average",
                   font=dict(color="#FFFFFF", size=13), x=0.5),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with sim_col:
    section_title("5 Most Similar Tracks")

    # Euclidean distance in scaled feature space
    df_feat   = df[FEATURES].fillna(0)
    df_sc_all = scaler.transform(df_feat)
    distances = np.sqrt(((df_sc_all - user_sc) ** 2).sum(axis=1))
    top5_idx  = distances.argsort()[:5]
    sim_df    = df.iloc[top5_idx]

    for i, (_, row) in enumerate(sim_df.iterrows()):
        hit_badge = ('<span class="badge">HIT</span>' if row["is_hit"]
                     else '<span class="badge badge-miss">MISS</span>')
        dist_val  = distances[top5_idx[i]]
        st.html(f"""
        <div class="sim-card">
          <div style="display:flex; align-items:flex-start; gap:10px;">
            <div class="rank-badge">{i+1}</div>
            <div style="flex:1; min-width:0;">
              <div style="font-weight:600; font-size:0.9rem; color:#FFF;
                          overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                {esc(row['track_name'])}
              </div>
              <div style="font-size:0.78rem; color:#A0A8B8; margin:3px 0;
                          overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                {esc(row['artists'])}
              </div>
              <div style="display:flex; gap:6px; align-items:center; margin-top:4px;">
                <span class="feature-tag" style="font-size:0.7rem;">{esc(row['track_genre'])}</span>
                {hit_badge}
              </div>
            </div>
            <div style="text-align:right; flex-shrink:0;">
              <div class="pop-badge">{int(row['popularity'])}</div>
              <div style="font-size:0.68rem; color:#A0A8B8; margin-top:4px;">d={dist_val:.2f}</div>
            </div>
          </div>
        </div>""")

    insight_box(
        "Similarity is computed as Euclidean distance across all 9 scaled audio features. "
        "A lower distance means a closer match to your audio profile.",
        icon="📐",
    )
page_footer()
