# coding: utf-8
"""src/utils.py -- Shared utilities for Spotify Intelligence Dashboard."""

import os
import html as _html
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ── Project paths ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── ML feature list ────────────────────────────────────────────────
FEATURES = [
    "danceability", "energy", "loudness", "valence",
    "tempo", "acousticness", "instrumentalness",
    "speechiness", "liveness",
]

# ── Palette ────────────────────────────────────────────────────────
COLORS = ["#1DB954", "#1ED760", "#F037A5", "#FFA42B", "#4B8BFF", "#A855F7", "#22D3EE"]


# ── Data / model loaders ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "spotify_cleaned.csv")
    if not os.path.exists(path):
        st.error("Dataset not found. Run `python src/prepare_data.py` first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model():
    mp = os.path.join(ROOT, "models", "best_model.pkl")
    sp = os.path.join(ROOT, "models", "scaler.pkl")
    if not os.path.exists(mp):
        st.error("Model not found. Run `python src/train_model.py` first.")
        st.stop()
    return joblib.load(mp), joblib.load(sp)


# ── CSS injector ───────────────────────────────────────────────────
def inject_css() -> None:
    css_path = os.path.join(ROOT, "assets", "style.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.html(f"<style>{f.read()}</style>")


# ── Plotly theme ───────────────────────────────────────────────────
def apply_theme(fig, height: int = 420, title: str = None) -> go.Figure:
    """Apply the custom dark Spotify-style Plotly theme to any figure."""
    updates = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18,24,32,0.6)",
        font=dict(family="Inter, sans-serif", color="#FFFFFF", size=12),
        height=height,
        margin=dict(t=55 if title else 35, b=40, l=50, r=25),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#A0A8B8"),
            title_font=dict(color="#A0A8B8"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#A0A8B8"),
            title_font=dict(color="#A0A8B8"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#A0A8B8"),
            bordercolor="rgba(255,255,255,0.08)",
        ),
        colorway=COLORS,
    )
    if title:
        updates["title"] = dict(
            text=title, font=dict(color="#FFFFFF", size=15), x=0.02
        )
    fig.update_layout(**updates)
    return fig


# ── KPI card HTML ──────────────────────────────────────────────────
def kpi_card_html(label: str, value: str, icon: str = "", sub: str = "",
                  color: str = "#1DB954") -> str:
    """Return HTML for a glassmorphism KPI card (no default st.metric)."""
    return f"""
    <div class="kpi-card" style="border-left-color:{color};">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value glow-metric" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


# ── Page header ────────────────────────────────────────────────────
def page_header(title: str, subtitle: str, icon: str = "") -> None:
    st.html(f"""
    <div class="hero-section" style="padding:2rem 2rem 1.2rem; text-align:center;">
        <h1 class="hero-title" style="font-size:clamp(1.8rem,3.5vw,2.4rem);">
            {icon}&nbsp;{title}
        </h1>
        <p class="hero-subtitle" style="margin-bottom:0;">{subtitle}</p>
    </div>""")


# ── Divider ────────────────────────────────────────────────────────
def divider() -> None:
    st.html('<div class="custom-divider"></div>')


# ── Section title ──────────────────────────────────────────────────
def section_title(text: str) -> None:
    st.html(f'<div class="section-title">{text}</div>')


# ── Insight / callout box ──────────────────────────────────────────
def insight_box(text: str, icon: str = "💡") -> None:
    st.html(f"""
    <div class="glass-card" style="border-left:3px solid #1DB954; padding:1.2rem 1.5rem; margin-top:0.8rem;">
        <div style="display:flex; gap:12px; align-items:flex-start;">
            <span style="font-size:1.4rem; flex-shrink:0;">{icon}</span>
            <div style="color:#A0A8B8; font-size:0.9rem; line-height:1.75;">{text}</div>
        </div>
    </div>""")


# ── HTML-safe string ───────────────────────────────────────────────
def esc(s) -> str:
    """Escape special HTML characters in a string."""
    return _html.escape(str(s))


# ── Track card HTML ────────────────────────────────────────────────
def track_card_html(rank: int, name: str, artist: str, genre: str,
                    popularity: int, is_hit: int) -> str:
    hit_badge = '<span class="badge">HIT</span>' if is_hit else '<span class="badge badge-miss">MISS</span>'
    return f"""
    <div class="track-card">
        <div class="rank-badge">{rank}</div>
        <div class="track-info">
            <div class="track-name">{esc(name)}</div>
            <div class="track-artist">{esc(artist)}&nbsp;&middot;&nbsp;
                <span class="feature-tag" style="font-size:0.7rem;">{esc(genre)}</span>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            {hit_badge}
            <div class="pop-badge">{popularity}</div>
        </div>
    </div>"""
