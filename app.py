"""
app.py — Spotify Music Intelligence  |  Landing Page
Premium dark dashboard styled like a music streaming app.
"""

import os
import sys
import pandas as pd
import streamlit as st

# ── Page config (MUST be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Spotify Music Intelligence",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ────────────────────────────────────────────────
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
with open(CSS_PATH, "r", encoding="utf-8") as f:
    css_content = f.read()
# Inject CSS — use st.html() for reliability in Streamlit 1.32+
st.html(f"<style>{css_content}</style>")

# ── Load Data ──────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "spotify_cleaned.csv")


@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.html("""
    <div style='padding:16px 0 12px;'>
        <div style='font-size:20px; font-weight:700;
                    background:linear-gradient(135deg,#1DB954,#1ED760);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; font-family:Poppins,sans-serif;'>🎵 Music Intel</div>
        <div style='font-size:12px; color:rgba(255,255,255,0.45); margin-top:3px;'>Decode what makes a hit</div>
        <div style='height:1px; background:linear-gradient(90deg,transparent,#1DB954,transparent);
                    margin:10px 0;'></div>
    </div>
    """)

    st.page_link("app.py",                         label="🏠  Home")
    st.page_link("pages/1_Overview.py",             label="📊  Overview")
    st.page_link("pages/2_Track_Explorer.py",       label="🔍  Track Explorer")
    st.page_link("pages/3_Feature_Analysis.py",     label="🎚️  Feature Analysis")
    st.page_link("pages/4_Model_Evaluation.py",     label="🤖  Model Evaluation")
    st.page_link("pages/5_Hit_Prediction.py",       label="⚡  Hit Prediction")
    st.page_link("pages/6_Genre_Intelligence.py",   label="🎸  Genre Intelligence")
    st.page_link("pages/7_Data_Quality.py",         label="🛡️  Data Quality")

    st.markdown("---")
    with st.expander("📊 Dataset Info"):
        st.markdown("""
**Source:** Kaggle Spotify Tracks

**Size:** 114,000 tracks

**Features:** 20 columns

**Genres:** 114+

**Target:** `is_hit` (pop > 50)
        """)
    with st.expander("🤖 Model Info"):
        st.markdown("""
**Best Model:** Random Forest

**F1-Score:** 0.42

**Recall:** 0.64

**AUC-ROC:** 0.68

**Imbalance:** class_weight='balanced'
        """)
    st.html("""
    <div style='margin-top:20px; padding:8px 12px;
                background:rgba(29,185,84,0.1); border-radius:8px;
                font-size:11px; color:#1DB954; text-align:center;
                border:1px solid rgba(29,185,84,0.25); font-weight:600;
                letter-spacing:0.05em; text-transform:uppercase;'>PADV Activity 2</div>
    """)

# ══════════════════════════════════════════════════════════════════
#  TOP NAV
# ══════════════════════════════════════════════════════════════════
st.html("""
<div class="top-nav">
    <div class="top-nav-logo">🎵 Music Intel</div>
    <div style="color:rgba(255,255,255,0.55); font-size:13px; font-weight:500;">Home</div>
</div>
""")

# ══════════════════════════════════════════════════════════════════
#  HERO SECTION
# ══════════════════════════════════════════════════════════════════
st.html("""
<div class="hero-section fade-in">
    <div class="hero-badge">
        <span class="pulse-dot"></span>
        Live ML Analytics
    </div>
    <h1 class="hero-title">Spotify Music Intelligence</h1>
    <p class="hero-subtitle">
        Decode what makes a hit song — powered by 114k tracks,
        advanced ML models, and real-time audio feature analysis.
    </p>
</div>
""")

# ══════════════════════════════════════════════════════════════════
#  DATA STATUS BANNER  (shown when data not yet prepared)
# ══════════════════════════════════════════════════════════════════
if df is None:
    st.html("""
    <div class="glass-card fade-in-up" style="border-left:3px solid #F59E0B; text-align:center; padding:2rem;">
        <div style="font-size:2rem; margin-bottom:0.5rem;">⚠️</div>
        <div style="font-size:1.1rem; font-weight:600; color:#F59E0B; margin-bottom:0.5rem;">
            Dataset not yet prepared
        </div>
        <div style="color:#A0A8B8; font-size:0.9rem;">
            Run <code style="background:rgba(255,255,255,0.1); padding:2px 8px; border-radius:6px;">
            python src/prepare_data.py</code> then
            <code style="background:rgba(255,255,255,0.1); padding:2px 8px; border-radius:6px;">
            python src/train_model.py</code> to unlock all features.
        </div>
    </div>
    """)
    total_tracks  = "~114K"
    hit_rate      = "~34%"
    unique_genres = "114+"
    avg_pop       = "~33.2"
else:
    total_tracks  = f"{len(df):,}"
    hit_rate      = f"{df['is_hit'].mean()*100:.1f}%" if "is_hit" in df.columns else "N/A"
    unique_genres = str(df["track_genre"].nunique()) if "track_genre" in df.columns else "N/A"
    avg_pop       = f"{df['popularity'].mean():.1f}" if "popularity" in df.columns else "N/A"

# ══════════════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════════════
st.html(f"""
<div class="kpi-grid fade-in-up">

    <div class="kpi-card">
        <span class="kpi-icon">🎵</span>
        <div class="kpi-label">Total Tracks</div>
        <div class="kpi-value glow-metric">{total_tracks}</div>
        <div class="kpi-sub">Unique Spotify songs</div>
    </div>

    <div class="kpi-card">
        <span class="kpi-icon">🏆</span>
        <div class="kpi-label">Hit Rate</div>
        <div class="kpi-value glow-metric">{hit_rate}</div>
        <div class="kpi-sub">Popularity &gt; 50</div>
    </div>

    <div class="kpi-card">
        <span class="kpi-icon">🎸</span>
        <div class="kpi-label">Unique Genres</div>
        <div class="kpi-value glow-metric">{unique_genres}</div>
        <div class="kpi-sub">Music categories</div>
    </div>

    <div class="kpi-card">
        <span class="kpi-icon">⭐</span>
        <div class="kpi-label">Avg Popularity</div>
        <div class="kpi-value glow-metric">{avg_pop}</div>
        <div class="kpi-sub">Out of 100</div>
    </div>

</div>
""")

# ── Divider ────────────────────────────────────────────────────────
st.html('<div class="custom-divider"></div>')

# ══════════════════════════════════════════════════════════════════
#  FEATURE SHOWCASE  (3-column glass cards)
# ══════════════════════════════════════════════════════════════════
st.html('<div class="section-title">What\'s Inside</div>')

c1, c2, c3 = st.columns(3)

with c1:
    st.html("""
    <div class="glass-card fade-in-up">
        <div style="font-size:2rem; margin-bottom:0.8rem;">🤖</div>
        <div style="font-size:1rem; font-weight:700; color:#fff; margin-bottom:0.5rem;">
            ML Hit Predictor
        </div>
        <div style="color:#A0A8B8; font-size:0.85rem; line-height:1.6;">
            Three trained models — Logistic Regression, Random Forest,
            and Gradient Boosting — compete to predict hit songs.
            Best model auto-selected by F1-score.
        </div>
        <div style="margin-top:1rem;">
            <span class="feature-tag">F1 Optimized</span>
            <span class="feature-tag">AUC-ROC</span>
            <span class="feature-tag">Balanced</span>
        </div>
    </div>
    """)

with c2:
    st.html("""
    <div class="glass-card fade-in-up">
        <div style="font-size:2rem; margin-bottom:0.8rem;">🎚️</div>
        <div style="font-size:1rem; font-weight:700; color:#fff; margin-bottom:0.5rem;">
            Audio Feature Analysis
        </div>
        <div style="color:#A0A8B8; font-size:0.85rem; line-height:1.6;">
            Deep-dive into danceability, energy, valence, tempo, and 5 more
            audio features. Discover what separates hits from flops.
        </div>
        <div style="margin-top:1rem;">
            <span class="feature-tag">9 Features</span>
            <span class="feature-tag">Correlations</span>
            <span class="feature-tag">Distributions</span>
        </div>
    </div>
    """)

with c3:
    st.html("""
    <div class="glass-card fade-in-up">
        <div style="font-size:2rem; margin-bottom:0.8rem;">🎸</div>
        <div style="font-size:1rem; font-weight:700; color:#fff; margin-bottom:0.5rem;">
            Genre Intelligence
        </div>
        <div style="color:#A0A8B8; font-size:0.85rem; line-height:1.6;">
            Explore 114+ music genres, their popularity distribution,
            audio fingerprints, and hit rates side-by-side.
        </div>
        <div style="margin-top:1rem;">
            <span class="feature-tag">114+ Genres</span>
            <span class="feature-tag">Radar Charts</span>
            <span class="feature-tag">Benchmarks</span>
        </div>
    </div>
    """)

# ── Divider ────────────────────────────────────────────────────────
st.html('<div class="custom-divider"></div>')
st.html('<div class="section-title">Audio Feature Dictionary</div>')

features_info = [
    ("🕺", "Danceability",     "How suitable a track is for dancing (0–1)"),
    ("⚡", "Energy",           "Intensity and activity level (0–1)"),
    ("🔊", "Loudness",         "Overall loudness in decibels (dB)"),
    ("😊", "Valence",          "Musical positiveness / mood (0–1)"),
    ("🥁", "Tempo",            "Estimated beats per minute (BPM)"),
    ("🎸", "Acousticness",     "Confidence the track is acoustic (0–1)"),
    ("🎹", "Instrumentalness", "Predicts if a track has no vocals (0–1)"),
    ("🗣️", "Speechiness",      "Detects spoken word presence (0–1)"),
    ("🏟️", "Liveness",         "Detects audience presence; live feel (0–1)"),
]

col_a, col_b, col_c = st.columns(3)
cols = [col_a, col_b, col_c]

for i, (icon, name, desc) in enumerate(features_info):
    with cols[i % 3]:
        st.html(f"""
        <div class="glass-card" style="padding:1rem 1.2rem; margin-bottom:0.8rem;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
                <span style="font-size:1.2rem;">{icon}</span>
                <span style="font-weight:700; color:#fff; font-size:0.92rem;">{name}</span>
            </div>
            <div style="color:#A0A8B8; font-size:0.8rem; line-height:1.5;">{desc}</div>
        </div>
        """)

# ══════════════════════════════════════════════════════════════════
#  POPULARITY DISTRIBUTION (if data available)
# ══════════════════════════════════════════════════════════════════
if df is not None and "popularity_segment" in df.columns:
    st.html('<div class="custom-divider"></div>')
    st.html('<div class="section-title">Dataset Snapshot</div>')

    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_THEME = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(18,24,32,0.6)",
        "font":          dict(family="Inter, sans-serif", color="#A0A8B8"),
        "xaxis":         dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        "yaxis":         dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        "colorway":      ["#1DB954", "#1ED760", "#F037A5", "#FFA42B", "#4B8BFF"],
    }

    snap_col1, snap_col2 = st.columns(2)

    with snap_col1:
        pop_counts = df["popularity_segment"].value_counts().reset_index()
        pop_counts.columns = ["Segment", "Count"]
        fig_pie = px.pie(
            pop_counts, values="Count", names="Segment",
            title="Popularity Segments",
            color_discrete_sequence=["#1DB954", "#1565C0", "#E53935"],
            hole=0.55,
        )
        fig_pie.update_layout(
            **PLOTLY_THEME,
            title_font=dict(size=15, color="#FFFFFF"),
            showlegend=True,
            legend=dict(font=dict(color="#A0A8B8")),
            margin=dict(t=50, b=20, l=20, r=20),
        )
        fig_pie.update_traces(textfont=dict(color="#FFFFFF"), textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with snap_col2:
        fig_hist = px.histogram(
            df, x="popularity",
            nbins=50,
            title="Popularity Distribution",
            color_discrete_sequence=["#1DB954"],
        )
        fig_hist.add_vline(
            x=50, line_dash="dash", line_color="#FF5252",
            annotation_text="Hit Threshold",
            annotation_font_color="#FF5252",
        )
        fig_hist.update_layout(
            **PLOTLY_THEME,
            title_font=dict(size=15, color="#FFFFFF"),
            xaxis_title="Popularity Score",
            yaxis_title="Track Count",
            bargap=0.05,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  GET STARTED BUTTON
# ══════════════════════════════════════════════════════════════════
st.html('<div class="custom-divider"></div>')
st.html("""
<div class="get-started-btn">
    <div style="margin-bottom:1rem; color:#A0A8B8; font-size:0.9rem;">
        Ready to explore the data?
    </div>
    <a href="/Overview" target="_self">🚀&nbsp; Get Started — View Overview</a>
</div>
""")

# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.html("""
<div class="app-footer">
    Built by <strong>Aaditya A</strong> &nbsp;·&nbsp; MCA
    &nbsp;·&nbsp; PADV Activity 2 &nbsp;·&nbsp; 2026<br>
    Spotify Music Intelligence Dashboard &nbsp;|&nbsp;
    Jain (Deemed-to-be University) &nbsp;|&nbsp;
    Dataset: maharshipandya / Spotify Tracks via KaggleHub
</div>
""")
