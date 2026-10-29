# coding: utf-8
"""pages/2_Track_Explorer.py -- Interactive track search & filter."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import streamlit as st
import pandas as pd

from src.utils import (
    load_data, inject_css, page_header, divider,
    section_title, insight_box, track_card_html, esc,
)
from src.ui import top_nav, page_footer

st.set_page_config(
    page_title="Track Explorer · Spotify Intelligence",
    page_icon="🔍", layout="wide",
)
inject_css()
top_nav("Track Explorer")
page_header("Track Explorer", "Browse, filter, and search 89,741 Spotify tracks", "🔍")

df = load_data()
all_genres = sorted(df["track_genre"].dropna().unique().tolist())

# ── Filter panel ────────────────────────────────────────────────────
st.html('<div class="filter-panel">')
section_title("Filters")

fc1, fc2, fc3 = st.columns([2, 1.2, 1])
with fc1:
    genres_sel = st.multiselect("Genre", all_genres, placeholder="All genres")
with fc2:
    hit_filter = st.radio("Category", ["All", "Hit Only", "Not Hit"], horizontal=True)
with fc3:
    sort_by = st.selectbox("Sort By", ["popularity", "danceability", "energy", "valence", "tempo"])

fs1, fs2, fs3 = st.columns(3)
with fs1:
    pop_range = st.slider("Popularity Range", 0, 100, (0, 100))
with fs2:
    energy_range = st.slider("Energy Range", 0.0, 1.0, (0.0, 1.0), 0.01)
with fs3:
    dance_range = st.slider("Danceability Range", 0.0, 1.0, (0.0, 1.0), 0.01)

st.html("</div>")

# ── Apply filters ───────────────────────────────────────────────────
fdf = df.copy()
if genres_sel:
    fdf = fdf[fdf["track_genre"].isin(genres_sel)]
fdf = fdf[
    (fdf["popularity"]    >= pop_range[0])    & (fdf["popularity"]    <= pop_range[1])    &
    (fdf["energy"]        >= energy_range[0]) & (fdf["energy"]        <= energy_range[1]) &
    (fdf["danceability"]  >= dance_range[0])  & (fdf["danceability"]  <= dance_range[1])
]
if hit_filter == "Hit Only":
    fdf = fdf[fdf["is_hit"] == 1]
elif hit_filter == "Not Hit":
    fdf = fdf[fdf["is_hit"] == 0]
fdf = fdf.sort_values(sort_by, ascending=False).reset_index(drop=True)

# ── Match count banner ──────────────────────────────────────────────
hit_c = int(fdf["is_hit"].sum())
hit_pct = 100 * hit_c / max(len(fdf), 1)
st.html(f"""
<div style="display:flex; align-items:center; gap:16px; margin:1rem 0 0.4rem;">
    <div style="font-size:1.8rem; font-weight:800; color:#1DB954;">{len(fdf):,}</div>
    <div style="color:#A0A8B8; font-size:0.9rem;">tracks match your filters</div>
    <span class="badge">{hit_c:,} hits ({hit_pct:.1f}%)</span>
</div>
""")

divider()

# ── Top 5 cards ─────────────────────────────────────────────────────
section_title("Top Matches")
top5 = fdf.head(5)
if len(top5) == 0:
    st.html("""
    <div class="glass-card" style="text-align:center; padding:2.5rem;">
        <div style="font-size:2rem; margin-bottom:0.5rem;">🔍</div>
        <div style="color:#A0A8B8;">No tracks match your current filters.</div>
    </div>""")
else:
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        st.html(
            track_card_html(rank, row["track_name"], row["artists"],
                            row["track_genre"], int(row["popularity"]), int(row["is_hit"]))
        )

divider()

# ── Full table ──────────────────────────────────────────────────────
section_title("Full Results Table")

show_cols = ["track_name", "artists", "track_genre", "popularity",
             "energy", "danceability", "valence", "tempo", "is_hit"]
show_df = fdf[show_cols].head(500).copy()
show_df["is_hit"] = show_df["is_hit"].map({1: "✓ Hit", 0: "✗ Not Hit"})
show_df.columns = ["Track", "Artists", "Genre", "Popularity",
                   "Energy", "Danceability", "Valence", "Tempo", "Status"]

st.dataframe(show_df, use_container_width=True, hide_index=True, height=420)

# ── Download ────────────────────────────────────────────────────────
st.markdown("<br>")
dl_col, _ = st.columns([1, 3])
with dl_col:
    buf = io.StringIO()
    fdf.to_csv(buf, index=False)
    st.download_button(
        label=f"Download {len(fdf):,} tracks as CSV",
        data=buf.getvalue(),
        file_name="spotify_filtered.csv",
        mime="text/csv",
    )

divider()

# ── Insight ─────────────────────────────────────────────────────────
insight_box(
    "Use the sliders above to find tracks with specific audio profiles. "
    "Try <strong style='color:#1DB954;'>High Energy (0.8+) + High Danceability (0.7+)</strong> "
    "to surface club-ready potential hits, or "
    "<strong style='color:#FFA42B;'>Low Acousticness + Low Speechiness</strong> "
    "to isolate pure electronic tracks."
)
page_footer()
