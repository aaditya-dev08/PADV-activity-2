# coding: utf-8
"""src/ui.py -- Shared top-nav and footer helpers for every page."""
import streamlit as st


def top_nav(page_name: str) -> None:
    """Render the sticky top navigation bar."""
    st.html(f"""
    <div class="top-nav">
        <div class="top-nav-logo">🎵 Music Intel</div>
        <div style="color:rgba(255,255,255,0.55); font-size:13px; font-weight:500;
                    letter-spacing:0.04em;">{page_name}</div>
    </div>
    """)


def page_footer() -> None:
    """Render the standard page footer."""
    st.html("""
    <div class="app-footer">
        Built by <strong>Aaditya A</strong> &nbsp;·&nbsp; MCA
        &nbsp;·&nbsp; PADV Activity 2 &nbsp;·&nbsp; 2026<br>
        Spotify Music Intelligence Dashboard &nbsp;|&nbsp;
        Jain (Deemed-to-be University)
    </div>
    """)
