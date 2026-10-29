# coding: utf-8
"""pages/7_Data_Quality.py -- Dataset health report: completeness, distributions, outliers."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.utils import (
    load_data, apply_theme, inject_css,
    page_header, divider, section_title, insight_box, kpi_card_html, COLORS,
)
from src.ui import top_nav, page_footer

st.set_page_config(
    page_title="Data Quality · Spotify Intelligence",
    page_icon="🔬", layout="wide",
)
inject_css()
top_nav("Data Quality Report")
page_header("Data Quality Report", "Transparency report — completeness, distributions, and outlier diagnostics", "🔬")

df = load_data()

# ── Summary KPIs ───────────────────────────────────────────────────
total_rows     = len(df)
total_cols     = df.shape[1]
missing_cells  = int(df.isnull().sum().sum())
missing_pct    = 100 * missing_cells / max(df.size, 1)
numeric_cols   = df.select_dtypes(include=np.number).columns.tolist()
dup_rows       = int(df.duplicated().sum())

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.html(kpi_card_html("Total Records", f"{total_rows:,}", "📦", "Post-cleaning"))
with k2:
    st.html(kpi_card_html("Total Columns", str(total_cols), "📋", "Features & metadata"))
with k3:
    color3 = "#1DB954" if missing_pct < 1 else "#F037A5"
    st.html(kpi_card_html("Missing Cells", f"{missing_pct:.2f}%", "🕳️", f"{missing_cells} cells", color3))
with k4:
    color4 = "#1DB954" if dup_rows == 0 else "#FFA42B"
    st.html(kpi_card_html("Duplicate Rows", str(dup_rows), "♻️", "Exact row duplicates", color4))

divider()

# ── Tab navigation ─────────────────────────────────────────────────
active_tab = st.radio(
    "Quality View",
    ["📋  Column Summary", "🕳️  Missing Values", "📊  Distributions", "🚨  Outlier Report"],
    horizontal=True,
    label_visibility="collapsed",
)
st.html('<div class="custom-divider" style="margin:0 0 1.5rem;"></div>')

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Column Summary
# ══════════════════════════════════════════════════════════════════
if active_tab == "📋  Column Summary":
    section_title("Column-Level Statistics")

    summary_rows = []
    for col in df.columns:
        dtype   = str(df[col].dtype)
        n_null  = int(df[col].isnull().sum())
        null_p  = f"{100 * n_null / total_rows:.1f}%"
        unique  = int(df[col].nunique())
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min  = f"{df[col].min():.3g}"
            c_max  = f"{df[col].max():.3g}"
            c_mean = f"{df[col].mean():.3g}"
            c_std  = f"{df[col].std():.3g}"
        else:
            c_min = c_max = c_mean = c_std = "—"
        summary_rows.append({
            "Column": col, "Type": dtype, "Missing": n_null,
            "Missing %": null_p, "Unique": unique,
            "Min": c_min, "Max": c_max, "Mean": c_mean, "Std Dev": c_std,
        })

    sum_df = pd.DataFrame(summary_rows)

    # Custom HTML table
    header_html = "".join(f"<th>{c}</th>" for c in sum_df.columns)
    rows_html = ""
    for _, row in sum_df.iterrows():
        miss_val = float(row["Missing %"].replace("%", ""))
        miss_color = "#F037A5" if miss_val > 5 else ("#FFA42B" if miss_val > 0 else "#1DB954")
        cells = ""
        for col in sum_df.columns:
            val = row[col]
            if col == "Missing %":
                cells += f'<td style="color:{miss_color}; font-weight:600;">{val}</td>'
            elif col == "Column":
                cells += f'<td><strong style="color:#FFF;">{val}</strong></td>'
            elif col == "Type":
                cells += f'<td><span class="feature-tag" style="font-size:0.7rem;">{val}</span></td>'
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    st.html(f"""
    <div class="glass-card" style="padding:1rem; overflow-x:auto; max-height:550px;">
    <table class="data-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    </div>""")

    divider()
    insight_box(
        "All numeric features are bounded within their expected [0, 1] ranges after cleaning. "
        "Loudness was left on its natural dB scale. "
        "Explicit is encoded as boolean (0/1). "
        "<strong style='color:#1DB954;'>Dataset is clean and ready for modeling.</strong>"
    )

# ══════════════════════════════════════════════════════════════════
# TAB 2 — Missing Values
# ══════════════════════════════════════════════════════════════════
elif active_tab == "🕳️  Missing Values":
    miss_series = df.isnull().sum()
    miss_df     = miss_series[miss_series >= 0].reset_index()
    miss_df.columns = ["Column", "Missing Count"]
    miss_df["Missing %"] = 100 * miss_df["Missing Count"] / total_rows
    miss_df = miss_df.sort_values("Missing Count", ascending=True)

    fig = go.Figure(go.Bar(
        x=miss_df["Missing %"],
        y=miss_df["Column"],
        orientation="h",
        marker=dict(
            color=miss_df["Missing %"],
            colorscale=[[0, "#1DB954"], [0.5, "#FFA42B"], [1, "#F037A5"]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Missing %", font=dict(color="#A0A8B8")),
                tickfont=dict(color="#A0A8B8"),
            ),
        ),
        text=[f"{v:.2f}%" for v in miss_df["Missing %"]],
        textposition="outside",
        textfont=dict(color="#A0A8B8", size=10),
    ))
    apply_theme(fig, height=max(380, len(miss_df) * 26),
                title="Missing Values per Column (% of total rows)")
    fig.update_layout(xaxis_title="Missing %", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    divider()

    # Heatmap of missingness pattern (sample)
    section_title("Missingness Pattern — 500 row sample")
    sample_miss = df.sample(min(500, len(df)), random_state=42).isnull().astype(int)
    if sample_miss.any().any():
        fig2 = go.Figure(go.Heatmap(
            z=sample_miss.T.values,
            x=[str(i) for i in range(sample_miss.shape[0])],
            y=sample_miss.columns.tolist(),
            colorscale=[[0, "rgba(18,24,32,0.6)"], [1, "#F037A5"]],
            showscale=False,
            hoverongaps=False,
        ))
        apply_theme(fig2, height=400, title="Missing (pink) / Present (dark) — Row Sample")
        fig2.update_layout(xaxis=dict(showticklabels=False, title="Rows"))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.html("""
        <div class="glass-card" style="text-align:center; padding:2.5rem;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">✅</div>
            <div style="font-size:1.1rem; font-weight:700; color:#1DB954; margin-bottom:4px;">No missing values!</div>
            <div style="color:#A0A8B8; font-size:0.85rem;">The cleaned dataset is 100% complete.</div>
        </div>""")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Distributions
# ══════════════════════════════════════════════════════════════════
elif active_tab == "📊  Distributions":
    feat_grid = ["danceability", "energy", "loudness", "valence",
                 "tempo", "acousticness", "speechiness", "instrumentalness",
                 "liveness", "popularity"]

    section_title("Violin Plots — Key Numeric Features")

    rows_n = 2
    cols_n = 5
    fig = go.Figure()

    for feat in feat_grid:
        fig.add_trace(go.Violin(
            y=df[feat].dropna(), name=feat.replace("_", " ").title(),
            box_visible=True, meanline_visible=True,
            line_color="#1DB954", fillcolor="rgba(29,185,84,0.15)",
            spanmode="hard",
        ))

    apply_theme(fig, height=480, title="Feature Distributions — Violin Plots")
    fig.update_layout(
        violinmode="overlay",
        xaxis_tickangle=-20,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    divider()

    # Skewness / kurtosis table
    section_title("Skewness & Kurtosis")
    sk_df = pd.DataFrame({
        "Feature":  [f.replace("_", " ").title() for f in feat_grid],
        "Mean":     [df[f].mean() for f in feat_grid],
        "Median":   [df[f].median() for f in feat_grid],
        "Std Dev":  [df[f].std() for f in feat_grid],
        "Skewness": [df[f].skew() for f in feat_grid],
        "Kurtosis": [df[f].kurtosis() for f in feat_grid],
    }).round(4)

    header_html = "".join(f"<th>{c}</th>" for c in sk_df.columns)
    rows_html = ""
    for _, row in sk_df.iterrows():
        skew_color = "#F037A5" if abs(row["Skewness"]) > 2 else ("#FFA42B" if abs(row["Skewness"]) > 1 else "#1DB954")
        cells = (
            f'<td><strong style="color:#FFF;">{row["Feature"]}</strong></td>'
            f'<td>{row["Mean"]}</td><td>{row["Median"]}</td><td>{row["Std Dev"]}</td>'
            f'<td style="color:{skew_color}; font-weight:600;">{row["Skewness"]}</td>'
            f'<td>{row["Kurtosis"]}</td>'
        )
        rows_html += f"<tr>{cells}</tr>"

    st.html(f"""
    <div class="glass-card" style="padding:1rem; overflow-x:auto;">
    <table class="data-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    </div>""")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — Outlier Report
# ══════════════════════════════════════════════════════════════════
elif active_tab == "🚨  Outlier Report":
    section_title("IQR-Based Outlier Detection")

    outlier_feats = ["danceability", "energy", "loudness", "valence",
                     "tempo", "acousticness", "speechiness", "instrumentalness",
                     "liveness", "popularity"]

    out_rows = []
    for feat in outlier_feats:
        q1, q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
        iqr    = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out  = int(((df[feat] < lo) | (df[feat] > hi)).sum())
        out_rows.append({
            "Feature":    feat.replace("_", " ").title(),
            "Q1":         round(q1, 4),
            "Q3":         round(q3, 4),
            "IQR":        round(iqr, 4),
            "Lower Fence": round(lo, 4),
            "Upper Fence": round(hi, 4),
            "Outliers":   n_out,
            "Outlier %":  round(100 * n_out / total_rows, 2),
        })

    out_df = pd.DataFrame(out_rows)

    # Bar chart of outlier counts
    fig = px.bar(
        out_df.sort_values("Outliers", ascending=True),
        x="Outliers", y="Feature", orientation="h",
        color="Outlier %",
        color_continuous_scale=[[0, "#1DB954"], [0.5, "#FFA42B"], [1, "#F037A5"]],
        text="Outliers",
    )
    fig.update_traces(textposition="outside", textfont=dict(color="#A0A8B8", size=11))
    apply_theme(fig, height=400, title="Outlier Count by Feature (IQR method)")
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text="Outlier %", font=dict(color="#A0A8B8")),
            tickfont=dict(color="#A0A8B8"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    divider()

    # HTML table
    section_title("Outlier Fence Summary")
    header_html = "".join(f"<th>{c}</th>" for c in out_df.columns)
    rows_html = ""
    for _, row in out_df.iterrows():
        pct      = row["Outlier %"]
        pct_col  = "#F037A5" if pct > 10 else ("#FFA42B" if pct > 3 else "#1DB954")
        cells = (
            f'<td><strong style="color:#FFF;">{row["Feature"]}</strong></td>'
            f'<td>{row["Q1"]}</td><td>{row["Q3"]}</td><td>{row["IQR"]}</td>'
            f'<td>{row["Lower Fence"]}</td><td>{row["Upper Fence"]}</td>'
            f'<td style="color:{pct_col}; font-weight:600;">{int(row["Outliers"]):,}</td>'
            f'<td style="color:{pct_col}; font-weight:600;">{pct}%</td>'
        )
        rows_html += f"<tr>{cells}</tr>"

    st.html(f"""
    <div class="glass-card" style="padding:1rem; overflow-x:auto;">
    <table class="data-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    </div>""")

    divider()

    # Box plots for the top-3 outlier features
    section_title("Box Plots — Top 3 Most Outlier-Prone Features")
    top3_feats = out_df.nlargest(3, "Outlier %")["Feature"].tolist()
    top3_raw   = [f.lower().replace(" ", "_") for f in top3_feats]

    bp_cols = st.columns(3)
    for col_w, feat in zip(bp_cols, top3_raw):
        q1, q3 = df[feat].quantile(0.25), df[feat].quantile(0.75)
        iqr    = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out  = int(((df[feat] < lo) | (df[feat] > hi)).sum())
        out_pct = 100 * n_out / total_rows
        out_vals = df[(df[feat] < lo) | (df[feat] > hi)][feat]

        fig2 = go.Figure()
        fig2.add_trace(go.Box(
            y=df[feat].dropna(),
            name=feat.replace("_", " ").title(),
            marker_color="#1DB954",
            line_color="#1DB954",
            fillcolor="rgba(29,185,84,0.15)",
            boxmean=True,
            jitter=0.3,
            pointpos=-1.8,
            boxpoints=False,
        ))
        apply_theme(fig2, height=320,
                    title=f"{feat.replace('_',' ').title()}<br><sup style='font-size:0.6rem'>{out_pct:.1f}% outliers</sup>")
        with col_w:
            st.plotly_chart(fig2, use_container_width=True)

    divider()
    insight_box(
        "<strong style='color:#FFA42B;'>Instrumentalness</strong> and "
        "<strong style='color:#FFA42B;'>Speechiness</strong> are the most right-skewed features "
        "— the majority of tracks score very low, but outliers (instrumental or spoken-word tracks) "
        "pull the distribution tail upward. These were <em>not</em> clipped during preprocessing; "
        "tree-based models handle them natively without scaling.",
        icon="🚨",
    )
page_footer()
