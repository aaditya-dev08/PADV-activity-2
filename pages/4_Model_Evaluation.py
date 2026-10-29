# coding: utf-8
"""pages/4_Model_Evaluation.py -- ML model comparison, ROC, confusion matrix, feature importance."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.utils import (
    load_data, load_model, apply_theme, inject_css,
    page_header, divider, section_title, insight_box, COLORS, FEATURES,
)
from src.ui import top_nav, page_footer

st.set_page_config(
    page_title="Model Evaluation · Spotify Intelligence",
    page_icon="🤖", layout="wide",
)
inject_css()
top_nav("Model Evaluation")
page_header("Model Evaluation", "Comparing classifiers on 17,949 held-out test tracks", "🤖")

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP_PATH = os.path.join(ROOT, "outputs", "model_comparison.csv")
FEAT_PATH = os.path.join(ROOT, "outputs", "feature_importance.csv")

# ── Load artifacts ─────────────────────────────────────────────────
comp_df  = pd.read_csv(COMP_PATH)
feat_df  = pd.read_csv(FEAT_PATH)
model, scaler = load_model()
df = load_data()

# Re-create test split with same seed
X = df[FEATURES]; y = df["is_hit"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_test_sc = scaler.transform(X_test)
y_pred    = model.predict(X_test_sc)
y_proba   = model.predict_proba(X_test_sc)[:, 1]

best_name = comp_df.loc[comp_df["F1"].idxmax(), "Model"]
best_row  = comp_df.loc[comp_df["F1"].idxmax()]

# ── Model leaderboard ──────────────────────────────────────────────
section_title("Model Leaderboard")
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC_ROC"]

# Custom HTML table
rows_html = ""
for _, row in comp_df.iterrows():
    is_best = row["Model"] == best_name
    bg = 'style="background:rgba(29,185,84,0.08);"' if is_best else ""
    crown = " 🏆" if is_best else ""
    rows_html += f"<tr {bg}><td><strong>{row['Model']}{crown}</strong></td>"
    for m in metrics:
        color = "#1DB954" if is_best else "#FFFFFF"
        weight = "800" if is_best else "400"
        rows_html += f'<td style="color:{color};font-weight:{weight};">{row[m]:.4f}</td>'
    rows_html += "</tr>"

st.html(f"""
<div class="glass-card" style="padding:1.2rem; overflow-x:auto;">
<table class="data-table">
<thead><tr>
  <th>Model</th>{''.join(f"<th>{m}</th>" for m in metrics)}
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>""")

divider()

# ── Grouped metric comparison chart ────────────────────────────────
section_title("Performance Comparison")
fig_bar = go.Figure()
for i, (_, row) in enumerate(comp_df.iterrows()):
    fig_bar.add_trace(go.Bar(
        name=row["Model"], x=metrics,
        y=[row[m] for m in metrics],
        marker_color=COLORS[i],
        text=[f"{row[m]:.3f}" for m in metrics],
        textposition="outside",
        textfont=dict(size=10, color="#A0A8B8"),
    ))
apply_theme(fig_bar, height=400, title="All Models — 5 Evaluation Metrics")
fig_bar.update_layout(barmode="group", yaxis_range=[0, 1.12],
                      xaxis_title="Metric", yaxis_title="Score")
st.plotly_chart(fig_bar, use_container_width=True)

# ── Best model callout ─────────────────────────────────────────────
st.html(f"""
<div class="glass-card" style="border:1px solid #1DB954;
     background:linear-gradient(135deg,rgba(29,185,84,0.1),rgba(18,24,32,0.85));">
  <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
    <div style="font-size:2.5rem;">🏆</div>
    <div>
      <div style="font-family:'Poppins',sans-serif; font-size:1.25rem; font-weight:700; color:#FFF;">
        Best Model: {best_name}
      </div>
      <div style="color:#A0A8B8; font-size:0.82rem; margin-top:4px;">
        Chosen by highest F1-score on the stratified test set
      </div>
    </div>
    <div style="margin-left:auto; display:flex; gap:8px; flex-wrap:wrap;">
      <span class="badge">F1 = {best_row['F1']:.4f}</span>
      <span class="badge">Recall = {best_row['Recall']:.4f}</span>
      <span class="badge">AUC = {best_row['AUC_ROC']:.4f}</span>
      <span class="badge">Acc = {best_row['Accuracy']:.4f}</span>
    </div>
  </div>
</div>""")

divider()

# ── Confusion matrix + ROC ─────────────────────────────────────────
section_title("Confusion Matrix & ROC Curve")
cm_col, roc_col = st.columns(2)

with cm_col:
    cm = confusion_matrix(y_test, y_pred)
    z  = cm[::-1]
    fig_cm = go.Figure(go.Heatmap(
        z=z,
        x=["Predicted Not Hit", "Predicted Hit"],
        y=["Actual Hit", "Actual Not Hit"],
        colorscale=[[0, "#0A3D1F"], [0.5, "#1DB954"], [1, "#1ED760"]],
        text=[[f"{v:,}" for v in row] for row in z],
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=18, color="#FFFFFF"),
        showscale=False,
    ))
    apply_theme(fig_cm, height=360, title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

with roc_col:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"AUC = {roc_auc:.3f}",
        line=dict(color="#1DB954", width=3),
        fill="tozeroy", fillcolor="rgba(29,185,84,0.1)",
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="#F037A5", width=1.5, dash="dash"),
    ))
    apply_theme(fig_roc, height=360, title=f"ROC Curve (AUC = {roc_auc:.3f})")
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    st.plotly_chart(fig_roc, use_container_width=True)

divider()

# ── Feature importance ─────────────────────────────────────────────
section_title("Feature Importance — Best Model")
feat_sorted = feat_df.sort_values("Importance", ascending=True)
fig_feat = go.Figure(go.Bar(
    x=feat_sorted["Importance"],
    y=feat_sorted["Feature"],
    orientation="h",
    marker=dict(
        color=feat_sorted["Importance"],
        colorscale=[[0, "#0E5C2C"], [0.5, "#1DB954"], [1, "#1ED760"]],
        showscale=False,
    ),
    text=[f"{v:.4f}" for v in feat_sorted["Importance"]],
    textposition="outside",
    textfont=dict(color="#A0A8B8", size=11),
))
apply_theme(fig_feat, height=380, title=f"Feature Importance ({best_name})")
st.plotly_chart(fig_feat, use_container_width=True)

# ── Precision / recall insight ─────────────────────────────────────
divider()
insight_box(
    "<strong style='color:#1DB954;'>Honest Assessment:</strong> "
    "Our best model (Random Forest) achieves F1=0.42 and AUC=0.675 — well above random (0.5 AUC), "
    "but far from perfect. This reflects a fundamental truth: "
    "<strong>audio features alone explain ~10–15% of popularity variance</strong>. "
    "The rest is driven by artist brand recognition, playlist curators, label marketing, "
    "social virality, and release timing — factors invisible in Spotify's audio metadata. "
    "Use this as a signal, not a crystal ball.",
    icon="⚠️",
)
page_footer()
