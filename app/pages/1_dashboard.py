"""
Page 1 — Dashboard Analytique (Vue Macro)
Affiche les performances comparatives sur le dataset de test.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

from app.styles import PREMIUM_CSS, kpi_card
from app.utils import load_model, load_test_data
from src.models.expert_system import predict_expert_system_on_df

st.set_page_config(page_title="Dashboard · Fraud POC", page_icon="📊", layout="wide")
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><h2 style="margin:0">📊 Dashboard Analytique</h2></div>',
    unsafe_allow_html=True,
)
st.caption("Performance globale sur le jeu de test BankSim (30% du dataset)")

# ── Load data ────────────────────────────────────────────────────────────────
test_bundle = load_test_data()
if test_bundle is None:
    st.warning("⚠️ Données de test introuvables. Lancez d'abord le script d'entraînement.")
    st.code("uv run python scripts/train_and_export.py --data notebook/bs140513_032310.csv")
    st.stop()

rf_model = load_model()

X_test_raw: pd.DataFrame = test_bundle["X_test_raw"]
X_test: pd.DataFrame = test_bundle["X_test"]
y_test: pd.Series = test_bundle["y_test"]

# ── Predictions ───────────────────────────────────────────────────────────────
y_pred_legacy = predict_expert_system_on_df(X_test_raw)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# ── KPI Cards ────────────────────────────────────────────────────────────────
rep_a = classification_report(y_test, y_pred_legacy, output_dict=True)
rep_b = classification_report(y_test, y_pred_rf, output_dict=True)

fp_legacy = int(confusion_matrix(y_test, y_pred_legacy)[0, 1])
fp_rf = int(confusion_matrix(y_test, y_pred_rf)[0, 1])
fp_reduction = round((1 - fp_rf / max(fp_legacy, 1)) * 100, 1)

recall_legacy = round(rep_a["1"]["recall"] * 100, 1)
recall_rf = round(rep_b["1"]["recall"] * 100, 1)

st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        kpi_card("Faux Positifs — Legacy", f"{fp_legacy:,}", "Clients légitimes bloqués", "kpi-red"),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        kpi_card("Faux Positifs — IA", f"{fp_rf:,}", f"− {fp_reduction}% vs Legacy", "kpi-green"),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        kpi_card("Rappel Fraudes — Legacy", f"{recall_legacy}%", "Fraudes détectées", "kpi-amber"),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        kpi_card("Rappel Fraudes — IA", f"{recall_rf}%", "Fraudes détectées", "kpi-green"),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Confusion Matrices ───────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><h3 style="margin:0">Matrices de Confusion</h3></div>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns(2, gap="large")

def plot_confusion(cm: np.ndarray, title: str, colorscale: str) -> go.Figure:
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Prédit Légitime", "Prédit Fraude"],
        y=["Réel Légitime", "Réel Fraude"],
        annotation_text=[[f"{v:,}" for v in row] for row in cm],
        colorscale=colorscale,
        showscale=False,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        margin=dict(t=60, b=20, l=20, r=20),
        height=300,
    )
    return fig

with col_a:
    cm_a = confusion_matrix(y_test, y_pred_legacy)
    st.plotly_chart(
        plot_confusion(cm_a, "Modèle A — Système Expert (Legacy)", "Reds"),
        use_container_width=True,
    )

with col_b:
    cm_b = confusion_matrix(y_test, y_pred_rf)
    st.plotly_chart(
        plot_confusion(cm_b, "Modèle B — Random Forest (IA)", "Greens"),
        use_container_width=True,
    )

st.divider()

# ── Precision-Recall Curves ───────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><h3 style="margin:0">Courbe Précision-Rappel</h3></div>',
    unsafe_allow_html=True,
)
st.caption("Préférable à la courbe ROC pour les données déséquilibrées (cf. specs.md)")

prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)
ap_rf = average_precision_score(y_test, y_prob_rf)

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(
    x=rec_rf, y=prec_rf,
    mode="lines", name=f"Random Forest (AP={ap_rf:.3f})",
    line=dict(color="#00D4A1", width=2.5),
    fill="tozeroy", fillcolor="rgba(0,212,161,0.07)",
))
# Baseline (legacy)
prec_leg = float(rep_a["1"]["precision"])
rec_leg = float(rep_a["1"]["recall"])
fig_pr.add_trace(go.Scatter(
    x=[rec_leg], y=[prec_leg],
    mode="markers", name="Système Expert (point fixe)",
    marker=dict(color="#FF4B6E", size=12, symbol="x"),
))
fig_pr.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA"),
    xaxis=dict(title="Rappel", gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(title="Précision", gridcolor="rgba(255,255,255,0.05)"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=20, b=20, l=20, r=20),
    height=350,
)
st.plotly_chart(fig_pr, use_container_width=True)

st.divider()

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><h3 style="margin:0">Importance des Variables</h3></div>',
    unsafe_allow_html=True,
)

features = ["step", "age", "gender", "category", "amount"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values()

fig_fi = go.Figure(go.Bar(
    x=importances.values,
    y=importances.index,
    orientation="h",
    marker=dict(
        color=importances.values,
        colorscale=[[0, "#1A1F2E"], [1, "#00D4A1"]],
        showscale=False,
    ),
))
fig_fi.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA"),
    xaxis=dict(title="Importance (Gini)", gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    margin=dict(t=10, b=20, l=20, r=20),
    height=300,
)
st.plotly_chart(fig_fi, use_container_width=True)
