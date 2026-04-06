"""
Page 3 — ROI Business & Impact Opérationnel
Calculateur interactif des économies générées par l'IA vs le système legacy.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from app.styles import PREMIUM_CSS, kpi_card
from app.utils import load_model, load_test_data
from src.models.expert_system import predict_expert_system_on_df

st.set_page_config(page_title="ROI Business · Fraud POC", page_icon="💸", layout="wide")
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
test_bundle = load_test_data()
if test_bundle is None:
    st.warning("⚠️ Données de test introuvables. Lancez d'abord le script d'entraînement.")
    st.code("uv run python scripts/train_and_export.py --data notebook/bs140513_032310.csv")
    st.stop()

rf_model = load_model()
X_test_raw: pd.DataFrame = test_bundle["X_test_raw"]
X_test: pd.DataFrame = test_bundle["X_test"]
y_test: pd.Series = test_bundle["y_test"]

y_pred_legacy = predict_expert_system_on_df(X_test_raw)
y_pred_rf = rf_model.predict(X_test)

cm_legacy = confusion_matrix(y_test, y_pred_legacy)
cm_rf = confusion_matrix(y_test, y_pred_rf)

tn_l, fp_l, fn_l, tp_l = cm_legacy.ravel()
tn_r, fp_r, fn_r, tp_r = cm_rf.ravel()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><h2 style="margin:0">💸 ROI Business & Impact Opérationnel</h2></div>',
    unsafe_allow_html=True,
)
st.caption("Traduisez les métriques techniques en gains financiers concrets pour votre organisation.")
st.markdown("<br>", unsafe_allow_html=True)

# ── Sliders ───────────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Paramètres Économiques")
st.caption("Ajustez les coûts pour votre contexte métier afin de voir l'impact calculé automatiquement.")

c1, c2 = st.columns(2, gap="large")
with c1:
    cost_fp = st.slider(
        "💳 Coût d'un Faux Positif [€]",
        min_value=1, max_value=500, value=25, step=5,
        help="Coût estimé d'un client légitime bloqué : abandon du paiement, appel au support, attrition potentielle.",
    )
    transactions_per_month = st.slider(
        "📦 Volume transactions / mois",
        min_value=10_000, max_value=1_000_000, value=100_000, step=10_000,
        format="%d",
    )
with c2:
    cost_fn = st.slider(
        "🔒 Coût d'une Fraude manquée [€]",
        min_value=50, max_value=5_000, value=350, step=50,
        help="Montant moyen d'une fraude non détectée : remboursement client, coût de traitement litige.",
    )
    months = st.slider("📅 Horizon de projection (mois)", min_value=1, max_value=24, value=12)

# ── Compute ROI ───────────────────────────────────────────────────────────────
# Scale confusion matrix from test-set to monthly volume
test_size = len(y_test)
scale = transactions_per_month / test_size

fp_legacy_month = int(fp_l * scale)
fp_rf_month = int(fp_r * scale)
fn_legacy_month = int(fn_l * scale)
fn_rf_month = int(fn_r * scale)

total_cost_legacy = (fp_legacy_month * cost_fp) + (fn_legacy_month * cost_fn)
total_cost_rf = (fp_rf_month * cost_fp) + (fn_rf_month * cost_fn)
monthly_saving = total_cost_legacy - total_cost_rf
annual_saving = monthly_saving * months

savings_timeline = [monthly_saving * m for m in range(1, months + 1)]
savings_labels = [f"M{m}" for m in range(1, months + 1)]

# ── KPI Cards ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📈 Résultats Chiffrés")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        kpi_card("Coût Mensuel — Legacy", f"{total_cost_legacy:,.0f} €", "Alertes + Fraudes", "kpi-red"),
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        kpi_card("Coût Mensuel — IA", f"{total_cost_rf:,.0f} €", "Alertes + Fraudes", "kpi-green"),
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        kpi_card("Économie Mensuelle", f"{monthly_saving:,.0f} €", "par mois", "kpi-blue"),
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        kpi_card(f"Économie sur {months} mois", f"{annual_saving:,.0f} €", "Projection cumulative", "kpi-green"),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Savings Timeline Chart ────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><h3 style="margin:0">Économies Cumulatives dans le Temps</h3></div>',
    unsafe_allow_html=True,
)

fig_tl = go.Figure()
fig_tl.add_trace(go.Bar(
    x=savings_labels, y=savings_timeline,
    marker=dict(
        color=savings_timeline,
        colorscale=[[0, "#1A2E2B"], [1, "#00D4A1"]],
        showscale=False,
    ),
    text=[f"{v:,.0f} €" for v in savings_timeline],
    textposition="outside",
    textfont=dict(color="#8892A4", size=10),
))
fig_tl.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA"),
    xaxis=dict(title="Mois", gridcolor="rgba(255,255,255,0.04)"),
    yaxis=dict(title="Économies cumulées (€)", gridcolor="rgba(255,255,255,0.04)"),
    margin=dict(t=30, b=20, l=20, r=20),
    height=360,
)
st.plotly_chart(fig_tl, use_container_width=True)

st.divider()

# ── FP/FN Breakdown Chart ─────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><h3 style="margin:0">Décomposition des Coûts par Source</h3></div>',
    unsafe_allow_html=True,
)
st.caption("FP = Faux Positifs (clients légitimes bloqués) · FN = Faux Négatifs (fraudes manquées)")

fig_decomp = go.Figure()
fig_decomp.add_trace(go.Bar(
    name="Coût FP (Friction UX)", x=["Système Expert", "Random Forest"],
    y=[fp_legacy_month * cost_fp, fp_rf_month * cost_fp],
    marker_color="#FF4B6E", opacity=0.85,
))
fig_decomp.add_trace(go.Bar(
    name="Coût FN (Fraudes manquées)", x=["Système Expert", "Random Forest"],
    y=[fn_legacy_month * cost_fn, fn_rf_month * cost_fn],
    marker_color="#FFB347", opacity=0.85,
))
fig_decomp.update_layout(
    barmode="stack",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#FAFAFA"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    yaxis=dict(title="Coût mensuel (€)", gridcolor="rgba(255,255,255,0.05)"),
    xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    margin=dict(t=20, b=20, l=20, r=20),
    height=340,
)
st.plotly_chart(fig_decomp, use_container_width=True)

# ── Business narrative ────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📝 Enseignements Métier")
col_l, col_r = st.columns(2, gap="large")
with col_l:
    fp_saved = fp_legacy_month - fp_rf_month
    st.metric(
        "Clients légitimes non-bloqués / mois",
        f"{fp_saved:,}",
        delta=f"−{fp_saved:,} frictions UX évitées",
    )
    st.caption(
        "Chaque Faux Positif évité = un client qui finalise son paiement sans contacter le support. "
        "Réduction directe du taux d'abandon et de l'attrition (churn)."
    )
with col_r:
    detection_gain = (tp_r / max(tp_r + fn_r, 1)) - (tp_l / max(tp_l + fn_l, 1))
    st.metric(
        "Gain en détection de fraudes réelles",
        f"+{detection_gain * 100:.1f}%",
        delta="Recall IA vs Legacy",
    )
    st.caption(
        "L'IA capture plus de fraudes réelles que le système expert, tout en générant moins de fausses alertes. "
        "C'est l'équilibre Précision-Rappel recherché par les équipes de conformité Open Banking."
    )
