"""
Page 2 — Simulateur Live PIS (Vue Micro)
Formulaire interactif de transaction avec verdict dual-modèle.
"""
import sys
from pathlib import Path
import time

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import shap

from app.styles import PREMIUM_CSS, verdict_card
from app.utils import load_model, load_scaler, load_label_encoders
from src.models.expert_system import predict_legacy
from src.schemas.fraud_detection import TransactionInput, RiskScoreOutput

st.set_page_config(page_title="Simulateur · Fraud POC", page_icon="💳", layout="wide")
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
rf_model = load_model()
scaler = load_scaler()
label_encoders = load_label_encoders()

# ── Mock API Function ────────────────────────────────────────────────────────
def predict_realtime(tx: TransactionInput) -> RiskScoreOutput:
    """Mock API endpoint processing a transaction and returning SHAP explainability."""
    row_df = pd.DataFrame([tx.model_dump()])
    
    # Preprocess
    for col_name in ["age", "gender", "category"]:
        le = label_encoders[col_name]
        val = row_df[col_name].iloc[0]
        row_df[col_name] = le.transform([val])[0] if val in le.classes_ else -1
        
    row_df["amount"] = scaler.transform(row_df[["amount"]])
    
    # Predict
    rf_prob = float(rf_model.predict_proba(row_df)[0, 1])
    rf_result = rf_prob >= 0.5
    
    # SHAP Explainability
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(row_df)
    
    # Handle SHAP output format depending on sklearn/shap version
    if isinstance(shap_values, list):
        shap_vals_1 = shap_values[1][0]
    else:
        if len(shap_values.shape) == 3:
            shap_vals_1 = shap_values[0, :, 1]
        else:
            shap_vals_1 = shap_values[0]
            
    explanation = {feat: float(val) for feat, val in zip(row_df.columns, shap_vals_1)}
    
    return RiskScoreOutput(
        risk_score=rf_prob,
        is_fraud=rf_result,
        explanation=explanation
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header"><h2 style="margin:0">💳 Simulateur de Paiement Live (PIS)</h2></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Initiez une transaction fictive et observez en temps réel la décision des deux systèmes de détection."
)
st.markdown("<br>", unsafe_allow_html=True)

# ── Transaction form ───────────────────────────────────────────────────────────
CATEGORIES = [
    "es_food", "es_health", "es_home", "es_leisure",
    "es_otherservices", "es_sportsandtoys", "es_tech",
    "es_travel", "es_transportation", "es_wellnessandbeauty",
]
AGES = ["1", "2", "3", "4", "5", "6", "U"]  # BankSim age buckets
GENDERS = ["M", "F", "E", "U"]

with st.form("transaction_form"):
    st.markdown("### 📋 Détails de la Transaction")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        amount = st.number_input(
            "💶 Montant (€)", min_value=0.01, max_value=10_000.0, value=450.0, step=10.0
        )
        category = st.selectbox("🛍️ Catégorie de dépense", options=CATEGORIES, index=3)
        step = st.number_input("🕐 Step (heure simulée)", min_value=1, max_value=743, value=150)
    with col2:
        gender = st.selectbox("👤 Genre du titulaire", options=GENDERS, index=0)
        age = st.selectbox("📅 Tranche d'âge (bucket BankSim)", options=AGES, index=2)
        st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🔍 Analyser la Transaction",
        use_container_width=True,
        type="primary",
    )

# ── Analysis & Verdict ────────────────────────────────────────────────────────
if submitted:
    # Simulated loading effect
    with st.spinner("🤖 IA en cours d'analyse..."):
        time.sleep(0.6)

    # ---- Système Expert (rule-based) ----
    row_raw = pd.Series({"amount": amount, "category": category})
    legacy_result = predict_legacy(row_raw)
    legacy_reason = (
        "Montant > 1 000€" if amount > 1000
        else f"Catégorie à risque ({category}) + Montant > 500€" if amount > 500 and category in ["es_leisure", "es_travel"]
        else "Transaction dans les seuils acceptables"
    )

    from typing import cast, Literal
    gender_literal = cast(Literal["M", "F", "E", "U"], gender)
    tx_input = TransactionInput(
        step=step,
        age=age,
        gender=gender_literal,
        category=category,
        amount=amount
    )
    
    response = predict_realtime(tx_input)
    rf_prob = response.risk_score
    rf_result = 1 if response.is_fraud else 0
    rf_reason = f"Score de risque IA : {rf_prob * 100:.1f}%"

    # ─ Display cards ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚖️ Verdict des Systèmes")

    v_col1, v_col2 = st.columns(2, gap="large")
    with v_col1:
        st.markdown(
            verdict_card(bool(legacy_result), "Système Expert (Règles Métier)", legacy_reason),
            unsafe_allow_html=True,
        )
    with v_col2:
        st.markdown(
            verdict_card(bool(rf_result), "IA — Random Forest", rf_reason),
            unsafe_allow_html=True,
        )

    # ─ Risk gauge ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📡 Score de Risque IA (Probabilité de Fraude)")

    gauge_color = "#FF4B6E" if rf_prob > 0.7 else "#FFB347" if rf_prob > 0.4 else "#00D4A1"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rf_prob * 100,
        delta={"reference": 50, "valueformat": ".1f"},
        number={"suffix": "%", "font": {"color": gauge_color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8892A4"},
            "bar": {"color": gauge_color, "thickness": 0.3},
            "bgcolor": "#1A1F2E",
            "steps": [
                {"range": [0, 40], "color": "rgba(0,212,161,0.1)"},
                {"range": [40, 70], "color": "rgba(255,179,71,0.1)"},
                {"range": [70, 100], "color": "rgba(255,75,110,0.1)"},
            ],
            "threshold": {
                "line": {"color": "#FAFAFA", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
        domain={"x": [0.1, 0.9], "y": [0, 1]},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA", size=14),
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ─ SHAP Explainability ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧠 Explicabilité du modèle (Valeurs SHAP)")
    st.caption("Cette analyse permet de comprendre instantanément ce qui a poussé l'IA vers la fraude ou la légitimité.")
    
    shap_dict = response.explanation
    shap_df = pd.DataFrame(list(shap_dict.items()), columns=["Variable", "Impact"])
    shap_df = shap_df.sort_values(by="Impact", ascending=True)
    
    fig_shap = go.Figure(go.Bar(
        x=shap_df["Impact"],
        y=shap_df["Variable"],
        orientation='h',
        marker_color=["#FF4B6E" if v > 0 else "#00D4A1" for v in shap_df["Impact"]]
    ))
    fig_shap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        title="Contribution des variables au score de risque (Rouge = Augmente, Vert = Baisse)",
        xaxis=dict(title="Impact SHAP", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=20, l=20, r=20),
        height=300,
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # ─ Narrative ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if legacy_result == 1 and rf_result == 0:
        st.success(
            "✅ **Cas typique de Faux Positif réduit !** Le système Expert aurait bloqué ce client légitime, "
            "créant de la friction. L'IA le laisse passer en toute confiance."
        )
    elif legacy_result == 0 and rf_result == 1:
        st.warning(
            "⚠️ **L'IA détecte une fraude que les règles manquent.** Ce cas illustre la capacité "
            "de l'apprentissage automatique à identifier des patterns non-codifiables manuellement."
        )
    elif legacy_result == 1 and rf_result == 1:
        st.error("🔴 **Les deux systèmes s'accordent : transaction à haut risque bloquée.**")
    else:
        st.info("🟢 **Transaction validée par les deux systèmes.** Aucun signal d'alerte.")
