"""
Fraud Analytics & Simulation Suite — Point d'entrée principal.
Lancement : uv run streamlit run app/main.py
"""
import streamlit as st
import sys
from pathlib import Path

# --- Path setup ---
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.styles import PREMIUM_CSS

st.set_page_config(
    page_title="Fraude POC — Open Banking",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 16px 0 24px 0;">
            <div style="font-size:2.2rem;">🔐</div>
            <div style="font-size:1.1rem; font-weight:700; color:#00D4A1;">Fraud-AI POC</div>
            <div style="font-size:0.75rem; color:#8892A4; margin-top:4px;">Open Banking · PIS Detection</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Mémoire Académique · IPSSI")
    st.caption("Dataset : BankSim (590k transactions)")
    st.divider()
    st.caption("Navigation via la liste de pages ci-dessus ↑")

# ---- Page d'accueil ----
st.markdown(
    """
    <div style="padding: 48px 0 24px 0; text-align:center;">
        <h1 style="font-size:2.8rem; font-weight:700; margin-bottom: 8px;">
            Système Expert <span style="color:#FF4B6E;">vs</span> Intelligence Artificielle
        </h1>
        <p style="font-size:1.1rem; color:#8892A4; max-width:640px; margin:0 auto;">
            Un prototype de détection de fraude comparant un moteur de règles bancaires traditionnel
            face à un classifieur <strong style="color:#00D4A1;">Random Forest</strong>
            dans le contexte de l'Initiation de Paiement (PIS).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(
        """
        <div class="kpi-card" style="padding:32px; text-align:left;">
            <div style="font-size:2rem; margin-bottom:12px;">📊</div>
            <div style="font-weight:600; font-size:1.05rem; margin-bottom:6px;">Dashboard Analytique</div>
            <div style="color:#8892A4; font-size:0.85rem;">
                Visualisez les performances globales des deux modèles sur l'ensemble du dataset BankSim.
                Matrices de confusion, métriques clés et importance des variables.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="kpi-card" style="padding:32px; text-align:left;">
            <div style="font-size:2rem; margin-bottom:12px;">💳</div>
            <div style="font-weight:600; font-size:1.05rem; margin-bottom:6px;">Simulateur Live PIS</div>
            <div style="color:#8892A4; font-size:0.85rem;">
                Initiez une transaction fictive et observez en temps réel le verdict des deux systèmes.
                Découvrez pourquoi l'IA réduit la friction client.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="kpi-card" style="padding:32px; text-align:left;">
            <div style="font-size:2rem; margin-bottom:12px;">💸</div>
            <div style="font-weight:600; font-size:1.05rem; margin-bottom:6px;">ROI Business</div>
            <div style="color:#8892A4; font-size:0.85rem;">
                Calculez l'impact économique réel de l'IA : réduction des alertes, gain client
                et économies opérationnelles mensuelles.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 Utilisez la navigation dans la **barre latérale gauche** pour explorer les sections.")
