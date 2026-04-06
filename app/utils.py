"""
Utilitaires partagés : chargement des artefacts ML.
"""
import sys
from pathlib import Path

# Make src/ importable when running from app/ directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import streamlit as st

MODEL_DIR = ROOT / "models"


@st.cache_resource
def load_model():
    """Load the trained Random Forest model (cached)."""
    path = MODEL_DIR / "rf_model.joblib"
    if not path.exists():
        st.error("❌ Modèle introuvable. Exécutez d'abord : `uv run python scripts/train_and_export.py --data <csv>`")
        st.stop()
    return joblib.load(path)


@st.cache_resource
def load_scaler():
    """Load the StandardScaler (cached)."""
    return joblib.load(MODEL_DIR / "scaler.joblib")


@st.cache_resource
def load_label_encoders():
    """Load the dict of LabelEncoders (cached)."""
    return joblib.load(MODEL_DIR / "label_encoders.joblib")


@st.cache_data
def load_test_data():
    """Load pre-split test data for the dashboard (cached)."""
    path = MODEL_DIR / "test_data.joblib"
    if not path.exists():
        return None
    return joblib.load(path)
