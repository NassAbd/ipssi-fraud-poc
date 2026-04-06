"""
CSS premium et utilitaires partagés pour l'application Streamlit.
"""

PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- KPI Cards ---- */
.kpi-card {
    background: linear-gradient(135deg, #1A1F2E 0%, #242938 100%);
    border: 1px solid rgba(0, 212, 161, 0.15);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s ease;
}
.kpi-card:hover { border-color: rgba(0, 212, 161, 0.4); }
.kpi-value { font-size: 2.2rem; font-weight: 700; margin: 4px 0; }
.kpi-label { font-size: 0.8rem; color: #8892A4; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-delta { font-size: 0.85rem; margin-top: 4px; }
.kpi-green { color: #00D4A1; }
.kpi-red   { color: #FF4B6E; }
.kpi-blue  { color: #4B9FFF; }
.kpi-amber { color: #FFB347; }

/* ---- Verdict cards (Simulateur) ---- */
.verdict-card {
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.verdict-blocked {
    background: linear-gradient(135deg, rgba(255,75,110,0.12) 0%, rgba(255,75,110,0.05) 100%);
    border: 1px solid rgba(255,75,110,0.4);
}
.verdict-valid {
    background: linear-gradient(135deg, rgba(0,212,161,0.12) 0%, rgba(0,212,161,0.05) 100%);
    border: 1px solid rgba(0,212,161,0.4);
}
.verdict-icon { font-size: 3rem; margin-bottom: 8px; }
.verdict-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
.verdict-subtitle { font-size: 0.85rem; color: #8892A4; }

/* ---- Section headers ---- */
.section-header {
    border-left: 3px solid #00D4A1;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #0E1117;
    border-right: 1px solid rgba(255,255,255,0.06);
}


</style>
"""


def kpi_card(label: str, value: str, delta: str = "", color_class: str = "kpi-green") -> str:
    delta_html = f'<div class="kpi-delta {color_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {color_class}">{value}</div>
        {delta_html}
    </div>
    """


def verdict_card(blocked: bool, model_name: str, reason: str = "") -> str:
    css_class = "verdict-blocked" if blocked else "verdict-valid"
    icon = "🔴" if blocked else "🟢"
    status = "BLOQUÉ" if blocked else "VALIDÉ"
    color = "#FF4B6E" if blocked else "#00D4A1"
    return f"""
    <div class="verdict-card {css_class}">
        <div class="verdict-icon">{icon}</div>
        <div class="verdict-title" style="color:{color}">{status}</div>
        <div style="font-size:0.9rem; color:#8892A4; margin: 6px 0;">{model_name}</div>
        <div class="verdict-subtitle">{reason}</div>
    </div>
    """
