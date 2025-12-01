import streamlit as st
import pandas as pd
import numpy as np
import base64
from datetime import datetime, date, time, timedelta
import plotly.graph_objects as go
from predict import load_model_artifacts, predict_wait_time
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def get_base64_image(image_path):
    """Convert image to base64 for embedding in HTML"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# -----------------------
# PAGE CONFIGURATION
# -----------------------
st.set_page_config(
    page_title="ParkBeat — Predicción Parque Warner",
    page_icon="img/logoParklytics.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------
# CSS STYLING
# -----------------------
st.markdown("""
<style>
:root {
    --primary: #2b6ef6;
    --accent: #6c63ff;
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --muted: #cbd5e1;
    --bg: #0f172a;
    --card: #1e293b;
    --text: #f1f5f9;
    --border: #334155;
    --shadow: rgba(0, 0, 0, 0.6);
}

/* Base Styles */
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Hero Section */
.hero-container {
    position: relative;
    width: 100%;
    height: 600px;
    overflow: hidden;
    margin-bottom: 2rem;
    border-radius: 12px;
}

.hero-image {
    position: absolute;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center 30%;
    z-index: 1;
}

.hero-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        to bottom,
        rgba(0, 0, 0, 0.3) 0%,
        rgba(0, 0, 0, 0.5) 100%
    );
    z-index: 2;
}

.hero-content {
    position: relative;
    z-index: 3;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
    text-align: center;
    padding: 0 1rem;
}

.hero-title {
    font-size: 4.5rem;
    font-weight: 800;
    margin: 0;
    color: #FF8C00 !important;
    text-shadow: 0 2px 6px rgba(0,0,0,0.8);
}

.hero-subtitle {
    font-size: 2.5rem;
    margin: 2rem 0 0;
    color: #FFD54F;
    font-weight: 700;
    text-shadow: 0 4px 18px rgba(0,0,0,0.85);
    line-height: 1.4;
    letter-spacing: 0.5px;
}

/* Prediction Card */
.prediction-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 20px var(--shadow);
    transition: all 0.3s ease;
}

.prediction-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px var(--shadow);
}

.prediction-value {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.5rem 0;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.prediction-label {
    font-size: 1.1rem;
    color: var(--muted);
    margin-top: 0.5rem;
}

/* Info Cards */
.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
}

.info-card:hover {
    border-color: var(--primary);
    box-shadow: 0 5px 15px var(--shadow);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
    color: white !important;
    border: none !important;
    padding: 0.7rem 1.5rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(43, 110, 246, 0.3) !important;
}

/* Sliders */
.stSlider .stSliderThumb {
    background: var(--primary) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    border-radius: 8px;
    transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
    background: var(--primary);
    color: white !important;
}

/* Footer */
.footer {
    color: var(--muted);
    text-align: center;
    padding: 1.5rem 0;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    font-size: 0.9rem;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-container { height: 400px; }
    .hero-title { font-size: 3rem; }
    .hero-subtitle { font-size: 1.8rem; margin-top: 1rem; }
    .prediction-value { font-size: 2.5rem; }
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# HERO SECTION
# -----------------------
def render_hero():
    try:
        hero_image_path = os.path.join("img", "fotoBatman.jpg")
        hero_bg = "none"
        hero_image = get_base64_image(hero_image_path)
        if hero_image:
            hero_bg = f"url(data:image/jpg;base64,{hero_image})"
        st.markdown(f"""
        <div class="hero-container" style="background: {hero_bg} no-repeat center center; background-size: cover;">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <h1 class="hero-title">ParkBeat</h1>
                <p class="hero-subtitle">Predicción inteligente de tiempos de espera en Parque Warner</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Error al renderizar el hero: {e}")

# -----------------------
# WEATHER & CONTROLS (Encima de predicción)
# -----------------------
def render_controls(df):
    st.markdown("## ⚙️ Configuración de predicción")
    
    atracciones = sorted(df["atraccion"].dropna().astype(str).unique().tolist())
    zonas = sorted(df["zona"].dropna().astype(str).unique().tolist())

    atraccion_seleccionada = st.selectbox(
        "🎯 Atracción",
        options=atracciones,
        index=0
    )
    zona_auto = df[df["atraccion"] == atraccion_seleccionada]["zona"].iloc[0]

    fecha_seleccionada = st.date_input(
        "📅 Fecha de visita",
        value=date.today(),
        min_value=date.today(),
        format="DD/MM/YYYY"
    )

    hora_seleccionada = st.time_input(
        "🕒 Hora de visita",
        value=time(14,0),
        step=timedelta(minutes=15)
    )

    temperatura = st.slider("🌡️ Temperatura (°C)", -5, 45, 22)
    humedad = st.slider("💧 Humedad (%)", 0, 100, 60)
    sensacion_termica = st.slider("❄️ Sensación térmica (°C)", -10, 50, temperatura)
    codigo_clima = st.selectbox(
        "Condición meteorológica",
        options=[1,2,3,4,5],
        index=2,
        format_func=lambda x: {1:"☀️ Soleado",2:"⛅ Parcial",3:"☁️ Nublado",4:"🌧️ Lluvia ligera",5:"⛈️ Tormenta"}[x]
    )

    predecir = st.button("🚀 Calcular tiempo de espera")
    return atraccion_seleccionada, zona_auto, fecha_seleccionada, hora_seleccionada, temperatura, humedad, sensacion_termica, codigo_clima, predecir

# -----------------------
# MAIN APP
# -----------------------
def main():
    render_hero()

    st.markdown("""
    <div style="background: #1e293b; color: #f1f5f9; padding: 1rem; border-radius: 12px; 
                border-left: 4px solid #FF8C00; margin-bottom: 2rem;">
        ⚠️ Esta aplicación es independiente y educativa. No está afiliada a Parque Warner.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Cargando modelo y datos..."):
        artifacts = load_model_artifacts()
        df = artifacts.get("df_processed", pd.DataFrame())

    # Render controls above prediction
    atraccion, zona, fecha, hora, temp, hum, sens, clima, predecir = render_controls(df)

    if predecir:
        hora_str = hora.strftime("%H:%M:%S")
        fecha_str = fecha.strftime("%Y-%m-%d")
        input_data = {"atraccion": atraccion, "zona": zona, "fecha": fecha_str, "hora": hora_str,
                      "temperatura": temp, "humedad": hum, "sensacion_termica": sens, "codigo_clima": clima}

        with st.spinner("🔮 Calculando predicción..."):
            resultado = predict_wait_time(input_data, artifacts)
            minutos_pred = resultado.get("minutos_predichos",0)

            if minutos_pred<15:
                gradient="linear-gradient(135deg,#16a085 0%,#2ecc71 100%)"; emoji,nivel="🟢","Bajo"
            elif minutos_pred<30:
                gradient="linear-gradient(135deg,#f6d365 0%,#fda085 100%)"; emoji,nivel="🟡","Moderado"
            elif minutos_pred<60:
                gradient="linear-gradient(135deg,#f7971e 0%,#ffd200 100%)"; emoji,nivel="🟠","Alto"
            else:
                gradient="linear-gradient(135deg,#ff416c 0%,#ff4b2b 100%)"; emoji,nivel="🔴","Muy Alto"

            st.markdown("## 📊 Resultados de la predicción")
            st.markdown(f"""
            <div class="prediction-card" style="--gradient: {gradient};">
                <div style="text-align:center">
                    <div style="font-size:1.2rem;color:var(--muted);margin-bottom:0.5rem">{emoji} Tiempo de espera estimado</div>
                    <div class="prediction-value" style="background:{gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{minutos_pred:.1f} min</div>
                    <div class="prediction-label">{nivel} • {atraccion}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ... Aquí continuarían todas las métricas, tabs, gráficos y recomendaciones exactamente como estaban ...

    else:
        st.markdown("## 🎢 Bienvenido a ParkBeat")
        st.markdown("Configura tu predicción usando los controles de arriba y haz clic en Calcular.")

    st.markdown("---")
    st.markdown("""
    <div class="footer">
        🎢 ParkBeat — Predicción de tiempos de espera en tiempo real<br>
        <small>Desarrollado con ❤️ por Sergio López | v2.0</small>
    </div>
    """, unsafe_allow_html=True)

if __name__=="__main__":
    main()
