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
    page_title="Parklytics — Predicción Parque Warner",
    page_icon="img/logoParklytics.png",
    layout="wide",
    initial_sidebar_state="expanded"
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
    --muted: #6b7280;
    --bg: #ffffff;
    --card: #f8fafc;
    --text: #111827;
    --border: #e6e9ee;
    --shadow: rgba(0, 0, 0, 0.1);
}

/* Base Styles */
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Hero Section - Updated for better image display */
.hero-container {
    position: relative;
    width: 100%;
    height: 600px;  /* Fixed height for a more compact look */
    overflow: hidden;
    margin: 0 0 2rem 0;
    padding: 0;
    box-shadow: 0 10px 25px var(--shadow);
}

.hero-image {
    position: absolute;
    width: 100%;
    height: 100%;
    object-fit: cover;  /* Changed back to cover but with adjusted positioning */
    object-position: center 30%;  /* Adjust vertical position to show more of the image */
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
        rgba(0, 0, 0, 0.2) 0%,
        rgba(255, 69, 0, 0.2) 30%,
        rgba(255, 165, 0, 0.15) 60%,
        rgba(0, 0, 0, 0.6) 100%
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
    color: white;
    padding: 0 1rem;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 1.3rem;
    margin: 0.8rem 0 0;
    font-weight: 400;
    max-width: 700px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    opacity: 0.95;
}

/* Prediction Card */
.prediction-card {
    background: white;
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

/* Responsive Design */
@media (max-width: 768px) {
    .hero-container {
        height: 300px;
    }
    .hero-title {
        font-size: 2.5rem;
    }
    .prediction-value {
        font-size: 2.8rem;
    }
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# HERO SECTION
# -----------------------
def render_hero():
    try:
        hero_image_path = os.path.join("img", "fotoBatman.jpg")
        if os.path.exists(hero_image_path):
            hero_image = get_base64_image(hero_image_path)
            hero_bg = f"url(data:image/jpg;base64,{hero_image})"
            
            st.markdown(f"""
            <style>
                .hero-container {{
                    position: relative;
                    width: 100%;
                    height: 500px;
                    background: {hero_bg} no-repeat center center;
                    background-size: cover;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 12px;
                    overflow: hidden;
                }}
                .hero-content {{
                    text-align: center;
                }}
                .hero-title {{
                    font-size: 4rem;
                    font-weight: 800;
                    margin: 0;
                    color: #FFA500;  /* Naranja dorado que combina con el atardecer */
                    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
                }}
                .hero-subtitle {{
                    font-size: 1.5rem;
                    margin: 1rem 0 0;
                    color: #FFD700;  /* Dorado para el subtítulo */
                    font-weight: 500;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
                }}
                @media (max-width: 768px) {{
                    .hero-container {{
                        height: 300px;
                    }}
                    .hero-title {{
                        font-size: 2.5rem;
                    }}
                    .hero-subtitle {{
                        font-size: 1.1rem;
                    }}
                }}
            </style>
            <div class="hero-container">
                <div class="hero-content">
                    <h1 class="hero-title">Parklytics</h1>
                    <p class="hero-subtitle">Predicción inteligente de tiempos de espera en Parque Warner</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Fallback si no se encuentra la imagen
            st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h1 style="color: #FF8C00; margin: 0;">Parklytics</h1>
                <p style="color: #FFA500; margin: 0.5rem 0 0; font-size: 1.2rem;">
                    Predicción inteligente de tiempos de espera en Parque Warner
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"Error al cargar la imagen: {e}")
        # Fallback en caso de error
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #FF8C00; margin: 0;">Parklytics</h1>
            <p style="color: #FFA500; margin: 0.5rem 0 0; font-size: 1.2rem;">
                Predicción inteligente de tiempos de espera en Parque Warner
            </p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------
# WEATHER CONTROLS
# -----------------------
def render_weather_controls():
    st.markdown("#### 🌤️ Condiciones meteorológicas")
    
    col1, col2 = st.columns(2)
    with col1:
        temperatura = st.slider(
            "Temperatura (°C)", 
            min_value=-5, 
            max_value=45, 
            value=22,
            help="Temperatura en grados Celsius",
            key="temp_slider"
        )
    with col2:
        humedad = st.slider(
            "Humedad (%)", 
            min_value=0, 
            max_value=100, 
            value=60,
            key="humidity_slider"
        )

    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    sensacion_termica = st.slider(
        "Sensación térmica (°C)", 
        min_value=-10, 
        max_value=50, 
        value=temperatura,
        key="feels_like_slider"
    )

    st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
    
    codigo_clima = st.selectbox(
        "Condición meteorológica",
        options=[1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: {
            1: "☀️ Soleado - Excelente",
            2: "⛅ Parcialmente nublado - Bueno",
            3: "☁️ Nublado - Normal",
            4: "🌧️ Lluvia ligera - Malo",
            5: "⛈️ Lluvia fuerte/Tormenta - Muy malo"
        }[x]
    )
    
    return temperatura, humedad, sensacion_termica, codigo_clima

# -----------------------
# MAIN APP
# -----------------------
def main():
    # Render hero section
    render_hero()

    # Disclaimer
    st.markdown("""
    <div style="background: #fff8e6; color: #5c3d00; padding: 1rem; border-radius: 12px; 
                border-left: 4px solid #ffc107; margin-bottom: 2rem;">
        <strong>⚠️ Aviso:</strong> Esta aplicación es independiente y educativa. 
        No está afiliada a Parque Warner.
    </div>
    """, unsafe_allow_html=True)

    # Load model and data
    with st.spinner("Cargando modelo y datos..."):
        try:
            artifacts = load_model_artifacts()
            if not artifacts or "error" in artifacts:
                st.error("❌ Error al cargar el modelo. Por favor, verifica los archivos del modelo.")
                st.stop()
                
            df = artifacts.get("df_processed", pd.DataFrame())
            if df.empty:
                st.error("❌ No se encontraron datos de entrenamiento.")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {str(e)}")
            st.stop()

    # Cached helper functions
    @st.cache_data
    def get_attractions():
        return sorted(df["atraccion"].dropna().astype(str).unique().tolist())

    @st.cache_data
    def get_zones():
        return sorted(df["zona"].dropna().astype(str).unique().tolist())

    def get_zone_for_attraction(attraction):
        row = df[df["atraccion"] == attraction]
        return row["zona"].iloc[0] if not row.empty else ""

    # Get data
    atracciones = get_attractions()
    zonas = get_zones()

    # SIDEBAR
    with st.sidebar:
        st.image("img/logoParklytics.png", use_column_width=True)
        st.markdown("---")
        st.markdown("### ⚙️ Configuración")
        st.markdown("Ajusta los parámetros para obtener una predicción precisa.")

        # Attraction selection
        atraccion_seleccionada = st.selectbox(
            "🎯 Atracción",
            options=atracciones,
            index=0,
            help="Selecciona la atracción que deseas consultar"
        )

        # Auto-detect zone
        zona_auto = get_zone_for_attraction(atraccion_seleccionada)
        if zona_auto:
            st.info(f"📍 Zona: **{zona_auto}**")

        st.markdown("---")

        # Date selection
        fecha_seleccionada = st.date_input(
            "📅 Fecha de visita",
            value=date.today(),
            min_value=date.today(),
            format="DD/MM/YYYY"
        )

        # Day info
        dia_semana_es = {
            "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
            "Thursday": "Jueves", "Friday": "Viernes", 
            "Saturday": "Sábado", "Sunday": "Domingo"
        }
        dia_nombre = fecha_seleccionada.strftime("%A")
        es_fin_semana = fecha_seleccionada.weekday() >= 5
        st.info(f"📆 {dia_semana_es.get(dia_nombre, dia_nombre)} — "
                f"{'Fin de semana' if es_fin_semana else 'Día laborable'}")

        st.markdown("---")

        # Time selection
        st.markdown("#### 🕒 Hora de visita")
        hora_seleccionada = st.time_input(
            "Selecciona la hora",
            value=time(14, 0),  # Default to 2 PM
            step=timedelta(minutes=15)
        )

        # Time of day classification
        hora_int = hora_seleccionada.hour
        if 12 <= hora_int <= 16:
            tipo_hora = "🔴 Hora Pico"
        elif 10 <= hora_int < 12:
            tipo_hora = "🟡 Hora Media"
        else:
            tipo_hora = "🟢 Hora Tranquila"
        st.info(tipo_hora)

        st.markdown("---")

        # Weather settings
        temperatura, humedad, sensacion_termica, codigo_clima = render_weather_controls()

        st.markdown("---")
        
        # Prediction button
        predecir = st.button(
            "🚀 Calcular tiempo de espera",
            type="primary",
            use_container_width=True
        )

    # MAIN CONTENT
    if predecir:
        # Prepare input data
        hora_str = hora_seleccionada.strftime("%H:%M:%S")
        fecha_str = fecha_seleccionada.strftime("%Y-%m-%d")
        
        input_data = {
            "atraccion": atraccion_seleccionada,
            "zona": zona_auto,
            "fecha": fecha_str,
            "hora": hora_str,
            "temperatura": temperatura,
            "humedad": humedad,
            "sensacion_termica": sensacion_termica,
            "codigo_clima": codigo_clima
        }

        # Make prediction
        with st.spinner("🔮 Calculando predicción..."):
            try:
                resultado = predict_wait_time(input_data, artifacts)
                minutos_pred = resultado.get("minutos_predichos", 0)
                
                # Determine prediction style
                if minutos_pred < 15:
                    gradient = "linear-gradient(135deg, #16a085 0%, #2ecc71 100%)"
                    emoji, nivel = "🟢", "Bajo"
                elif minutos_pred < 30:
                    gradient = "linear-gradient(135deg, #f6d365 0%, #fda085 100%)"
                    emoji, nivel = "🟡", "Moderado"
                elif minutos_pred < 60:
                    gradient = "linear-gradient(135deg, #f7971e 0%, #ffd200 100%)"
                    emoji, nivel = "🟠", "Alto"
                else:
                    gradient = "linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)"
                    emoji, nivel = "🔴", "Muy Alto"

                # Display results
                st.markdown("## 📊 Resultados de la predicción")
                
                # Main prediction card
                st.markdown(f"""
                <div class="prediction-card" style="--gradient: {gradient};">
                    <div style="text-align: center; padding: 1.5rem 1rem;">
                        <div style="font-size: 1.2rem; color: var(--muted); margin-bottom: 0.5rem;">
                            {emoji} Tiempo de espera estimado
                        </div>
                        <div class="prediction-value" style="background: {gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {minutos_pred:.1f} min
                        </div>
                        <div class="prediction-label">
                            {nivel} • {atraccion_seleccionada}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                st.markdown("### 📈 Métricas clave")
                col1, col2, col3, col4 = st.columns(4)
                
                metrics = [
                    ("📊 Predicción Base", f"{resultado.get('prediccion_base', 0):.1f} min"),
                    ("📈 P75 Histórico", f"{resultado.get('p75_historico', 0):.1f} min"),
                    ("📉 Mediana", f"{resultado.get('median_historico', 0):.1f} min"),
                    ("🎯 Especificidad", resultado.get('especificidad_historico', 'N/A').replace('_', ' ').title())
                ]
                
                for (col, (label, value)) in zip([col1, col2, col3, col4], metrics):
                    with col:
                        st.metric(label, value)

                # Tabs for detailed information
                tab1, tab2, tab3 = st.tabs(["📝 Información", "🔍 Contexto", "💡 Recomendaciones"])

                with tab1:
                    st.markdown("### 📝 Información de la predicción")
                    info_cols = st.columns(2)
                    
                    with info_cols[0]:
                        st.markdown("#### 📅 Fecha y hora")
                        st.markdown(f"""
                        <div class="info-card">
                            <strong>Día de la semana:</strong> {resultado.get('dia_semana', 'N/A')}<br>
                            <strong>Día del mes:</strong> {resultado.get('dia_mes', 'N/A')}<br>
                            <strong>Hora seleccionada:</strong> {hora_seleccionada.strftime('%H:%M')}<br>
                            <strong>Muestra histórica:</strong> {resultado.get('count_historico', 0):,} registros
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with info_cols[1]:
                        weather_emoji = {
                            1: '☀️ Soleado',
                            2: '⛅ Parcial',
                            3: '☁️ Nublado',
                            4: '🌧️ Lluvia',
                            5: '⛈️ Tormenta'
                        }.get(codigo_clima, 'N/A')
                        
                        st.markdown("#### 🌦️ Condiciones")
                        st.markdown(f"""
                        <div class="info-card">
                            <strong>Temperatura:</strong> {temperatura}°C<br>
                            <strong>Humedad:</strong> {humedad}%<br>
                            <strong>Sensación térmica:</strong> {sensacion_termica}°C<br>
                            <strong>Condición:</strong> {weather_emoji}
                        </div>
                        """, unsafe_allow_html=True)

                with tab2:
                    st.markdown("### 🔍 Contexto")
                    
                    # Context cards
                    context_items = [
                        ("📅 Fin de semana", resultado.get('es_fin_de_semana', False)),
                        ("🌉 Es puente", resultado.get('es_puente', False)),
                        ("⏰ Hora de apertura", resultado.get('es_hora_apertura', False)),
                        ("🔥 Hora pico", resultado.get('es_hora_pico', False)),
                        ("🌿 Hora valle", resultado.get('es_hora_valle', False))
                    ]
                    
                    cols = st.columns(2)
                    for i, (label, value) in enumerate(context_items):
                        with cols[i % 2]:
                            st.markdown(f"""
                            <div class="info-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span>{label}</span>
                                    <span style="color: {'#16a085' if value else '#6b7280'}; 
                                                  font-weight: 600;">
                                        {'Sí' if value else 'No'}
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Chart
                    st.markdown("### 📊 Comparación de predicciones")
                    valores = {
                        "Predicción Final": minutos_pred,
                        "Modelo Base": resultado.get('prediccion_base', 0),
                        "P75 Histórico": resultado.get('p75_historico', 0),
                        "Mediana": resultado.get('median_historico', 0)
                    }
                    
                    fig = go.Figure(go.Bar(
                        x=list(valores.keys()),
                        y=list(valores.values()),
                        text=[f"{v:.1f} min" for v in valores.values()],
                        textposition='auto',
                        marker_color=['#6c63ff', '#4facfe', '#43e97b', '#f6d365']
                    ))
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        margin=dict(t=20, b=20, l=20, r=20),
                        yaxis_title="Minutos",
                        xaxis_title="",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.markdown("### 💡 Recomendaciones")
                    
                    recommendations = []
                    
                    # Time-based recommendations
                    if minutos_pred < 15:
                        recommendations.append(("✅", "Excelente momento", 
                            f"El tiempo de espera es bajo ({minutos_pred:.1f} min). Aprovecha para subir ahora."))
                    elif minutos_pred < 30:
                        recommendations.append(("👍", "Buen momento", 
                            f"El tiempo de espera es moderado ({minutos_pred:.1f} min). Un buen momento para hacer cola."))
                    elif minutos_pred < 60:
                        recommendations.append(("⚠️", "Tiempo de espera alto", 
                            f"El tiempo de espera es alto ({minutos_pred:.1f} min). Considera planificar para otro momento o usar acceso rápido si está disponible."))
                    else:
                        recommendations.append(("🚫", "Tiempo de espera muy alto", 
                            f"El tiempo de espera es muy alto ({minutos_pred:.1f} min). Te recomendamos cambiar de atracción o volver en otro momento."))
                    
                    # Context-based recommendations
                    if resultado.get('es_hora_pico'):
                        recommendations.append(("⏰", "Hora pico", 
                            "Estás en horario de mayor afluencia (11:00-16:00). Las esperas suelen ser más largas."))
                    
                    if resultado.get('es_fin_de_semana'):
                        recommendations.append(("📅", "Fin de semana", 
                            "Los fines de semana suelen tener más visitantes. Si puedes, considera visitar entre semana."))
                    
                    
                    # Display recommendations
                    for emoji, title, text in recommendations:
                        with st.expander(f"{emoji} {title}", expanded=True):
                            st.markdown(f"<div style='padding: 0.5rem 0;'>{text}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error al realizar la predicción: {str(e)}")
                st.exception(e)  # Show full error for debugging

    else:
        # Initial state - show welcome and instructions
        st.markdown("""
        ## 🎢 Bienvenido a Parklytics
        
        Utiliza el panel lateral para configurar tu predicción de tiempo de espera en las atracciones de Parque Warner.
        
        ### Cómo funciona:
        1. Selecciona una atracción de la lista
        2. Elige la fecha y hora de tu visita
        3. Ajusta las condiciones meteorológicas
        4. Haz clic en **Calcular tiempo de espera**
        
        ### 📊 Estadísticas rápidas
        """, unsafe_allow_html=True)
        
        # Quick stats
        if not df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Atracciones", len(atracciones))
            
            with col2:
                st.metric("Zonas", len(zonas))
            
            with col3:
                st.metric("Registros históricos", f"{len(df):,}")
            
            with col4:
                avg_wait = df["tiempo_espera"].mean() if "tiempo_espera" in df.columns else 0
                st.metric("Tiempo medio", f"{avg_wait:.1f} min")
        
        st.markdown("---")
        
        # Tips section
        st.markdown("""
        ### 💡 Consejos para tu visita
        - Las horas con menos afluencia suelen ser a primera hora de la mañana o última de la tarde
        - Los días laborables suelen tener menos visitantes que los fines de semana
        - El tiempo de espera puede variar según las condiciones meteorológicas
        - Revisa las atracciones con menor tiempo de espera en el parque
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        🎢 Parklytics — Predicción de tiempos de espera en tiempo real<br>
        <small>Desarrollado con ❤️ por Sergio López | v2.0</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()