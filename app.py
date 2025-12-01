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

def get_base64_image(image_path):
    """Convert image to base64 for embedding in HTML"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Page Configuration
st.set_page_config(
    page_title="ParkBeat — Predicción Parque Warner",
    page_icon="img/logoParklytics.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Base Styles */
    .hero-container {
        position: relative;
        width: 100%;
        height: 500px;
        overflow: hidden;
        margin: 0 0 2rem 0;
        padding: 0;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .hero-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center 30%;
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
    }
    
    .hero-content {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        color: white;
        width: 100%;
        padding: 0 1rem;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
        color: #FF8C00;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 2rem;
        margin: 1rem 0 0;
        color: #FFD54F;
        font-weight: 700;
        text-shadow: 0 4px 18px rgba(0,0,0,0.85);
    }
    
    /* Card Styles */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e6e9ee;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2d3748;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-container {
            height: 350px;
        }
        .hero-title {
            font-size: 2.5rem;
        }
        .hero-subtitle {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def render_hero():
    try:
        hero_image_path = os.path.join("img", "fotoBatman.jpg")
        hero_image = get_base64_image(hero_image_path)
        
        st.markdown(f"""
        <div class="hero-container">
            <img src="data:image/jpg;base64,{hero_image}" class="hero-image" alt="Parque Warner Madrid">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <h1 class="hero-title">ParkBeat</h1>
                <p class="hero-subtitle">Predicción de tiempos de espera en tiempo real</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0; background: #f8f9fa; border-radius: 12px; margin-bottom: 2rem;">
            <h1 style="color: #2b6ef6; margin: 0; font-size: 3rem; font-weight: 800;">ParkBeat</h1>
            <p style="color: #4a5568; margin: 0.5rem 0 0; font-size: 1.5rem; font-weight: 500;">
                Predicción de tiempos de espera en tiempo real
            </p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Hero Section
    render_hero()
    
    # Welcome Section
    st.markdown("""
    ## 🎢 Bienvenido a ParkBeat
    
    Predice los tiempos de espera en las atracciones del Parque Warner Madrid con precisión. 
    Simplemente selecciona una atracción, la fecha y la hora de tu visita, y te mostraremos una 
    estimación del tiempo de espera esperado.
    """)
    
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

    # Main Controls Section
    st.markdown("## ⚙️ Configura tu predicción")
    
    # Create columns for better organization
    col1, col2 = st.columns(2)
    
    with col1:
        # Attraction selection
        with st.container():
            st.markdown("### 🎢 Selecciona una atracción")
            atraccion_seleccionada = st.selectbox(
                "Elige una atracción de la lista",
                options=atracciones,
                index=0,
                help="Selecciona la atracción que deseas consultar"
            )
            
            # Auto-detect zone
            zona_auto = get_zone_for_attraction(atraccion_seleccionada)
            if zona_auto:
                st.info(f"📍 **Zona:** {zona_auto}")

    with col2:
        # Date and time selection
        with st.container():
            st.markdown("### 📅 Fecha y hora de visita")
            
            # Date selection
            fecha_seleccionada = st.date_input(
                "Selecciona la fecha",
                value=date.today(),
                min_value=date.today(),
                format="DD/MM/YYYY"
            )
            
            # Time selection
            hora_seleccionada = st.time_input(
                "Hora de la visita",
                value=time(14, 0),  # Default to 2 PM
                step=timedelta(minutes=15)
            )
            
            # Day info
            dia_semana_es = {
                "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
                "Thursday": "Jueves", "Friday": "Viernes", 
                "Saturday": "Sábado", "Sunday": "Domingo"
            }
            dia_nombre = fecha_seleccionada.strftime("%A")
            es_fin_semana = fecha_seleccionada.weekday() >= 5
            st.info(f"📆 **Día:** {dia_semana_es.get(dia_nombre, dia_nombre)} - {'Fin de semana' if es_fin_semana else 'Día laborable'}")

    # Weather Section
    with st.expander("🌤️ Configurar condiciones meteorológicas (opcional)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            temperatura = st.slider(
                "Temperatura (°C)", 
                min_value=-5, 
                max_value=45, 
                value=22,
                help="Temperatura en grados Celsius"
            )
            
        with col2:
            humedad = st.slider(
                "Humedad (%)", 
                min_value=0, 
                max_value=100, 
                value=60
            )

        sensacion_termica = st.slider(
            "Sensación térmica (°C)", 
            min_value=-10, 
            max_value=50, 
            value=22
        )

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

    # Prediction button
    predecir = st.button(
        "🚀 Calcular tiempo de espera",
        type="primary",
        use_container_width=True,
        key="predict_button"
    )

    # PREDICTION RESULTS
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
                
                # Display results
                st.markdown("## 📊 Resultados de la predicción")
                
                # Main prediction card
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### Tiempo de espera estimado")
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: #f8f9fa; 
                                border-radius: 12px; border-left: 6px solid #2b6ef6; 
                                margin: 1rem 0;">
                        <div style="font-size: 3.5rem; font-weight: 800; color: #2b6ef6;">
                            {minutos_pred:.0f} min
                        </div>
                        <div style="font-size: 1.1rem; color: #4a5568;">
                            {atraccion_seleccionada}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Detalles")
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px;">
                        <p><strong>📅 Fecha:</strong> {fecha_seleccionada.strftime('%A %d/%m/%Y')}</p>
                        <p><strong>⏰ Hora:</strong> {hora_seleccionada.strftime('%H:%M')}</p>
                        <p><strong>🌡️ Temperatura:</strong> {temperatura}°C</p>
                        <p><strong>💧 Humedad:</strong> {humedad}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Additional details in tabs
                tab1, tab2 = st.tabs(["📈 Análisis", "💡 Recomendaciones"])

                with tab1:
                    st.markdown("### 📊 Datos históricos")
                    # Add your historical data visualization here
                    st.line_chart(data=pd.DataFrame({
                        'Hora del día': ['9:00', '12:00', '15:00', '18:00'],
                        'Tiempo de espera (min)': [15, 45, 60, 30]
                    }).set_index('Hora del día'))

                with tab2:
                    st.markdown("### 💡 Consejos para tu visita")
                    st.markdown("""
                    - Las horas con menos afluencia suelen ser a primera hora de la mañana o última de la tarde
                    - Los días laborables suelen tener menos visitantes que los fines de semana
                    - El tiempo de espera puede variar según las condiciones meteorológicas
                    - Revisa las atracciones con menor tiempo de espera en el parque
                    """)

            except Exception as e:
                st.error(f"❌ Error al realizar la predicción: {str(e)}")

    # How it works section (shown when no prediction has been made)
    if not predecir:
        st.markdown("""
        ## 🎯 ¿Cómo funciona?
        
        1. **Selecciona una atracción** de la lista desplegable
        2. **Elige la fecha y hora** de tu visita
        3. **Ajusta las condiciones meteorológicas** si lo deseas
        4. Haz clic en **Calcular tiempo de espera**
        
        ¡Obtendrás una predicción precisa basada en datos históricos y condiciones actuales!
        
        ### 📊 Estadísticas rápidas
        """)
        
        # Quick stats
        if not df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Atracciones disponibles", len(atracciones))
            
            with col2:
                st.metric("Zonas del parque", len(zonas))
            
            with col3:
                st.metric("Registros históricos", f"{len(df):,}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 1.5rem 0;">
        🎢 ParkBeat — Predicción de tiempos de espera en tiempo real<br>
        <small>Desarrollado con ❤️ por Sergio López | v2.0</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()