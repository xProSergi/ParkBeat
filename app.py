import streamlit as st
import pandas as pd
import numpy as np
import base64
from datetime import datetime, date, time, timedelta
import plotly.graph_objects as go
# Asegúrate de que 'predict' exista y contenga las funciones 'load_model_artifacts' y 'predict_wait_time'
# from predict import load_model_artifacts, predict_wait_time 
import warnings
import os

# --- MOCKUP DE FUNCIONES PARA QUE EL CÓDIGO SEA EJECUTABLE SIN ARCHIVOS EXTERNOS ---
# ELIMINA ESTAS LÍNEAS SI TIENES EL MÓDULO 'predict' Y LOS DATOS REALES
def load_model_artifacts():
    # Simula la carga de un artefacto con datos mínimos
    data = {
        'atraccion': ['Superman', 'Batman', 'Coaster'],
        'zona': ['DC Super Heroes World', 'DC Super Heroes World', 'Movie World Studios'],
        'wait_time_min': [30, 25, 40]
    }
    df = pd.DataFrame(data)
    return {"df_processed": df, "model": "simulated_model", "scaler": "simulated_scaler"}

def predict_wait_time(input_data, artifacts):
    # Simula una predicción
    # Una lógica simple para variar el resultado
    base_time = 30
    if "hora" in input_data:
        h = int(input_data["hora"].split(":")[0])
        if 11 <= h <= 16: # Hora pico
            base_time += 15
    if input_data.get('es_fin_de_semana', False):
        base_time += 10
        
    prediccion = base_time + np.random.randint(-10, 10) 
    prediccion = max(5, prediccion) # Mínimo 5 min

    return {
        "minutos_predichos": float(prediccion),
        "prediccion_base": float(prediccion - 5),
        "p75_historico": float(prediccion + 8),
        "median_historico": float(prediccion - 10),
        "es_fin_de_semana": input_data.get('fecha', date.today()).weekday() >= 5,
        "es_puente": False,
        "es_hora_pico": 11 <= int(input_data.get("hora", "14:00:00").split(":")[0]) <= 16,
        "es_hora_valle": not (11 <= int(input_data.get("hora", "14:00:00").split(":")[0]) <= 16),
        "dia_semana": {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}.get(input_data.get('fecha', date.today()).weekday()),
        "dia_mes": input_data.get('fecha', date.today()).day
    }
# ----------------------------------------------------------------------------------

# Suppress warnings
warnings.filterwarnings('ignore')

def get_base64_image(image_path):
    """Convert image to base64 for embedding in HTML"""
    # MOCK: solo verifica si la ruta existe para evitar errores, pero no se necesita el base64 real si no se renderiza como imagen
    if not os.path.exists(image_path):
        return None 
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception:
        return None # Devuelve None si no encuentra la imagen

# Page Configuration
st.set_page_config(
    page_title="ParkBeat — Predicción Parque Warner",
    page_icon="🎢", # He cambiado a un emoji para que funcione sin el archivo
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Global Overrides */
    html, body, #root, .stApp {
        margin: 0 !important;
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Main content container */
    .main .block-container {
        padding: 0 1rem 1rem 1rem !important; /* Ajuste para que no se pegue al borde */
        max-width: 100% !important;
    }

    /* FIX para el modo claro/oscuro en títulos y textos */
    /* Asegura que el color de texto del Hero sea siempre visible */
    /* Este es el fix para el título que se volvía negro */
    .stApp.light .hero-title, .stApp.dark .hero-title {
        color: #FF8C00 !important; /* Naranja Intenso para Parklytics */
        text-shadow: 0 2px 8px rgba(0,0,0,0.9), 0 0 4px rgba(255,255,255,0.7) !important; 
    }
    .stApp.light .hero-subtitle, .stApp.dark .hero-subtitle {
        color: #FFD54F !important; /* Amarillo para el subtítulo */
        text-shadow: 0 2px 6px rgba(0,0,0,0.85) !important;
    }
    
    /* Hero Section */
    .hero-container {
        position: relative;
        width: 100%;
        height: 400px; /* Reducido un poco para no ocupar tanto espacio */
        overflow: hidden;
        margin: 0;
        padding: 0;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .hero-content {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        width: 100%;
        padding: 0 1rem;
        z-index: 1;
    }
    
    .hero-title {
        font-size: 4.5rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 2.5rem;
        margin: 1rem 0 0;
        font-weight: 700;
        line-height: 1.4;
        letter-spacing: 0.5px;
        display: inline-block;
        position: relative;
    }
    
    /* Ajustes para el sidebar */
    [data-testid="stSidebar"] {
        padding-top: 2rem;
    }

    /* Redefinición de colores de texto en modo claro para asegurar legibilidad */
    .stApp.light .stMarkdown, .stApp.light .stText, .stApp.light label {
        color: #1a1a1a; /* Texto oscuro en modo claro */
    }
    .stApp.light [data-testid="stInfo"] {
        background-color: #f0f8ff; /* Fondo claro para los cuadros de info */
        border: 1px solid #b3d9ff;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-container {
            height: 300px;
        }
        .hero-title {
            font-size: 3rem;
        }
        .hero-subtitle {
            font-size: 1.5rem;
            margin-top: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def render_hero():
    """Renders the main hero section with background image and title."""
    hero_image_path = os.path.join("img", "fotoBatman.jpg")
    hero_bg = ""
    
    # Try to get the image as base64
    base64_img = get_base64_image(hero_image_path)
    
    if base64_img:
        hero_bg = f"url(data:image/jpg;base64,{base64_img})"
    
    # Use inline style for background if image is available, otherwise just use a fallback color
    background_style = f"background: {hero_bg if hero_bg else '#333333'} no-repeat center center;"
    background_size_style = "background-size: cover;" if hero_bg else ""
    
    st.markdown(f"""
    <div class="hero-container" style="{background_style} {background_size_style}">
        <div class="hero-overlay" style="background: rgba(0, 0, 0, 0.4);"></div>
        <div class="hero-content">
            <h1 class="hero-title">Parklytics</h1>
            <p class="hero-subtitle">Predicción inteligente de tiempos de espera en Parque Warner</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def sidebar_content():
    """Renders the content for the Streamlit sidebar."""
    st.sidebar.markdown("## ℹ️ Sobre este proyecto")
    st.sidebar.image("img/logoParklytics.png") # Muestra el logo si existe
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 🎢 ¿Qué es ParkBeat?")
    st.sidebar.info("""
        **ParkBeat** es una herramienta de predicción impulsada por **Parklytics**.
        Utiliza **modelos de Machine Learning** entrenados con **datos históricos de tiempos de espera**,
        integrando variables clave como:
        
        * Día de la semana / Mes / Año
        * Hora del día
        * Condiciones meteorológicas (temperatura, humedad, clima)
        * Contexto (días festivos, puentes)

        Su objetivo es proporcionar a los visitantes estimaciones precisas de la cola para optimizar su día en el parque.
    """)
    
    st.sidebar.markdown("### 🎯 ¿Por qué este proyecto?")
    st.sidebar.markdown("""
        El proyecto Parklytics nace de la necesidad de mejorar la **experiencia del visitante**. Al reducir la incertidumbre sobre las colas,
        permitimos a los usuarios planificar mejor sus rutas, **minimizar el tiempo de espera** y **maximizar la diversión**. Es una
        demostración práctica del potencial de la Inteligencia Artificial y el Data Science aplicados al entretenimiento.
    """)
    st.sidebar.markdown("---")
    st.sidebar.info("¡Selecciona tus parámetros y haz clic en 'Calcular' para empezar!")


def main():
    # Render Sidebar
    sidebar_content()

    # Hero Section
    render_hero()
    
    # Welcome Section
    st.markdown("---")
    st.markdown("""
    ## 🎢 Bienvenido a ParkBeat
    
    Predice los **tiempos de espera** en las atracciones del **Parque Warner Madrid** con precisión. 
    Simplemente selecciona una atracción, la fecha y la hora de tu visita, y te mostraremos una 
    estimación del tiempo de espera esperado.
    """)
    
    # Load model and data
    with st.spinner("Cargando modelo y datos..."):
        try:
            # Reemplaza 'load_model_artifacts' con tu importación real
            artifacts = load_model_artifacts()
            if not artifacts or "error" in artifacts:
                st.error("❌ Error al cargar el modelo. Por favor, verifica los archivos del modelo.")
                # st.stop() # Comentado para que el mockup funcione
            
            df = artifacts.get("df_processed", pd.DataFrame())
            if df.empty:
                st.warning("⚠️ No se encontraron datos de entrenamiento. Usando datos simulados para demostración.")
                # Simular datos si falla la carga para el mockup
                df = pd.DataFrame({
                    'atraccion': ['Superman', 'Batman', 'Coaster'],
                    'zona': ['DC Super Heroes World', 'DC Super Heroes World', 'Movie World Studios']
                })
                # st.stop() # Comentado para que el mockup funcione
                
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {str(e)}")
            # st.stop() # Comentado para que el mockup funcione

    # Cached helper functions
    @st.cache_data
    def get_attractions():
        return sorted(df["atraccion"].dropna().astype(str).unique().tolist())

    @st.cache_data
    def get_zones():
        return sorted(df["zona"].dropna().astype(str).unique().tolist())

    def get_zone_for_attraction(attraction):
        row = df[df["atraccion"] == attraction]
        return row["zona"].iloc[0] if not row.empty else "Zona no disponible"

    # Get data
    atracciones = get_attractions()
    zonas = get_zones()
    
    # Fallback si no hay atracciones (ejecutando solo el mockup)
    if not atracciones:
        atracciones = ['Superman', 'Batman', 'Coaster']
        
    # Main Controls Section
    st.markdown("## ⚙️ Configura tu predicción")
    
    # Create columns for better organization
    col1, col2 = st.columns(2)
    
    with col1:
        # Attraction selection
        with st.container(border=True):
            st.markdown("### 🎢 Selecciona una atracción")
            atraccion_seleccionada = st.selectbox(
                "Elige una atracción de la lista",
                options=atracciones,
                index=0,
                key="select_atraccion",
                help="Selecciona la atracción que deseas consultar"
            )
            
            # Auto-detect zone
            zona_auto = get_zone_for_attraction(atraccion_seleccionada)
            st.info(f"📍 **Zona:** {zona_auto}")

    with col2:
        # Date and time selection
        with st.container(border=True):
            st.markdown("### 📅 Fecha y hora de visita")
            
            # Date selection
            fecha_seleccionada = st.date_input(
                "Selecciona la fecha",
                value=date.today(),
                min_value=date.today(),
                format="DD/MM/YYYY",
                key="select_fecha"
            )
            
            # Time selection
            hora_seleccionada = st.time_input(
                "Hora de la visita",
                value=time(14, 0),  # Default to 2 PM
                step=timedelta(minutes=15),
                key="select_hora"
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
        col1_w, col2_w = st.columns(2)
        
        with col1_w:
            temperatura = st.slider(
                "Temperatura (°C)", 
                min_value=-5, 
                max_value=45, 
                value=22,
                key="slider_temp",
                help="Temperatura en grados Celsius"
            )
            
            sensacion_termica = st.slider(
                "Sensación térmica (°C)", 
                min_value=-10, 
                max_value=50, 
                value=22,
                key="slider_sens_term"
            )
            
        with col2_w:
            humedad = st.slider(
                "Humedad (%)", 
                min_value=0, 
                max_value=100, 
                value=60,
                key="slider_humedad"
            )

            codigo_clima = st.selectbox(
                "Condición meteorológica",
                options=[1, 2, 3, 4, 5],
                index=2,
                key="select_clima",
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
            "fecha": fecha_seleccionada, # Pasa el objeto date
            "hora": hora_str,
            "temperatura": temperatura,
            "humedad": humedad,
            "sensacion_termica": sensacion_termica,
            "codigo_clima": codigo_clima
        }

        # Make prediction
        st.divider()
        with st.spinner("🔮 Calculando predicción..."):
            try:
                # Reemplaza 'predict_wait_time' con tu importación real
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
                
                # Main prediction card: Uso de variables CSS para adaptarse al tema
                st.markdown(f"""
                <div style="
                    background: var(--secondary-background-color, #f0f2f6);
                    border: 1px solid var(--border-color);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                ">
                    <div style="
                        text-align: center;
                        padding: 1.5rem 1rem;
                    ">
                        <div style="
                            font-size: 1.2rem;
                            color: var(--text-color);
                            margin-bottom: 0.5rem;
                            font-weight: 500;
                        ">
                            {emoji} Tiempo de espera estimado
                        </div>
                        <div style="
                            font-size: 3.5rem;
                            font-weight: 800;
                            margin: 0.5rem 0;
                            background: {gradient};
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            background-clip: text;
                        ">
                            {minutos_pred:.0f} min
                        </div>
                        <div style="
                            font-size: 1.1rem;
                            color: var(--text-color);
                            opacity: 0.9;
                        ">
                            **{nivel}** • {atraccion_seleccionada}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Add tabs for detailed information
                tab1, tab2, tab3 = st.tabs(["📝 Información", "🔍 Contexto", "💡 Recomendaciones"])

                with tab1:
                    st.markdown("### 📝 Información de la predicción")
                    info_cols = st.columns(2)
                    
                    with info_cols[0]:
                        st.markdown("#### 📅 Fecha y hora")
                        st.markdown(f"""
                        * **Fecha seleccionada:** {fecha_seleccionada.strftime('%d/%m/%Y')}
                        * **Hora seleccionada:** {hora_seleccionada.strftime('%H:%M')}
                        * **Día de la semana:** {resultado.get('dia_semana', 'N/A')}
                        """)
                    
                    with info_cols[1]:
                        weather_emoji = {
                            1: '☀️ Soleado', 2: '⛅ Parcial', 3: '☁️ Nublado',
                            4: '🌧️ Lluvia', 5: '⛈️ Tormenta'
                        }.get(codigo_clima, 'N/A')
                        
                        st.markdown("#### 🌦️ Condiciones")
                        st.markdown(f"""
                        * **Temperatura:** {temperatura}°C
                        * **Humedad:** {humedad}%
                        * **Sensación térmica:** {sensacion_termica}°C
                        * **Condición:** {weather_emoji}
                        """)

                with tab2:
                    st.markdown("### 🔍 Contexto y Factores")
                    
                    # Context cards
                    context_items = [
                        ("📅 Fin de semana", resultado.get('es_fin_de_semana', False)),
                        ("🌉 Es puente", resultado.get('es_puente', False)),
                        ("🔥 Hora pico (11:00-16:00)", resultado.get('es_hora_pico', False)),
                        ("🌿 Hora valle", resultado.get('es_hora_valle', False))
                    ]
                    
                    cols_cont = st.columns(4)
                    for i, (label, value) in enumerate(context_items):
                        with cols_cont[i]:
                            st.metric(label, "Sí" if value else "No", delta="Alto Impacto" if value and ("pico" in label or "Fin de semana" in label) else "Bajo Impacto" if not value else None)

                    # Chart
                    st.markdown("---")
                    st.markdown("### 📊 Comparación de predicciones")
                    valores = {
                        "Predicción Final": minutos_pred,
                        "Modelo Base": resultado.get('prediccion_base', 0),
                        "P75 Histórico": resultado.get('p75_historico', 0),
                        "Mediana": resultado.get('median_historico', 0)
                    }
                    
                    # Determinar el color de texto para Plotly según el tema de Streamlit
                    text_color = "black" if st.get_option("theme.base") == "light" else "white"
                    grid_color = "#e6e6e6" if st.get_option("theme.base") == "light" else "#444444"
                    
                    fig = go.Figure(go.Bar(
                        x=list(valores.keys()),
                        y=list(valores.values()),
                        text=[f"{v:.0f} min" for v in valores.values()],
                        textposition='outside',
                        marker_color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'] # Colores distintivos
                    ))
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        margin=dict(t=50, b=20, l=20, r=20),
                        yaxis_title="Minutos de espera",
                        xaxis_title="",
                        showlegend=False,
                        font=dict(color=text_color),
                        xaxis=dict(tickfont=dict(color=text_color)),
                        yaxis=dict(gridcolor=grid_color)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.markdown("### 💡 Recomendaciones")
                    
                    recommendations = []
                    
                    # Time-based recommendations
                    if minutos_pred < 15:
                        recommendations.append(("✅", "¡Adelante!", f"El tiempo de espera es **bajo** ({minutos_pred:.1f} min). ¡Aprovecha para subir ahora y maximizar tu tiempo!"))
                    elif minutos_pred < 30:
                        recommendations.append(("👍", "Buen momento", f"El tiempo de espera es **moderado** ({minutos_pred:.1f} min). Es un momento razonable para hacer cola."))
                    elif minutos_pred < 60:
                        recommendations.append(("⚠️", "Espera considerable", f"El tiempo de espera es **alto** ({minutos_pred:.1f} min). Considera planificar para las últimas horas del parque o usar el pase Correcaminos si lo tienes."))
                    else:
                        recommendations.append(("🚫", "Busca alternativas", f"El tiempo de espera es **muy alto** ({minutos_pred:.1f} min). Es mejor ir a otra atracción y volver en otro momento del día."))
                    
                    # Context-based recommendations
                    if resultado.get('es_hora_pico'):
                        recommendations.append(("⏰", "Horario concurrido", "Estás en **horario de máxima afluencia**. Espera hasta después de las 16:00 para tiempos más bajos."))
                    
                    if resultado.get('es_fin_de_semana'):
                        recommendations.append(("📅", "Más afluencia", "Los fines de semana siempre tienen más visitantes. Prioriza las atracciones más populares temprano o tarde."))
                    
                    # Display recommendations
                    for emoji, title, text in recommendations:
                        with st.container(border=True):
                            st.markdown(f"**{emoji} {title}**")
                            st.markdown(f"{text}")

            except Exception as e:
                st.error(f"❌ Error al realizar la predicción: {str(e)}")
                # st.exception(e)  # Muestra el error completo para debug

    # How it works section (shown when no prediction has been made)
    if not predecir:
        st.markdown("---")
        st.markdown("""
        ## 🎯 ¿Cómo funciona?
        
        1. **Selecciona una atracción** de la lista desplegable.
        2. **Elige la fecha y hora** de tu visita.
        3. **(Opcional)** **Ajusta las condiciones meteorológicas** si lo deseas.
        4. Haz clic en **Calcular tiempo de espera**.
        
        ¡Obtendrás una predicción precisa basada en datos históricos y Machine Learning!
        
        ### 📊 Estadísticas rápidas
        """)
        
        # Quick stats
        if not df.empty and atracciones and zonas:
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
    <div style="text-align: center; color: var(--text-color-faded); padding: 1.5rem 0;">
        🎢 ParkBeat — Predicción de tiempos de espera en tiempo real<br>
        <small>Desarrollado con ❤️ por Sergio López | v2.0</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()