# ====================================================
# PARK WAIT TIME PREDICTOR - INTERFAZ STREAMLIT
# Sistema de predicción de tiempos de espera
# ====================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, time
import plotly.express as px
import plotly.graph_objects as go
from predict import load_model_artifacts, predict_wait_time, parse_hora
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción afluencia Parque Warner",
    page_icon="🎢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la interfaz
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    .metric-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00bcd4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Cache para cargar el modelo (solo se carga una vez)
@st.cache_resource
def load_model():
    """Carga el modelo y los artefactos"""
    try:
        return load_model_artifacts()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.info("Asegúrate de que los archivos del modelo estén en la carpeta ../models/")
        return None

# Cache para obtener lista de atracciones
@st.cache_data
def get_attractions_list(_artifacts):
    """Obtiene la lista única de atracciones"""
    if artifacts is None:
        return []
    df = artifacts["df_processed"]
    atracciones = sorted(df["atraccion"].unique().tolist())
    return atracciones

@st.cache_data
def get_zones_list(_artifacts):
    """Obtiene la lista única de zonas"""
    if artifacts is None:
        return []
    df = artifacts["df_processed"]
    zonas = sorted(df["zona"].unique().tolist())
    return zonas

@st.cache_data
def get_zone_for_attraction(_artifacts, atraccion):
    """Obtiene la zona de una atracción"""
    if _artifacts is None:
        return ""
    df = _artifacts["df_processed"]
    zona = df[df["atraccion"] == atraccion]["zona"].iloc[0] if not df[df["atraccion"] == atraccion].empty else ""
    return zona

# Título principal
st.markdown('<h1 class="main-header">Afluencia tiempos Parque Warner</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predicción inteligente de tiempos de espera en atracciones</p>', unsafe_allow_html=True)

# Cargar modelo
artifacts = load_model()

if artifacts is None:
    st.stop()

# Obtener listas de atracciones y zonas
atracciones = get_attractions_list(artifacts)
zonas = get_zones_list(artifacts)

# Sidebar para inputs
with st.sidebar:
    st.header("⚙️ Configuración de Predicción")
    st.markdown("---")
    
    # Selección de atracción
    st.subheader("🎯 Atracción")
    atraccion_seleccionada = st.selectbox(
        "Selecciona una atracción:",
        options=atracciones,
        index=0 if atracciones else None,
        help="Elige la atracción para la que quieres predecir el tiempo de espera"
    )
    
    # Obtener zona automáticamente
    zona_auto = get_zone_for_attraction(artifacts, atraccion_seleccionada)
    
    st.markdown("---")
    
    # Fecha
    st.subheader("📅 Fecha")
    fecha_seleccionada = st.date_input(
        "Selecciona la fecha:",
        value=date.today(),
        min_value=date.today(),
        help="La fecha para la que quieres predecir"
    )
    
    # Información del día
    dia_semana_nombre = fecha_seleccionada.strftime("%A")
    dia_semana_es = {
        "Monday": "Lunes",
        "Tuesday": "Martes",
        "Wednesday": "Miércoles",
        "Thursday": "Jueves",
        "Friday": "Viernes",
        "Saturday": "Sábado",
        "Sunday": "Domingo"
    }
    es_fin_semana = fecha_seleccionada.weekday() >= 5
    tipo_dia = "Fin de semana" if es_fin_semana else "Día laborable"
    
    st.info(f"📆 {dia_semana_es.get(dia_semana_nombre, dia_semana_nombre)} - {tipo_dia}")
    
    st.markdown("---")
    
    # Hora
    st.subheader("🕐 Hora")
    hora_seleccionada = st.time_input(
        "Selecciona la hora:",
        value=time(12, 0),
        help="La hora del día para la predicción"
    )
    
    # Determinar tipo de hora
    hora_int = hora_seleccionada.hour
    if hora_int >= 10 and hora_int < 11:
        tipo_hora = "🟢 Apertura"
        color_hora = "green"
    elif hora_int >= 11 and hora_int <= 16:
        tipo_hora = "🔴 Hora Pico"
        color_hora = "red"
    else:
        tipo_hora = "🟡 Hora Valle"
        color_hora = "orange"
    
    st.info(f"{tipo_hora}")
    
    st.markdown("---")
    
    # Condiciones climáticas
    st.subheader("🌤️ Condiciones Climáticas")
    
    temperatura = st.slider(
        "Temperatura (°C):",
        min_value=-5,
        max_value=45,
        value=22,
        step=1,
        help="Temperatura ambiente en grados Celsius"
    )
    
    humedad = st.slider(
        "Humedad (%):",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        help="Humedad relativa en porcentaje"
    )
    
    sensacion_termica = st.slider(
        "Sensación Térmica (°C):",
        min_value=-10,
        max_value=50,
        value=temperatura,
        step=1,
        help="Sensación térmica percibida"
    )
    
    codigo_clima = st.selectbox(
        "Código de Clima:",
        options=[1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: {
            1: "☀️ Soleado - Excelente",
            2: "⛅ Parcialmente nublado - Bueno",
            3: "☁️ Nublado - Normal",
            4: "🌧️ Lluvia ligera - Malo",
            5: "⛈️ Lluvia fuerte/Tormenta - Muy malo"
        }[x],
        help="Condiciones meteorológicas generales"
    )
    
    st.markdown("---")
    
    # Botón de predicción
    predecir = st.button(
        "🚀 Predecir Tiempo de Espera",
        type="primary",
        use_container_width=True
    )

# Área principal
if predecir:
    # Preparar input para predicción
    hora_str = f"{hora_seleccionada.hour:02d}:{hora_seleccionada.minute:02d}:00"
    fecha_str = fecha_seleccionada.strftime("%Y-%m-%d")
    
    input_dict = {
        "atraccion": atraccion_seleccionada,
        "zona": zona_auto,
        "fecha": fecha_str,
        "hora": hora_str,
        "temperatura": temperatura,
        "humedad": humedad,
        "sensacion_termica": sensacion_termica,
        "codigo_clima": codigo_clima
    }
    
    # Realizar predicción
    with st.spinner("🔮 Calculando predicción..."):
        try:
            resultado = predict_wait_time(input_dict, artifacts)
        except Exception as e:
            st.error(f"Error al realizar la predicción: {str(e)}")
            st.stop()
    
    # Mostrar resultado principal
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        minutos_pred = resultado["minutos_predichos"]
        
        # Color según el tiempo de espera
        if minutos_pred < 15:
            color_grad = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
            emoji = "🟢"
            nivel = "Bajo"
        elif minutos_pred < 30:
            color_grad = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
            emoji = "🟡"
            nivel = "Moderado"
        elif minutos_pred < 60:
            color_grad = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
            emoji = "🟠"
            nivel = "Alto"
        else:
            color_grad = "linear-gradient(135deg, #ff0844 0%, #ffb199 100%)"
            emoji = "🔴"
            nivel = "Muy Alto"
        
        st.markdown(f"""
            <div class="prediction-box" style="background: {color_grad};">
                <div class="prediction-label">{emoji} Tiempo de Espera Predicho</div>
                <div class="prediction-value">{minutos_pred:.1f}</div>
                <div class="prediction-label">minutos - {nivel}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Información detallada en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Predicción Base",
            f"{resultado['prediccion_base']:.1f} min",
            help="Predicción del modelo base sin ajustes"
        )
    
    with col2:
        st.metric(
            "📈 P75 Histórico",
            f"{resultado['p75_historico']:.1f} min",
            help="Percentil 75 del histórico para esta combinación"
        )
    
    with col3:
        st.metric(
            "📉 Mediana Histórica",
            f"{resultado['median_historico']:.1f} min",
            help="Mediana del histórico"
        )
    
    with col4:
        st.metric(
            "🎯 Especificidad",
            resultado['especificidad_historico'].replace("_", " ").title(),
            help="Nivel de especificidad del histórico usado"
        )
    
    st.markdown("---")
    
    # Información adicional
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ℹ️ Información de la Predicción")
        
        info_items = [
            ("🎯 Ajuste Aplicado", resultado['ajuste_aplicado'].replace("_", " ").title()),
            ("📅 Día de la Semana", resultado['dia_semana']),
            ("📆 Día del Mes", f"Día {resultado['dia_mes']}"),
            ("🕐 Hora", f"{resultado['hora']:.2f}"),
            ("📊 Muestra Histórica", f"{resultado['count_historico']} registros"),
        ]
        
        for label, value in info_items:
            st.markdown(f"""
                <div class="info-box">
                    <strong>{label}:</strong> {value}
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🔍 Detalles del Contexto")
        
        context_items = [
            ("🏖️ Fin de Semana", "Sí" if resultado['es_fin_de_semana'] else "No"),
            ("🦇 Batman Octubre", "Sí" if resultado['es_batman_octubre'] else "No"),
            ("🌉 Es Puente", "Sí" if resultado['es_puente'] else "No"),
            ("🟢 Hora Apertura", "Sí" if resultado['es_hora_apertura'] else "No"),
            ("🔴 Hora Pico", "Sí" if resultado['es_hora_pico'] else "No"),
            ("🟡 Hora Valle", "Sí" if resultado['es_hora_valle'] else "No"),
        ]
        
        for label, value in context_items:
            color = "green" if value == "Sí" else "gray"
            st.markdown(f"""
                <div class="info-box">
                    <strong>{label}:</strong> <span style="color: {color};">{value}</span>
                </div>
            """, unsafe_allow_html=True)
    
    # Visualización gráfica
    st.markdown("---")
    st.subheader("📊 Comparación de Predicciones")
    
    # Crear gráfico de comparación
    fig = go.Figure()
    
    valores = {
        "Predicción Final": resultado['minutos_predichos'],
        "Modelo Base": resultado['prediccion_base'],
        "P75 Histórico": resultado['p75_historico'],
        "Mediana Histórica": resultado['median_historico']
    }
    
    colores = ["#667eea", "#f093fb", "#4facfe", "#43e97b"]
    
    fig.add_trace(go.Bar(
        x=list(valores.keys()),
        y=list(valores.values()),
        marker_color=colores,
        text=[f"{v:.1f} min" for v in valores.values()],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Comparación de Valores de Predicción",
        xaxis_title="Tipo de Predicción",
        yaxis_title="Minutos",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recomendaciones
    st.markdown("---")
    st.subheader("💡 Recomendaciones")
    
    recomendaciones = []
    
    if minutos_pred < 15:
        recomendaciones.append("✅ **Excelente momento**: El tiempo de espera es muy bajo. Es el mejor momento para visitar esta atracción.")
    elif minutos_pred < 30:
        recomendaciones.append("👍 **Buen momento**: El tiempo de espera es moderado. Considera visitar esta atracción ahora.")
    elif minutos_pred < 60:
        recomendaciones.append("⚠️ **Tiempo moderado-alto**: El tiempo de espera es considerable. Podrías considerar esperar a otra hora o usar el sistema de acceso rápido si está disponible.")
    else:
        recomendaciones.append("🚫 **Tiempo muy alto**: El tiempo de espera es muy elevado. Se recomienda visitar esta atracción en otro momento del día o considerar otras opciones.")
    
    if resultado['es_hora_pico']:
        recomendaciones.append("⏰ **Hora pico detectada**: Estás en el período de mayor afluencia. Considera visitar fuera de las 11:00-16:00 para tiempos de espera más cortos.")
    
    if resultado['es_fin_de_semana']:
        recomendaciones.append("📅 **Fin de semana**: Los fines de semana suelen tener mayor afluencia. Si es posible, considera visitar en día laborable.")
    
    if resultado['es_batman_octubre']:
        recomendaciones.append("🎃 **Octubre especial**: Octubre es temporada alta para Batman debido a eventos especiales. Los tiempos de espera pueden ser más altos de lo normal.")
    
    for rec in recomendaciones:
        st.markdown(f"<div class='info-box'>{rec}</div>", unsafe_allow_html=True)

else:
    # Mensaje inicial
    st.info("👈 **Configura tu predicción en el panel lateral** y haz clic en 'Predecir Tiempo de Espera' para obtener resultados.")
    
    # Información sobre el sistema
    st.markdown("---")
    st.subheader("📖 Sobre el Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Características
        
        - ✅ Predicción basada en Machine Learning (XGBoost)
        - ✅ Considera día de semana, mes y hora
        - ✅ Incluye condiciones climáticas
        - ✅ Usa históricos granulares por atracción
        - ✅ Detecta eventos especiales (puentes, festivos)
        - ✅ Optimizado para temporadas altas (octubre, verano)
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Factores Considerados
        
        - 📅 **Temporal**: Día de semana, mes, hora del día
        - 🎢 **Atracción**: Características específicas de cada atracción
        - 🌤️ **Clima**: Temperatura, humedad, condiciones meteorológicas
        - 📈 **Históricos**: Patrones de comportamiento pasados
        - 🎉 **Eventos**: Puentes, festivos, temporadas especiales
        """)
    
    st.markdown("---")
    
    # Estadísticas rápidas
    st.subheader("📈 Estadísticas del Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎢 Atracciones", len(atracciones))
    
    with col2:
        st.metric("🌍 Zonas", len(zonas))
    
    with col3:
        df = artifacts["df_processed"]
        total_registros = len(df)
        st.metric("📊 Registros Históricos", f"{total_registros:,}")
    
    with col4:
        tiempo_medio = df["tiempo_espera"].mean()
        st.metric("⏱️ Tiempo Medio", f"{tiempo_medio:.1f} min")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🎢 Predicción afluencias Parque Warner | Sistema de Predicción Inteligente | Powered by XGBoost & Streamlit"
    "</div>",
    unsafe_allow_html=True
)

