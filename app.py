# app_streamlit_mejorada.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import plotly.graph_objects as go
from predict import load_model_artifacts, predict_wait_time, parse_hora
import warnings
warnings.filterwarnings('ignore')

# -----------------------
# Configuración de la página
# -----------------------
st.set_page_config(
    page_title="Parklytics — Predicción Parque Warner",
    page_icon="img/logoParklytics.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# CSS PERSONALIZADO
# -----------------------
st.markdown(
    """
    <style>
    :root{
        --primary: #2b6ef6;
        --accent: #6c63ff;
        --muted: #6b7280;
        --bg: #ffffff;
        --card: #f8fafc;
        --border: #e6e9ee;
        --text: #111827;
    }

    html, body, .css-18e3th9 {
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    body, .block-container {
        font-family: Inter, system-ui;
        line-height: 1.45;
    }

    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--primary);
    }
    .sub-header {
        color: var(--muted);
        margin-bottom: 0.9rem;
        font-size: 1rem;
    }

    .header-row {
        display:flex;
        align-items:center;
        gap:1rem;
    }
    .logo-img {
        width:64px;
        height:64px;
        border-radius:12px;
        box-shadow: 0 6px 18px rgba(43,110,246,0.12);
    }

    .prediction-box{
        border-radius:14px;
        padding:1.2rem;
        color: #fff;
        box-shadow: 0 12px 30px rgba(16,24,40,0.06);
        text-align:center;
    }
    .prediction-value { font-size:3.2rem; font-weight:800; }
    .prediction-label { font-size:1rem; opacity:0.95; margin-top:0.25rem; }

    .info-box {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius:10px;
        padding:0.9rem;
        margin-bottom:0.6rem;
    }

    .disclaimer {
        background: linear-gradient(180deg, #fff8e6, #fffdf6);
        color: #5c3d00;
        padding:0.9rem;
        border-left:4px solid #ffc107;
        border-radius:10px;
    }

    .footer {
        color:var(--muted);
        text-align:center;
        padding:0.75rem 0;
        margin-top:1.25rem;
        border-top:1px solid var(--border);
    }

    .stButton>button {
        background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1rem !important;
        border-radius: 10px !important;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# CACHE y UTILIDADES
# -----------------------
@st.cache_resource
def load_model():
    try:
        return load_model_artifacts()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data
def get_attractions_list(artifacts):
    if not artifacts or "error" in artifacts:
        return []
    df = artifacts["df_processed"]
    return sorted(df["atraccion"].dropna().unique().tolist())

@st.cache_data
def get_zones_list(artifacts):
    if not artifacts or "error" in artifacts:
        return []
    df = artifacts["df_processed"]
    return sorted(df["zona"].dropna().unique().tolist())

@st.cache_data
def get_zone_for_attraction(artifacts, atraccion):
    if not artifacts or "error" in artifacts:
        return ""
    df = artifacts["df_processed"]
    row = df[df["atraccion"] == atraccion]
    if row.empty:
        return ""
    return str(row["zona"].iloc[0])


# -----------------------
# CABECERA
# -----------------------
col1, col2 = st.columns([0.12, 0.88])
with col1:
    try:
        st.image("img/fotoBatman.jpg", width=64)
    except:
        pass

with col2:
    st.markdown(
        '<div class="header-row"><div><h1 class="main-header">Parklytics</h1><div class="sub-header">Predicción inteligente de tiempos de espera — Parque Warner</div></div></div>',
        unsafe_allow_html=True
    )

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Aviso:</strong> Esta aplicación es independiente y educativa. No está afiliada a Parque Warner.
</div>
""", unsafe_allow_html=True)

# -----------------------
# Cargar modelo/artifacts
# -----------------------
artifacts = load_model()
if not artifacts:
    st.error("Error al cargar artefactos del modelo. Revisa la carpeta ../models/ o los logs.")
    st.stop()
if "error" in artifacts:
    st.error(f"Error cargando modelo: {artifacts['error']}")
    st.stop()

# Listas
atracciones = get_attractions_list(artifacts)
zonas = get_zones_list(artifacts)

# -----------------------
# Sidebar (inputs agrupados y claros)
# -----------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption("Ajusta los filtros y parámetros para obtener una predicción precisa.")
    st.markdown("---")

    # Atracción (searchable)
    st.subheader("🎯 Atracción")
    if atracciones:
        atr_index = 0
        try:
            atr_index = atracciones.index(atracciones[0])
        except Exception:
            atr_index = 0
        atraccion_seleccionada = st.selectbox(
            "Selecciona una atracción",
            options=atracciones,
            index=0,
            help="Busca o selecciona la atracción (puedes escribir para filtrar)."
        )
    else:
        atraccion_seleccionada = st.text_input("Atracción (lista vacía)", value="")

    # Zona auto
    zona_auto = get_zone_for_attraction(artifacts, atraccion_seleccionada) if atraccion_seleccionada else ""
    if zona_auto:
        st.info(f"🔎 Zona detectada: **{zona_auto}**")

    st.markdown("---")
    # Fecha
    st.subheader("📅 Fecha")
    fecha_seleccionada = st.date_input(
        "Fecha de visita",
        value=date.today(),
        min_value=date.today(),
        help="Selecciona la fecha para la que quieres la predicción"
    )

    # Info día
    dia_semana_es = {
        "Monday": "Lunes","Tuesday": "Martes","Wednesday": "Miércoles",
        "Thursday": "Jueves","Friday": "Viernes","Saturday": "Sábado","Sunday": "Domingo"
    }
    dia_nombre = fecha_seleccionada.strftime("%A")
    es_fin_semana = fecha_seleccionada.weekday() >= 5
    st.info(f"📆 {dia_semana_es.get(dia_nombre, dia_nombre)} — {'Fin de semana' if es_fin_semana else 'Día laborable'}")

    st.markdown("---")
    # Hora: generador de opciones (12:00 - 00:00 cada 15min)
    st.subheader("🕐 Hora")
    def generate_time_options():
        times = []
        t = datetime.strptime("12:00", "%H:%M")
        end = datetime.strptime("23:59", "%H:%M")
        while t <= end:
            times.append(t.strftime("%H:%M"))
            t += timedelta(minutes=15)
        times.append("00:00")
        return times
    time_options = generate_time_options()
    hora_seleccionada_str = st.selectbox(
        "Selecciona la hora (12:00 - 00:00)",
        options=time_options,
        index=0,
        help="Selecciona o escribe para filtrar la hora (intervalos de 15 min)."
    )
    hora_seleccionada = datetime.strptime(hora_seleccionada_str, "%H:%M").time()

    # Tipo de hora (mejor etiquetado)
    hora_int = hora_seleccionada.hour
    if 12 <= hora_int <= 16 or (hora_int == 0 and hora_seleccionada.minute == 0):
        tipo_hora = "🔴 Hora Pico"
    elif 10 <= hora_int < 11:
        tipo_hora = "🟢 Apertura"
    else:
        tipo_hora = "🟡 Hora Valle"
    st.info(tipo_hora)

    st.markdown("---")
    # Clima (sliders con labels claros)
    st.subheader("🌤️ Clima")
    temperatura = st.slider("Temperatura (°C)", min_value=-5, max_value=45, value=22, step=1)
    humedad = st.slider("Humedad (%)", min_value=0, max_value=100, value=60, step=5)
    sensacion_termica = st.slider("Sensación térmica (°C)", min_value=-10, max_value=50, value=temperatura, step=1)

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

    st.markdown("---")
    # Botón predict
    predecir = st.button("🚀 Predecir Tiempo de Espera")

# -----------------------
# Zona principal: resultado o guía
# -----------------------
if predecir:
    # Preparamos input
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

    # Llamada predict
    with st.spinner("🔮 Calculando..."):
        try:
            resultado = predict_wait_time(input_dict, artifacts)
        except Exception as e:
            st.error(f"Error al predecir: {e}")
            st.stop()

    # Resultados principales (tarjeta grande, colores y copy claro)
    minutos_pred = resultado.get("minutos_predichos", 0.0)
    if minutos_pred < 15:
        color_grad = "linear-gradient(135deg, #16a085 0%, #2ecc71 100%)"
        emoji = "🟢"
        nivel = "Bajo"
    elif minutos_pred < 30:
        color_grad = "linear-gradient(135deg,#f6d365 0%,#fda085 100%)"
        emoji = "🟡"
        nivel = "Moderado"
    elif minutos_pred < 60:
        color_grad = "linear-gradient(135deg,#f7971e 0%,#ffd200 100%)"
        emoji = "🟠"
        nivel = "Alto"
    else:
        color_grad = "linear-gradient(135deg,#ff416c 0%,#ff4b2b 100%)"
        emoji = "🔴"
        nivel = "Muy Alto"

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
            <div class="prediction-box" style="background: {color_grad};">
                <div class="prediction-label">{emoji} Tiempo de Espera Predicho</div>
                <div class="prediction-value">{minutos_pred:.1f}</div>
                <div class="prediction-label">{nivel} — {atraccion_seleccionada}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Metrics resumidas
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Predicción Base", f"{resultado.get('prediccion_base', 0.0):.1f} min")
    with mcol2:
        st.metric("P75 Histórico", f"{resultado.get('p75_historico', 0.0):.1f} min")
    with mcol3:
        st.metric("Mediana Histórica", f"{resultado.get('median_historico', 0.0):.1f} min")
    with mcol4:
        st.metric("Especificidad", resultado.get('especificidad_historico', 'N/A').replace("_", " ").title())

    st.markdown("---")

    # Info y contexto (tarjetas)
    info_col, context_col = st.columns(2)
    with info_col:
        st.subheader("ℹ️ Información de la predicción")
        hora_formateada = hora_seleccionada.strftime("%H:%M")
        info_items = [
            ("Ajuste aplicado", resultado.get('ajuste_aplicado', "N/A").replace("_", " ").title()),
            ("Día de la semana", resultado.get('dia_semana', "N/A")),
            ("Día del mes", f"Día {resultado.get('dia_mes', 'N/A')}"),
            ("Hora", hora_formateada),
            ("Muestra histórica", f"{resultado.get('count_historico', 0)} registros"),
        ]
        for label, value in info_items:
            st.markdown(f"<div class='info-box'><strong>{label}:</strong> {value}</div>", unsafe_allow_html=True)

    with context_col:
        st.subheader("🔍 Contexto")
        context_items = [
            ("Fin de semana", "Sí" if resultado.get('es_fin_de_semana') else "No"),
            ("Evento Batman octubre", "Sí" if resultado.get('es_batman_octubre') else "No"),
            ("Es puente", "Sí" if resultado.get('es_puente') else "No"),
            ("Hora apertura", "Sí" if resultado.get('es_hora_apertura') else "No"),
            ("Hora pico", "Sí" if resultado.get('es_hora_pico') else "No"),
            ("Hora valle", "Sí" if resultado.get('es_hora_valle') else "No"),
        ]
        for label, value in context_items:
            color = "#16a085" if value == "Sí" else "#6b7280"
            st.markdown(f"<div class='info-box'><strong>{label}:</strong> <span style='color:{color}'>{value}</span></div>", unsafe_allow_html=True)

    # Gráfico comparativo
    st.markdown("---")
    st.subheader("📊 Comparación de predicciones")
    valores = {
        "Predicción Final": resultado.get('minutos_predichos', 0.0),
        "Modelo Base": resultado.get('prediccion_base', 0.0),
        "P75 Histórico": resultado.get('p75_historico', 0.0),
        "Mediana Histórica": resultado.get('median_historico', 0.0),
    }
    colores = ["#6c63ff", "#f6d365", "#4facfe", "#43e97b"]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(valores.keys()), y=list(valores.values()), marker_color=colores, text=[f"{v:.1f} min" for v in valores.values()], textposition="auto"))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=380,
        margin=dict(t=40, b=20, l=20, r=20),
        yaxis_title="Minutos"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recomendaciones con copy más humano
    st.markdown("---")
    st.subheader("💡 Recomendaciones")
    recomendaciones = []
    if minutos_pred < 15:
        recomendaciones.append("✅ **Excelente momento**: espera muy baja. Aprovecha para subir ahora.")
    elif minutos_pred < 30:
        recomendaciones.append("👍 **Buen momento**: espera moderada. Ideal si no quieres largas colas.")
    elif minutos_pred < 60:
        recomendaciones.append("⚠️ **Tiempo moderado/alto**: considera planificación o acceso rápido si está disponible.")
    else:
        recomendaciones.append("🚫 **Tiempo muy alto**: valora cambiar de atracción o esperar a otra franja horaria.")

    if resultado.get('es_hora_pico'):
        recomendaciones.append("⏰ Detectada hora pico: evita 11:00–16:00 para reducir esperas.")
    if resultado.get('es_fin_de_semana'):
        recomendaciones.append("📅 Fin de semana: suele haber más afluencia. Si puedes, visita entre semana.")
    if resultado.get('es_batman_octubre'):
        recomendaciones.append("🎃 Evento especial: octubre aumenta la afluencia en Batman.")

    for rec in recomendaciones:
        st.markdown(f"<div class='info-box'>{rec}</div>", unsafe_allow_html=True)

else:
    # Pantalla inicial con guías y stats
    st.info("👈 Configura tu predicción en el panel lateral y pulsa 'Predecir Tiempo de Espera'.")

    st.markdown("---")
    st.subheader("📖 Sobre el sistema")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - ✅ Modelo XGBoost entrenado con históricos por atracción
        - ✅ Considera clima, día y eventos especiales
        - ✅ Recomendaciones en lenguaje claro
        """)
    with col2:
        st.markdown("""
        - 📍 Fácil de usar: ajusta parámetros en la barra lateral
        - ⚠️ No oficial: datos orientativos y educativos
        - 🧠 Mejora continua con nuevos históricos
        """)

    st.markdown("---")
    st.subheader("📈 Estadísticas rápidas")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Atracciones", len(atracciones))
    s2.metric("Zonas", len(zonas))
    df = artifacts.get("df_processed", pd.DataFrame())
    s3.metric("Registros históricos", f"{len(df):,}")
    tiempo_medio = df["tiempo_espera"].mean() if not df.empty and "tiempo_espera" in df.columns else 0.0
    s4.metric("Tiempo medio", f"{tiempo_medio:.1f} min")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    🎢 Parklytics — Predicción de afluencias | Hecho por Sergio López
</div>
""", unsafe_allow_html=True)
