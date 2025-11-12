import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from datetime import datetime
from inference import BAEVitInferencer
from utils import overlay_heatmap, normalize_heat
from streamlit_image_comparison import image_comparison

# -------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------------------
st.set_page_config(
    page_title="BAE-ViT – Estimación de Edad Ósea",
    layout="wide",
    page_icon="🦴"
)

# -------------------------------------------------------------
# MODELO (solo se carga una vez)
# -------------------------------------------------------------
ckpt_path = "models/baevit-ckpt_epoch_299.pth"
if "infer" not in st.session_state:
    st.session_state.infer = BAEVitInferencer(ckpt_path=ckpt_path)
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------------------
# ENCABEZADO PRINCIPAL
# -------------------------------------------------------------
st.title("🦴 BAE-ViT — Estimación de Edad Ósea (Regresión con Transformers)")
st.caption(
    "Sube una radiografía de mano, selecciona sexo y obtén la edad ósea estimada. "
    "La aplicación utiliza un modelo Transformer (BAE-ViT) entrenado sobre el dataset RSNA."
)

# -------------------------------------------------------------
# ENTRADA DE DATOS
# -------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    up = st.file_uploader("📤 Radiografía de mano (PNG o JPG)", type=["png", "jpg", "jpeg"])
with col2:
    sex = st.selectbox("Sexo", ["Femenino (F)", "Masculino (M)"])

btn = st.button("🧮 Predecir edad ósea")

# -------------------------------------------------------------
# INFERENCIA
# -------------------------------------------------------------
if btn and up is not None:
    img = Image.open(up).convert("L")
    st.image(img, caption="Radiografía cargada", use_container_width=True)

    # Ejecuta inferencia (usa Dummy o real según el modelo)
    sex_token = "F" if sex.startswith("F") else "M"
    y = st.session_state.infer.predict(img, sex=sex_token, multi_crop=False)

    años = y / 12
    categoria = (
        "Prepuberal" if años < 10 else
        "Puberal" if años < 15 else
        "Postpuberal"
    )

    st.success(f"**Edad ósea estimada:** {y:.1f} meses ({años:.1f} años) — {categoria}")

    # ---------------------------------------------------------
    # HEATMAP SIMULADO (placeholder, reemplázalo con Score-CAM)
    # ---------------------------------------------------------
    fake_heat = normalize_heat(np.random.rand(224, 224))
    overlay = overlay_heatmap(img.convert("RGB"), fake_heat)

    st.markdown("### 🔍 Comparador: radiografía vs. mapa de atención")
    image_comparison(
        img1=img.convert("RGB"),
        img2=overlay,
        label1="Radiografía original",
        label2="Mapa de atención (placeholder)",
        width=700
    )

    # ---------------------------------------------------------
    # HISTORIAL LOCAL
    # ---------------------------------------------------------
    st.session_state.history.append({
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sexo": sex_token,
        "edad_meses": round(y, 1),
        "categoria": categoria
    })

# -------------------------------------------------------------
# PANEL LATERAL (historial + distribución)
# -------------------------------------------------------------
with st.sidebar:
    st.header("📈 Historial de inferencias")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        fig = px.histogram(
            df, x="edad_meses", nbins=15,
            title="Distribución de edades estimadas",
            color_discrete_sequence=["#2ca02c"]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aún no hay predicciones registradas.")