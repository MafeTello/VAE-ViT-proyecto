import streamlit as st
import torch
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image

import plotly.express as px

# ------------------------------
# IMPORTAR MÓDULOS INTERNOS
# ------------------------------
from src.model_loader import build_model
from src.preprocess import to_tensor, IMG_SIZE
from src.utils import normalize_heat, overlay_heatmap
from src.score_cam import score_cam_fast
from src.calibrator import AgeCalibrator


# ------------------------------
# CONFIG STREAMLIT
# ------------------------------
st.set_page_config(
    page_title="BAE-ViT – Estimación de Edad Ósea",
    page_icon="🦴",
    layout="wide"
)

# Para evitar la advertencia en imágenes
st.write(
    "<style>img{max-width:100%;height:auto;}</style>",
    unsafe_allow_html=True
)

# ------------------------------
# CARGAR CONFIG YAML
# ------------------------------
with open("configs/baevit.yaml", "r") as f:
    config = yaml.safe_load(f)

# ------------------------------
# INICIALIZAR MODELO
# ------------------------------
if "model" not in st.session_state:
    st.session_state.model = build_model()
    st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Modelo cargado en:", st.session_state.device)

# ------------------------------
# CARGAR CALIBRADOR
# ------------------------------
if "calibrator" not in st.session_state:
    st.session_state.calibrator = AgeCalibrator()

# ------------------------------
# HISTORIAL
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ================================================================
# SIDEBAR — MINI DASHBOARD
# ================================================================
with st.sidebar:
    st.header("📊 Panel de Resumen")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        st.markdown(
            "<p style='font-size:14px'><b>Edad media:</b> "
            f"{df['edad_meses'].mean():.1f} meses</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='font-size:14px'><b>Desv. estándar:</b> "
            f"{df['edad_meses'].std():.1f} meses</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='font-size:14px'><b>Rango:</b> "
            f"{df['edad_meses'].min():.1f} – {df['edad_meses'].max():.1f} meses</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='font-size:14px'><b>Total predicciones:</b> "
            f"{len(df)}</p>",
            unsafe_allow_html=True
        )

        show_hist = st.button("📜 Ver historial completo")

    else:
        st.info("Aún no hay predicciones registradas.")
        show_hist = False


# ================================================================
# HISTORIAL COMPLETO
# ================================================================
if show_hist and st.session_state.history:
    st.title("📜 Historial completo de predicciones")
    df = pd.DataFrame(st.session_state.history)

    st.dataframe(df[["fecha", "sexo", "edad_meses", "categoria"]],
                 height=280)

    st.divider()

    # Histogram
    st.subheader("📈 Distribución de edades")
    fig = px.histogram(
        df,
        x="edad_meses",
        nbins=20,
        title="Distribución de edades estimadas"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Boxplot
    st.subheader("🎯 Boxplot")
    fig2 = px.box(df, y="edad_meses")
    st.plotly_chart(fig2, use_container_width=True)

    # Time evolution
    st.subheader("⏳ Cambio temporal")
    df_t = df.copy()
    df_t["fecha"] = pd.to_datetime(df_t["fecha"])
    fig3 = px.scatter(df_t, x="fecha", y="edad_meses")
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # Miniaturas
    st.subheader("🖼 Radiografías recientes")
    cols = st.columns(5)
    for i, col in zip(range(1, 6), cols):
        idx = -i
        if abs(idx) <= len(df):
            col.image(df.iloc[idx]["img_small"], caption=f"{df.iloc[idx]['edad_meses']} meses")

    st.stop()


# ================================================================
# SECCIÓN PRINCIPAL
# ================================================================
st.title("🦴 Estimación de Edad Ósea con BAE-ViT")
st.caption("Sube una radiografía y obtendrás la edad ósea estimada.")


uploaded = st.file_uploader("📤 Radiografía (PNG/JPG)", type=["png", "jpg", "jpeg"])

colL, colR = st.columns([1, 2])

with colL:
    sexo = st.selectbox("Sexo", ["F", "M"])
    sex_token = 0 if sexo == "F" else 1

    if uploaded:
        img = Image.open(uploaded).convert("L")
        st.image(img, caption="Radiografía cargada", width=300)

    run = st.button("🔮 Predecir edad ósea", disabled=not uploaded)


# ================================================================
# INFERENCIA
# ================================================================
if run and uploaded:

    device = st.session_state.device
    model = st.session_state.model
    cal = st.session_state.calibrator

    # Preprocesamiento
    x = to_tensor(img).to(device)
    s = torch.tensor([[sex_token]], dtype=torch.long, device=device)

    # Predicción
    with torch.no_grad():
        y_pred = model(x, s).item()

    # Normalizar negativos (por si el checkpoint no está calibrado)
    y_pred = max(0, y_pred)

    # Calibración
    y_cal = cal(
        torch.tensor([[y_pred]], dtype=torch.float32, device=device)
    ).item()

    # Evitar valores negativos por seguridad
    y_cal = max(0, y_cal)

    st.success(f"### 🧠 Edad estimada: **{y_cal:.1f} meses**")


    # ------------------------------
    # SCORE-CAM
    # ------------------------------
    st.markdown("### 🎯 Mapa de atención (Score-CAM)")
    st.info("Generando mapa de atención…")

    heat = score_cam_fast(
        model=model,
        pil_img=img,
        sex_token=sex_token,
        device=device,
        top_k=32
    )

    overlay = overlay_heatmap(img.convert("RGB"), heat)

    st.image(overlay, caption="Mapa de atención", width=500)

    # ------------------------------
    # GUARDAR HISTORIAL
    # ------------------------------
    thumb = img.copy()
    thumb.thumbnail((150, 150))

    st.session_state.history.append({
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sexo": sexo,
        "edad_meses": round(y_cal, 1),
        "categoria": categoria,
        "img_small": thumb
    })