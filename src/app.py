# src/app.py (esto es para el deploy en streamlit)
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
from inference import BAEVitInferencer
from utils import overlay_heatmap, normalize_heat

st.set_page_config(page_title="BAE-ViT — Bone Age Estimation", layout="centered")

st.title("🦴 BAE-ViT — Estimación de Edad Ósea (Regresión con Transformers)")
st.caption("Sube una radiografía de mano (PNG/JPG), selecciona sexo, y obtén la edad ósea en meses.")

ckpt_path = "models/baevit-ckpt_epoch_299.pth"  # ← TODO: ajusta al nombre real del checkpoint
if "infer" not in st.session_state:
    st.session_state.infer = BAEVitInferencer(ckpt_path=ckpt_path)

col1, col2 = st.columns([2,1])
with col1:
    up = st.file_uploader("Imagen de entrada", type=["png", "jpg", "jpeg"])
with col2:
    sex = st.selectbox("Sexo", ["F", "M"])
multi = st.toggle("Multi-crop (robustez)", value=True)

btn = st.button("Predecir edad ósea")

if btn and up is not None:
    img = Image.open(up).convert("L")
    st.image(img, caption="Radiografía cargada", use_container_width=True)
    y = st.session_state.infer.predict(img, sex=sex, multi_crop=multi)
    st.success(f"Edad ósea estimada: **{y:.1f} meses**")

    # Placeholder de heatmap para que la UI muestre algo.
    # Conecta tu Score-CAM real cuando enganches el modelo definitivo.
    fake_heat = normalize_heat(np.random.rand(224, 224))
    st.image(overlay_heatmap(img.convert("RGB"), fake_heat),
             caption="Mapa de atención (placeholder)", use_container_width=True)

elif btn and up is None:
    st.warning("Sube una imagen primero 🙌")
