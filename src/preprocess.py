# src/preprocess.py
# Preprocesamiento de imágenes (PNG/JPG) y opcional DICOM

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from typing import Tuple, Union

IMG_SIZE = 224  # el paper usa 224x224

def build_transforms() -> T.Compose:
    # Ajusta mean/std si el repo usa otras estadísticas
    return T.Compose([
        T.Grayscale(num_output_channels=1),     # radiografía en 1 canal
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

def load_image(file) -> Image.Image:
    """
    Carga PNG/JPG en PIL.Image.
    Si quieres DICOM, conviértelo a 8-bit antes (ver función opcional abajo).
    """
    img = Image.open(file).convert("L")
    return img

def to_tensor(img: Image.Image) -> torch.Tensor:
    tf = build_transforms()
    x = tf(img).unsqueeze(0)  # [1,1,224,224]
    return x

# ====== Opcional: DICOM a PIL ======
def dicom_to_pil(dcm) -> Image.Image:
    """
    Convierte un DICOM (pydicom) a PIL Image 8-bit.
    Úsalo solo si decides aceptar DICOM en la UI.
    """
    arr = dcm.pixel_array.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)