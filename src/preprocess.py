# src/preprocess.py
# Preprocesamiento de radiografías (PNG/JPG) para BAE-ViT (RSNA, 512×512)
"""
En este archivo se inicia el preprocesamento de datos de entrada:
Este módulo transforma las imágenes de radiografía al formato requerido por BAE-ViT:
En el cual se realiza la conversión a escala de grises (1 canal, donde se redimensionamiento a 512×512 píxeles
y se aplica la normalización con estadísticas del dataset RSNA;finalmente se realiza la transformación a tensor de PyTorch
"""

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

# Tamaño de entrada coherente con el checkpoint RSNA
IMG_SIZE = 512

# Estadísticas del dataset (1 canal) tomadas de la config
MEAN = [0.18819728989576712]
STD  = [0.03907734117789998]


def build_transforms() -> T.Compose:
    """
     
    PIPELINE DE PREPROCESAMIENTO:
    Define la secuencia de transformaciones para preparar la imagen:
    - Grayscale: Convierte a 1 canal (radiografía)
    - Resize: Estandariza tamaño a 512×512
    - ToTensor: Convierte a tensor y normaliza a [0,1]
    - Normalize: Aplica mean/std del dataset RSNA
    """
    
    return T.Compose([
        T.Grayscale(num_output_channels=1),#radiografia 1 canal
        T.Resize((IMG_SIZE, IMG_SIZE)), #tamaño fijo para el modelo
        T.ToTensor(), #tensor 
        T.Normalize(mean=MEAN, std=STD), #normalizacion
    ])


def to_tensor(img: Image.Image) -> torch.Tensor:
    """
      CONVERSIÓN A TENSOR:
    - Aplica pipeline de transformaciones
    - Añade dimensión de batch [1, 1, H, W]
    - Retorna tensor listo para inferencia
    """
    tf = build_transforms()
    x = tf(img).unsqueeze(0)
    return x


# ---------- (Opcional) utilidades extra ----------

def load_image(path_or_file) -> Image.Image:
    """
    CARGA DE IMAGEN:
    - Abre archivo PNG/JPG
    - Convierte a escala de grises (8-bit)
    - Retorna objeto PIL Image
    """
    return Image.open(path_or_file).convert("L")


def normalize_heat(h: np.ndarray) -> np.ndarray:
    """
    NORMALIZACIÓN DE MAPAS DE CALOR:
    - Escala valores a rango [0,1]
    - Útil para visualización de Score-CAM
    """
    h = h.astype(np.float32)
    h -= h.min()
    mx = h.max()
    if mx > 0:
        h /= mx
    return h