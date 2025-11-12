# src/preprocess.py
# Preprocesamiento de radiografías (PNG/JPG) para BAE-ViT (RSNA, 512×512)

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
    Pipeline de preprocesamiento:
      - Escala de grises (1 canal)
      - Resize a 512×512
      - Tensor [0,1]
      - Normalización con MEAN/STD del dataset
    """
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convierte una PIL.Image en un tensor listo para el modelo: [1, 1, H, W]
    """
    tf = build_transforms()
    x = tf(img).unsqueeze(0)
    return x


# ---------- (Opcional) utilidades extra ----------

def load_image(path_or_file) -> Image.Image:
    """
    Abre PNG/JPG y lo devuelve como PIL (L = 8-bit grayscale).
    """
    return Image.open(path_or_file).convert("L")


def normalize_heat(h: np.ndarray) -> np.ndarray:
    """
    Normaliza mapas de calor a [0,1] (útil para overlays).
    """
    h = h.astype(np.float32)
    h -= h.min()
    mx = h.max()
    if mx > 0:
        h /= mx
    return h