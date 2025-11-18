"""
UTILIDADES PARA VISUALIZACIÓN Y PROCESAMIENTO:
Funciones auxiliares para manejo de imágenes y mapas de calor
en la aplicación Streamlit de BAE-ViT.
"""

# src/utils.py
import numpy as np
from PIL import Image
import matplotlib.cm as cm
"""
    NORMALIZACIÓN DE MAPAS DE CALOR:
    Escala valores arbitrarios a rango [0,1] para visualización.
    
    Proceso:
    1. Resta el mínimo (centra en 0)
    2. Divide por el máximo (escala a [0,1])
    3. Maneja caso de máximo cero (evita división por cero)
    
    Args:
        h: Array numpy con valores de heatmap crudos
        
    Returns:
        h_normalized: Array numpy con valores en [0,1]
"""
def normalize_heat(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float32)
    h -= h.min()
    mx = h.max()
    if mx > 0:
        h /= mx
    return h

def overlay_heatmap(img_pil: Image.Image, heat: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """
    
    SUPERPOSICIÓN DE HEATMAP EN IMAGEN ORIGINAL:
    Combina imagen de radiografía con mapa de calor Score-CAM
    para visualización interpretativa.
    
    Proceso:
    1. Normaliza heatmap a [0,1]
    2. Aplica colormap "jet" (azul → rojo)
    3. Redimensiona heatmap al tamaño de la imagen
    4. Mezcla con transparencia controlada por alpha
    
    Args:
        img_pil: Imagen PIL original (radiografía)
        heat: Array numpy con heatmap Score-CAM
        alpha: Transparencia del heatmap (0.0-1.0)
        
    Returns:
        combined: Imagen PIL con heatmap superpuesto
    
    """
    heat_norm = normalize_heat(heat)
    cmap = cm.get_cmap("jet")
    heat_rgba = (cmap(heat_norm) * 255).astype(np.uint8)

    heat_img = Image.fromarray(heat_rgba).resize(
        img_pil.size, Image.BILINEAR
    ).convert("RGBA")

    base = img_pil.convert("RGBA")
    heat_mask = heat_img.split()[3].point(lambda x: int(x * alpha))
    heat_img.putalpha(heat_mask)

    return Image.alpha_composite(base, heat_img)
