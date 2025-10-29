# src/utils.py
# Utilidades para overlays/heatmaps. Score-CAM aquí es un placeholder
# para que luego enganches a la última capa visual del encoder.

import numpy as np
from PIL import Image

def overlay_heatmap(img_pil: Image.Image, heat: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Superpone un heatmap [H,W] normalizado (0..1) sobre la imagen en color falso.
    """
    heat_rgb = np.uint8(255 * heat)
    heat_rgb = Image.fromarray(heat_rgb, mode="L").resize(img_pil.size, Image.BILINEAR).convert("RGBA")
    # mapear a colormap simple (rojo)
    r = np.array(heat_rgb)
    rgba = np.zeros((*r.shape[:2], 4), dtype=np.uint8)
    rgba[..., 0] = r[..., 0]         # canal R
    rgba[..., 3] = (r[..., 0] * alpha).astype(np.uint8)  # alpha
    heat_rgba = Image.fromarray(rgba, mode="RGBA")
    base = img_pil.convert("RGBA")
    return Image.alpha_composite(base, heat_rgba)

def normalize_heat(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float32)
    h -= h.min()
    mx = h.max()
    if mx > 0:
        h /= mx
    return h