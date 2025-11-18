# src/score_cam.py
"""
score_cam_fast: versión rápida para RSNA_BAEViT
- Engancha la salida espacial del último layer (model.model.layers[-1])
- Calcula energía por canal y usa softmax como pesos (no hace re-forwards)
- top_k: si se especifica, usa solo los top_k canales (más rápido)
Retorna heatmap numpy (H_img, W_img) con valores en [0,1].
"""
"""

Este módulo genera mapas de calor que muestran las regiones de la imagen 
que más contribuyen a la predicción de edad ósea.

Método Fast-ScoreCAM optimizado para BAE-ViT:
- Captura activaciones de la última capa del transformer
- Calcula pesos por canal usando energía y softmax
- Combina activaciones ponderadas sin re-ejecutar el modelo
- Genera heatmaps de alta resolución para interpretación clínica
"""
"""
    GENERACIÓN RÁPIDA DE MAPAS SCORE-CAM:
    Proceso optimizado que evita múltiples forward passes:
    1. Single forward pass para capturar activaciones
    2. Cálculo de energía por canal como medida de importancia
    3. Selección opcional de top-K canales más relevantes
    4. Combinación lineal ponderada de activaciones
    5. Interpolación a tamaño original de la imagen
    
    Args:
        model: Modelo BAE-ViT para inferencia
        pil_img: Imagen PIL en escala de grises o RGB
        sex_token: Token de sexo (0=F, 1=M) para multimodalidad
        device: Dispositivo de ejecución (auto-detecta GPU/CPU)
        top_k: Número de canales a considerar (None = todos)
    
    Returns:
        heatmap: Array numpy [H_orig, W_orig] con valores normalizados [0,1]
    """
from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
from src.preprocess import to_tensor, IMG_SIZE
from PIL import Image

def _register_acts_hook(model, storage):
    """Registra hook en el último layer para capturar tokens espaciales."""
    last_layer = model.model.layers[-1]

    def hook_fn(module, inp, out):
        # out is (tokens, gender) per model implementation
        try:
            tokens, _ = out
        except Exception:
            tokens = out
        # tokens: [B, L, C]
        B, L, C = tokens.shape
        S = int((L) ** 0.5)
        # reshape a [B, S, S, C]
        spatial = tokens.reshape(B, S, S, C)
        storage['acts'] = spatial.detach().cpu()

    handle = last_layer.register_forward_hook(hook_fn)
    return handle

def score_cam_fast(model,
                   pil_img: Image.Image,
                   sex_token,
                   device: Optional[str] = None,
                   top_k: Optional[int] = None) -> np.ndarray:
    """
    Genera heatmap rápido.
    - model: TimmRegressor (tu wrapper)
    - pil_img: PIL.Image (modo L o RGB)
    - sex_token: int (0/1) o torch.Tensor
    - device: 'cpu' o 'cuda' (si None, se detecta)
    - top_k: int or None (si int -> usa solo top_k canales)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # normalizar sex_token a tensor [1,1]
    if isinstance(sex_token, (int, float)):
        stoken = torch.tensor([[int(sex_token)]], dtype=torch.long, device=device)
    elif isinstance(sex_token, torch.Tensor):
        stoken = sex_token.to(device)
        if stoken.dim() == 1:
            stoken = stoken.unsqueeze(1)
    else:
        v = 1 if str(sex_token).upper().startswith("M") else 0
        stoken = torch.tensor([[v]], dtype=torch.long, device=device)

    # register hook
    storage = {'acts': None}
    hook = _register_acts_hook(model, storage)

    # prepare input
    x = to_tensor(pil_img).to(device)  # [1,1,H,W]

    # forward once to capture activations
    with torch.no_grad():
        _ = model(x, stoken)

    # remove hook
    hook.remove()

    if storage['acts'] is None:
        raise RuntimeError("No se capturaron activaciones del modelo (hook falló).")

    acts = storage['acts'].squeeze(0)  # [S, S, C] CPU tensor
    S_h, S_w, C = acts.shape

    # convert to tensor on device for speed
    acts_t = acts.permute(2, 0, 1).unsqueeze(1).to(device)  # [C,1,S,S]

    # compute channel energy
    channel_energy = acts_t.abs().view(C, -1).sum(dim=1)  # [C]

    # select top_k if requested
    if top_k is not None and top_k < C:
        _, idx = torch.topk(channel_energy, k=top_k, largest=True)
        acts_sel = acts_t[idx]
    else:
        acts_sel = acts_t  # [K,1,S,S]

    K = acts_sel.shape[0]

    # compute weights: softmax over channel sums
    channel_sums = acts_sel.view(K, -1).sum(dim=1)  # [K]
    if torch.all(channel_sums == 0):
        weights = torch.ones_like(channel_sums) / float(K)
    else:
        weights = torch.softmax(channel_sums, dim=0)

    # weighted combination
    weighted = torch.zeros((S_h, S_w), dtype=torch.float32, device=device)
    for i in range(K):
        a = acts_sel[i, 0].to(device)
        w = float(weights[i].item())
        weighted += w * a

    heat = weighted.cpu().numpy()
    # relu and normalize
    heat = np.maximum(heat, 0.0)
    if heat.max() > 0:
        heat = heat / (heat.max() + 1e-8)

    # upsample to original image size
    heat_t = torch.tensor(heat).unsqueeze(0).unsqueeze(0)
    heat_up = F.interpolate(heat_t, size=(pil_img.size[1], pil_img.size[0]),
                            mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    # final normalize
    heat_up = np.clip(heat_up, 0.0, 1.0)
    if heat_up.max() > 0:
        heat_up = heat_up / (heat_up.max() + 1e-8)

    return heat_up