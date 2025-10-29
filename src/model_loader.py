# src/model_loader.py
# Carga del modelo BAE-ViT y de los pesos
# Ajusta los TODO a la implementación exacta del repo uw-rad-mitrp/BAE_ViT

import torch
import torch.nn as nn
from typing import Optional

# === TODO 1: importa la clase/modelo real desde el repo ===
# Ejemplos (elige el que aplique):
# from models.bae_vit import BAEViT
# from bae_vit.model import BAEViT
# from tinyvit import tiny_vit_bae as BAEViT

class DummyBAEViT(nn.Module):
    """
    Modelo dummy para que el proyecto sea ejecutable antes de conectar el modelo real.
    Devuelve un escalar ~120.0 meses (10 años), solo para probar la UI.
    Reemplázalo por la clase real del repo y elimina esta clase.
    """
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.head.weight[:] = 120.0  # valor fijo

    def forward(self, x, sex_token: Optional[torch.Tensor] = None):
        b = x.shape[0]
        one = torch.ones(b, 1, device=x.device, dtype=x.dtype)
        y = self.head(one).squeeze(1)  # [B]
        return y


def build_model(device: Optional[str] = None) -> nn.Module:
    """
    Construye la arquitectura del modelo.
    Devuelve el modelo en modo eval y movido a device.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # === TODO 2: reemplaza DummyBAEViT() por tu clase real (ej. BAEViT(...)) ===
    # model = BAEViT(img_size=224, use_sex_token=True, ...)
    model = DummyBAEViT()

    model.eval().to(device)
    return model


def load_weights(model: nn.Module, ckpt_path: str) -> None:
    """
    Carga pesos desde un checkpoint .pth / .pt / TorchScript si aplica.
    Maneja map_location automáticamente.
    """
    map_loc = next(model.parameters()).device
    try:
        state = torch.load(ckpt_path, map_location=map_loc)
        # Si el checkpoint tiene 'state_dict', úsalo; si no, intenta cargar directo.
        if isinstance(state, dict) and "state_dict" in state:
            state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
            model.load_state_dict(state, strict=False)
        elif isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
        else:
            # TorchScript (rare) → sugerimos no usar aquí.
            raise RuntimeError("Formato de checkpoint no soportado automáticamente.")
    except Exception as e:
        print(f"[WARN] No se pudieron cargar pesos desde {ckpt_path}: {e}\n"
              f"→ Revisa la ruta y el formato. Ejecutando con pesos por defecto (solo demo).")