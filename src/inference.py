# src/inference.py
# Inferencia: imagen + sexo → meses (float)

from typing import Literal, Optional, List
from PIL import Image

import torch
import torch.nn.functional as F

# IMPORTES ABSOLUTOS (sin punto) para evitar "attempted relative import"
from model_loader import build_model, load_weights
from preprocess import to_tensor, IMG_SIZE

Sex = Literal["F", "M"]


class BAEVitInferencer:
    """
    Wrapper de inferencia para BAE-ViT.
    - Carga (o construye) el modelo
    - Prepara el token de sexo
    - Soporta TTA sencillo (multi-crop) con mediana
    """
    def __init__(self, ckpt_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # build_model puede aceptar 'device' o un objeto config; aquí usamos device
        self.model = build_model(self.device)
        self.model.to(self.device).eval()
        if ckpt_path:
            load_weights(self.model, ckpt_path)

    @staticmethod
    def _sex_token(sex: Sex, device: str, batch: int = 1) -> torch.Tensor:
        """
        Convierte F/M en entero 0/1 y lo devuelve como tensor [B,1] (long).
        El backbone hará el embedding interno.
        """
        v = 1 if str(sex).upper().startswith("M") else 0
        return torch.full((batch, 1), v, dtype=torch.long, device=device)

    def _tta_multicrop(self, x: torch.Tensor, scales=(1.0, 0.9, 0.8)) -> List[torch.Tensor]:
        """
        Test-time augmentation muy simple: escalados centrados con padding
        para volver a IMG_SIZE. Devuelve una lista de tensores [1,1,H,W].
        """
        outs = []
        for s in scales:
            if s == 1.0:
                outs.append(x)
                continue
            sz = int(IMG_SIZE * s)
            xr = F.interpolate(x, size=(sz, sz), mode="bilinear", align_corners=False)
            pad = IMG_SIZE - sz
            left = pad // 2
            right = pad - left
            outs.append(F.pad(xr, (left, right, left, right)))
        return outs

    @torch.no_grad()
    def predict(self, img: Image.Image, sex: Sex = "F", multi_crop: bool = True) -> float:
        """
        Devuelve la edad ósea estimada (meses).
        """
        self.model.eval()
        x = to_tensor(img).to(self.device)                  # [1,1,512,512]
        stoken = self._sex_token(sex, self.device, 1)       # [1,1] (0=F,1=M)

        if not multi_crop:
            y = self.model(x, stoken)                       # -> [1]
            return float(y.squeeze(0).item())

        preds = []
        for c in self._tta_multicrop(x):
            y = self.model(c, stoken)
            preds.append(float(y.squeeze(0).item()))

        # mediana: robusta ante outliers
        preds.sort()
        mid = len(preds) // 2
        return preds[mid] if len(preds) % 2 else 0.5 * (preds[mid - 1] + preds[mid])