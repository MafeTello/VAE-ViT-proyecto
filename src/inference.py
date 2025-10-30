# src/inference.py
# Lógica de inferencia: imagen + token de sexo → predicción (meses)
import torch
import torch.nn.functional as F
from typing import Literal, Optional, List
from PIL import Image
from .model_loader import build_model, load_weights
from .preprocess import to_tensor, IMG_SIZE

Sex = Literal["F", "M"]

class BAEVitInferencer:
    def __init__(self, ckpt_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.device)
        if ckpt_path:
            load_weights(self.model, ckpt_path)

    @staticmethod
    def sex_token(sex: Sex, device: str):
        """
        Convierte F/M a token. En el modelo real será un embedding/linear dentro del forward.
        Aquí solo entregamos un entero para conservar la firma.
        """
        v = 1 if str(sex).upper().startswith("M") else 0
        return torch.tensor([v], dtype=torch.long, device=device)

    def _tta_multicrop(self, x: torch.Tensor, scales=(1.0, 0.9, 0.8)) -> List[torch.Tensor]:
        """
        Genera crops escalados centrados (TTA simple).
        El paper reporta mejoras con 'multi-crop test'.
        """
        outs = []
        for s in scales:
            if s == 1.0:
                xx = x
            else:
                sz = int(IMG_SIZE * s)
                xr = F.interpolate(x, size=(sz, sz), mode="bilinear", align_corners=False)
                pad = IMG_SIZE - sz
                # padding simétrico
                left = pad // 2
                right = pad - left
                xx = F.pad(xr, (left, right, left, right))
            outs.append(xx)
        return outs

    @torch.no_grad()
    def predict(self, img: Image.Image, sex: Sex = "F", multi_crop: bool = True) -> float:
        """
        Devuelve edad ósea en meses (float).
        """
        self.model.eval()
        x = to_tensor(img).to(self.device)  # [1,1,224,224]
        stoken = self.sex_token(sex, self.device)  # [1]

        if not multi_crop:
            y = self.model(x, stoken)  # → [B]
            return float(y.squeeze().item())

        # TTA multicrop + mediana (robustez)
        crops = self._tta_multicrop(x)
        preds = []
        for c in crops:
            y = self.model(c, stoken)
            preds.append(float(y.squeeze().item()))
        # Mediana robustece frente a outliers
        preds = sorted(preds)
        mid = len(preds) // 2
        if len(preds) % 2 == 1:
            return preds[mid]
        return 0.5 * (preds[mid - 1] + preds[mid])