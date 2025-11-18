# src/inference.py
# Fast-ScoreCAM para RSNA-BAEViT (sin modificar el modelo original)
# --------------------------------------------------------------

from typing import Literal, Optional, List
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from model_loader import build_model, load_weights
from preprocess import to_tensor, IMG_SIZE

Sex = Literal["F", "M"]


class BAEVitInferencer:
    """
    Inferencia + Fast-ScoreCAM para RSNA-BAEViT.
    Hook en la última capa (antes del mean pooling).
    """
    def __init__(self, ckpt_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.device)
        self.model.to(self.device).eval()

        # Aquí guardaremos activaciones espaciales [B, S, S, C]
        self._acts = None
        self._register_hook()

        if ckpt_path:
            load_weights(self.model, ckpt_path)

    # ---------------------------------------------------------
    # Hook para capturar activaciones antes del pooling final
    # ---------------------------------------------------------
    def _register_hook(self):
        """
        En RSNA-BAEViT la salida espacial final viene de:
        model.model.layers[-1]
        
        Esa capa recibe x = (tokens, gender_embed)
        Retorna (tokens, gender)
        
        tokens tiene shape [B, 16*16, C]
        """
        last_layer = self.model.model.layers[-1]

        def hook_fn(module, input, output):
            tokens, _ = output  # tokens=[B,256,C]
            B, L, C = tokens.shape
            S = int(L ** 0.5)  # 16×16
            spatial = tokens.reshape(B, S, S, C)
            self._acts = spatial.detach()  # guardado para ScoreCAM

        last_layer.register_forward_hook(hook_fn)

    # ---------------------------------------------------------
    # Token de sexo
    # ---------------------------------------------------------
    @staticmethod
    def _sex_token(sex: Sex, device: str, batch: int = 1) -> torch.Tensor:
        v = 1 if str(sex).upper().startswith("M") else 0
        return torch.full((batch, 1), v, dtype=torch.long, device=device)

    # ---------------------------------------------------------
    # Predicción usual
    # ---------------------------------------------------------
    @torch.no_grad()
    def predict(self, img: Image.Image, sex: Sex = "F", multi_crop: bool = False) -> float:
        self.model.eval()
        x = to_tensor(img).to(self.device)
        stoken = self._sex_token(sex, self.device)

        y = self.model(x, stoken)
        return float(y.item())

    # ---------------------------------------------------------
    # Fast-ScoreCAM
    # ---------------------------------------------------------
    @torch.no_grad()
    def fast_scorecam(self, img: Image.Image, sex: Sex = "F") -> np.ndarray:
        """
        Genera heatmap [H,W] usando pesos canal-wise (softmax)
        sobre activaciones espaciales de la última capa.
        """
        self._acts = None

        # 1) Forward para llenar self._acts
        x = to_tensor(img).to(self.device)
        stoken = self._sex_token(sex, self.device)
        _ = self.model(x, stoken)

        if self._acts is None:
            raise RuntimeError("Activaciones no capturadas. Verifica hook.")

        # self._acts: [1,16,16, C]
        A = self._acts.squeeze(0)  # [16,16,C]
        Hs, Ws, C = A.shape

        # 2) Extraemos la predicción para calcular los pesos por canal
        pred = float(self.model(x, stoken).item())

        # 3) Peso por canal usando softmax en la suma espacial por canal
        #    ideal para Fast-ScoreCAM
        channel_sums = A.reshape(-1, C).sum(dim=0)  # [C]
        weights = torch.softmax(channel_sums, dim=0)  # [C]

        # 4) Heatmap combinado: Σ w_c * A[:,:,c]
        weighted = (A * weights)  # broadcasting
        heat = weighted.sum(dim=2).cpu().numpy()  # [16,16]

        # 5) Normalizar y reescalar al tamaño de la imagen original
        heat = np.maximum(heat, 0)
        heat /= (heat.max() + 1e-8)

        heat_t = torch.tensor(heat).unsqueeze(0).unsqueeze(0)
        heat_up = F.interpolate(
            heat_t,
            size=(img.size[1], img.size[0]),
            mode="bilinear",
            align_corners=False
        )[0, 0].cpu().numpy()

        return heat_up
