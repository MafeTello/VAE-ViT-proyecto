# models/model_zoo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.registry import register_model
from .baevit import RSNA_BAEViT

# ======================================================
# 1Registrar modelo RSNA_BAEViT dentro de timm
# ======================================================
@register_model
def rsna_baevit(
    pretrained=False,
    num_classes=1000,
    drop_path_rate=0.2,
    layer_lr_decay=1.0,
    img_size=512,
    in_chans=1,
    **kwargs
):
    return RSNA_BAEViT(
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=drop_path_rate,
        layer_lr_decay=layer_lr_decay,
    )


# ======================================================
# Wrapper general para modelos de regresión
# ======================================================
class TimmRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        feature_dim: int = 576,
        is_sigmoid: bool = False,
        img_size: int = 512,
        config=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.is_sigmoid = is_sigmoid
        self.img_size = img_size

        # Crear modelo base (ya registrado)
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            in_chans=1,
        )

        # Capa lineal de regresión
        self.linear = nn.Linear(feature_dim, 1)
        self.sigmoid = nn.Sigmoid() if is_sigmoid else None

    # ==================================================
    # forward con soporte para token de sexo
    # ==================================================
    def forward(self, x, sex_token=None):
        device = x.device
        B = x.shape[0]

        # Normalizar token de sexo
        if sex_token is None:
            sex_token = torch.zeros(B, 1, dtype=torch.long, device=device)
        else:
            sex_token = sex_token.to(device)
            if sex_token.dim() == 1:
                sex_token = sex_token.unsqueeze(1)

        # Forward: (imagen, sexo)
        feats = self.model((x, sex_token))
        out = self.linear(feats)
        if self.sigmoid:
            out = self.sigmoid(out)
        return out.squeeze(-1)  # [B]

    # ==================================================
    # Carga de pesos preentrenados
    # ==================================================
    def load_pretrained_model(self, pretrained_path: str):
        ckpt = torch.load(pretrained_path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt, strict=False)
        print(f"[OK] Pesos cargados desde {pretrained_path}")
