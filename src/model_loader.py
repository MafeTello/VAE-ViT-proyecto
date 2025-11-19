# src/model_loader.py
# Carga del modelo REAL BAE-ViT (no dummy)

#CARGA DE PESOS DEL MODELO BAE-ViT:En este modulo podemos observar que se encarga de cargar la configuración del modelo desde archivo YAML
#donde se construye la arquitectura BAE-ViT usando timm, en el cual se Carga los pesos preentrenados desde checkpoint .pth y finalmente
# Configurar el modelo en modo evaluación
# CARGA DE PESOS PRETRENADOS:En este punto se intenta múltiples estrategias de carga del checkpoint
# donde se manejan diferentes formatos de archivos .pth
# Usando helper de TimmRegressor para carga robusta
# Y Aplicando pesos al backbone del modelo

# src/model_loader.py
# Carga del modelo REAL BAE-ViT (no dummy)

import os
import yaml
import torch
import torch.nn as nn
from types import SimpleNamespace

# Importa el builder del repo (esto registra rsna_baevit en timm)
from models.build import build_model as build_baevit   # models/build.py
from models.model_zoo import TimmRegressor            # asegura que model_zoo.py exista

# Utilidad: dict -> objeto con atributos (recursivo)
def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d

def _load_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    # completa flags que usa build.py / model_zoo.py
    # valores por defecto si no aparecen en el YAML
    cfg_dict.setdefault("MODEL", {})
    cfg_dict["MODEL"].setdefault("DROP_PATH_RATE", 0.2)
    cfg_dict.setdefault("MI", {})
    cfg_dict["MI"].setdefault("LAYER_LR_DECAY", 1.0)
    cfg_dict.setdefault("FUSED_LAYERNORM", False)
    return _to_ns(cfg_dict)

def build_model(device: str | None = None) -> nn.Module:
    """
    Construye el TimmRegressor('rsna_baevit') con la config YAML.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_config(os.path.join("configs", "baevit.yaml"))

    # build_baevit retorna un TimmRegressor según el TYPE/NAME del YAML
    model = build_baevit(cfg)
    model = model.to(device).eval()
    return model

def load_weights(model: nn.Module, ckpt_path: str) -> None:
    """
    Carga pesos del checkpoint. TimmRegressor expone .model (backbone timm)
    y un helper para cargar checkpoints con llaves diversas.
    """
    map_loc = next(model.parameters()).device
    ckpt = torch.load(ckpt_path, map_location=map_loc)

    # TimmRegressor del repo trae un helper
    # Intentamos varias formas de llave:
    try:
        # 1) si el ckpt viene con {"model": ...}
        model.load_pretrained_model(ckpt_path)
        print(f"[OK] Checkpoint cargado con TimmRegressor.load_pretrained_model: {ckpt_path}")
        return
    except Exception:
        pass

    try:
        state = ckpt.get("model", ckpt)
        model.model.load_state_dict(state, strict=False)
        print(f"[OK] Pesos cargados en backbone timm desde: {ckpt_path}")
    except Exception as e:
        print(f"[WARN] No se pudieron cargar pesos desde {ckpt_path}: {e}")
        print("     → Verifica la ruta y el formato del checkpoint.")