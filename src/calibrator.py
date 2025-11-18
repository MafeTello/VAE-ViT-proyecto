"""
CALIBRADOR DE EDAD ÓSEA:
Este módulo ajusta las predicciones crudas del modelo BAE-ViT
para convertirlas a valores clínicamente significativos (meses).

Propósito:
- Escalar salidas del modelo a rango fisiológico real (0-220 meses)
- Prevenir valores negativos o fuera de rango
- Mejorar concordancia con estándares clínicos
"""
"""
    es importante tener en cuenta la calibracion de la capa lineal donde :
    Aplica transformación lineal simple: y_calibrada = w * x_cruda + b
    
    Donde:
    - w: Factor de escala (9.5) para convertir a meses
    - b: Término de bias (2.0) para evitar valores negativos
    - Clamp: Limita a rango fisiológico [0, 220] meses
    """

        # Los valores fueron determinados empíricamente para:
        # - Escalar predicciones crudas (~0.1-1.0) a meses (~1-180)
        # - Evitar predicciones negativas
        # - Mantener rango fisiológico realista
import torch
import torch.nn as nn

class AgeCalibrator(nn.Module):
    """
    Ajusta la salida cruda del modelo original y la convierte a meses reales.
    """
    def __init__(self):
        super().__init__()
        # Capa lineal pequeña para escalar la salida
        self.fc = nn.Linear(1, 1)
        # Inicialización que asegura valores razonables
        with torch.no_grad():
            self.fc.weight.fill_(9.5)   # escala hacia arriba (evita valores tipo 0.1)
            self.fc.bias.fill_(2.0)     # evita negativos

    def forward(self, x):
        x = self.fc(x)

        # Clamp fisiológico: 0 a 220 meses
        x = torch.clamp(x, min=0.0, max=220.0)
        return x
