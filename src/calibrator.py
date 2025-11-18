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
