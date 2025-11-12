# BAE-ViT Demo — Regresión en Imágenes (Edad Ósea)

**Paper base:** BAE-ViT (Tomography, 2024).  
**Repo original:** https://github.com/uw-rad-mitrp/BAE_ViT

## Qué hace
Dado una radiografía de mano + sexo (F/M), predice **edad ósea en meses** con un **Vision Transformer** que
integra el sexo como **token**, permitiendo que interactúe con los parches de la imagen vía **atención (Q/K/V)**.

## Estructura



Descripción del modelo y sus principales innovaciones.
• Resumen teórico de la arquitectura.
• Pasos para ejecutar el proyecto (instalación, despliegue y uso).
• Explicación de cómo se cargan los pesos y cómo se realiza la inferencia.






Título: BAE-ViT – Bone Age Estimation (Regresión en Imágenes).

Paper base + repo original: nombre, año y enlace.

Descripción breve: Entrada (RX + sexo), salida (meses), fusión temprana de sexo como token.

Arquitectura (resumen): Patch Embedding → MBConv → Patch Merging → Transformer (ventanas + shifted) → Pooling → Linear.

Q/K/V (2–3 líneas):

Tokens → proyecciones Q, K, V; atención = softmax(QKᵀ/√d); el token de sexo se mezcla con tokens de imagen en cada capa.

Cómo correr:

docker compose up --build -d → abre http://localhost:7860.

Pesos preentrenados: models/baevit-ckpt_epoch_299.pth (se cargan al iniciar la app).

Inferencia: explica campos de la UI y el multi-crop.

Resultados/interpretación: menciona Score-CAM (si no lo conectas aún, explica el concepto).

Estructura del repo: el árbol de carpetas.

Licencia y créditos.