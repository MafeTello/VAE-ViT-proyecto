# 🦴 BAE-ViT – Bone Age Estimation (Regresión en Imágenes)

## 📋 Información del Proyecto

- **Artículo Base:** *BAE-ViT: An Efficient Multimodal Vision Transformer for Bone Age Estimation*  
- **Año:** 2024  
- **Repositorio Original:** https://github.com/uw-rad-mitrp/BAE_ViT  
- **Paper:** Tomography Journal  
- **Dataset:** RSNA Pediatric Bone Age Challenge 2017  

---
## 🎯 Descripción del Modelo
BAE-ViT es un modelo multimodal basado en Vision Transformer diseñado para estimar la edad ósea a partir de radiografías de mano. Su principal innovación es la fusión temprana del token de sexo dentro del mecanismo de atención del Transformer, permitiendo interacciones ricas entre características visuales y datos clínicos.

- **Entrada:** Radiografía de mano + sexo (F/M)  
- **Salida:** Edad ósea en meses  
- **Innovación clave:** Token de sexo fusionado en el mecanismo Q/K/V desde las primeras capas  

###  Principales Innovaciones

- 🔹 **Fusión Temprana de Tokens:** Integración del sexo como token adicional desde las primeras capas  
- 🔹 **Arquitectura Híbrida Eficiente:** Combina MBConv blocks con capas Transformer  
- 🔹 **Shifted Window Attention:** Atención eficiente con ventanas desplazadas  
- 🔹 **Multimodalidad Nativa:** Procesamiento conjunto de imagen y datos clínicos  
- 🔹 **Mecanismo Q/K/V Mejorado:** El token de sexo participa en todas las capas de atención  

---
## 🏗️ Resumen Teórico de la Arquitectura
## Flujo de la Arquitectura:

Patch Embedding → MBConv Blocks → Patch Merging → Transformer Layers (Window + Shifted Window) → Global Pooling → Linear Regression

### Mecanismo de Atención (Q/K/V)
El modelo utiliza atención multi-cabeza donde los tokens de imagen y sexo se proyectan en tres espacios:

- **Q (Query):** Qué busca cada token  
- **K (Key):** Qué ofrece cada token  
- **V (Value):** Información que aporta cada token  


Fórmula de Atención:

Atención = softmax(Q·Kᵀ/√d_k)·V

**Característica única:**El token de sexo se mezcla con los tokens de imagen en cada capa de atención, permitiendo interacciones complejas entre características visuales y biológicas desde las primeras etapas del procesamiento.

## 🚀 Pasos para Ejecutar el Proyecto
### Prerrequisitos
- Docker

- Docker Compose

### Instalación y Despliegue 🐳

# Construir y levantar el contenedor
docker compose up --build -d


## 📁Uso de la Aplicación
- Acceder a la interfaz: Abrir http://localhost:7860 en el navegador

- Cargar radiografía: Subir una imagen de radiografía de mano (formatos: PNG, JPG, JPEG)

- Seleccionar sexo: Elegir entre Femenino (F) o Masculino (M)

- Ejecutar inferencia: Hacer clic en "Estimar Edad Ósea"

### Visualizar resultados:

- Edad ósea estimada en meses

- Nivel de confianza de la predicción- Ubicada en el panel de resumen

## 🧩Características de la Interfaz:
- **Multi-Crop:** Generación automática de múltiples recortes para mayor robustez

- **Procesamiento en tiempo real:** Inferencia en menos de 2 segundos por imagen

- **Interfaz intuitiva:** Diseño optimizado para uso clínico

## 📦🔧 Carga de Pesos e Inferencia
Carga de Pesos Preentrenados
Los pesos se cargan automáticamente al iniciar la aplicación desde:


- **models/baevit-ckpt_epoch_299.pth**

### Proceso de carga:

- Inicialización del Modelo: Se crea la arquitectura BAE-ViT según configs/baevit.yaml

- Carga de Pesos: Se cargan los parámetros preentrenados usando PyTorch

- Configuración de Dispositivo: Detección automática de GPU/CPU

- **Modo Evaluación:** El modelo se establece en modo evaluación (eval())

### Proceso de Inferencia
#### Preprocesamiento:

- Normalización de valores de píxel (media: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

- Redimensionamiento a 224×224 píxeles

- Conversión BGR a RGB

- Multi-Crop Processing:

- Generación de 5 crops diferentes de la imagen

- Aumentación de robustez mediante variaciones de escala y posición

**Forward Pass:*

- División de imagen en parches de 4×4 píxeles

- Embedding de parches + token de sexo

- Procesamiento through MBConv blocks y Transformer layers

- Mecanismo de atención con ventanas y ventanas desplazadas

**Post-procesamiento:**


- Conversión a edad en meses

- Cálculo de métricas de confianza

## Interpretabilidad:


- Visualización de regiones anatómicas relevantes
- Superposición de heatmaps en imagen original

##  Resultados e Interpretación
### 📈Métricas de Rendimiento:
- Error Absoluto Medio (MAE): 4.1 meses (configuración Multi-Crop)

- Tiempo de Inferencia: < 2 segundos por imagen

- Consistencia Clínica: Coincidencia con criterios de Greulich & Pyle

- Score-CAM (Class Activation Mapping):
- Los mapas de activación Score-CAM permiten visualizar las regiones anatómicas que más contribuyen a la predicción:

- Articulaciones interfalángicas y metacarpofalángicas

- Zonas de osificación del radio y cúbito

- Placas de crecimiento en falanges y metacarpianos

📁 Estructura del Repositorio
VAE-ViT-proyecto/
├── configs/
│   └── baevit.yaml
├── docker/
│   └── Dockerfile
├── models/
│   ├── baevit.py
│   ├── build.py
│   ├── model_zoo.py
│   └── baevit-ckpt_epoch_299.pth
├── src/
│   ├── model_loader.py
│   ├── inference.py
│   ├── preprocess.py
│   ├── score_cam.py
│   ├── calibrator.py
│   └── utils.py
├── test_images/
├── app.py
├── docker-compose.yml
├── requirements.txt
└── README.md


#### 📄 Licencia y Créditos
- Desarrollado por:

- **María Fernanda Tello Vergara**

- **Ana Cristina Quintero Carpintero**

- **Universidad: Universidad Autónoma de Occidente**
- **Asignatura: Analítica de Datos - Semestre 7**
- **Año: 2025**

- Basado en:
Zhang, J., Chen, W., Joshi, T., et al. "BAE-ViT: An Efficient Multimodal Vision Transformer for Bone Age Estimation." Tomography, 2024.

- Licencia: Proyecto académico para fines educativos