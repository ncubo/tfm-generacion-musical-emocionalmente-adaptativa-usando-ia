# Backend - TFM Generación Musical Emocional

Backend del sistema de generación musical adaptativa basada en reconocimiento emocional mediante IA.

## 📋 Estructura del Proyecto

```
backend/
├── requirements.txt          # Dependencias del proyecto
├── src/
│   ├── __init__.py
│   ├── app.py               # Aplicación principal (placeholder)
│   └── core/
│       ├── __init__.py
│       ├── camera/
│       │   ├── __init__.py
│       │   └── webcam.py    # Módulo de captura de webcam
│       └── emotion/
│           ├── __init__.py
│           └── deepface_detector.py  # Detector emocional con DeepFace
├── scripts/
│   └── run_webcam_demo.py   # Demo de webcam + reconocimiento emocional
└── .gitignore
```

## 🚀 Instalación

### 1. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En macOS/Linux
# o
venv\Scripts\activate  # En Windows
```

### 2. Instalar dependencias

```bash
cd backend
pip install -r requirements.txt
```

## 🎥 Demo de Webcam con Reconocimiento Emocional

Para probar la captura de webcam con detección de emociones en tiempo real:

```bash
# Desde la raíz del proyecto
python backend/scripts/run_webcam_demo.py
```

**Controles:**
- Presiona `q` para salir del demo

**Nota:** La primera ejecución puede tardar más tiempo ya que DeepFace descarga los modelos preentrenados automáticamente.
- Presiona `q` para salir del demo

## 🧪 Verificar instalación

```bash
python backend/src/app.py
```

Debería mostrar información sobre la aplicación sin errores.

## 📦 Dependencias Actuales
- **deepface**: Reconocimiento emocional facial con modelos preentrenados

- **opencv-python**: Captura y procesamiento de video
- **numpy**: Manejo de arrays y frames
 los Módulos

### WebcamCapture

```python
from core.camera import WebcamCapture

# Opción 1: Uso manual
webcam = WebcamCapture(camera_index=0)
webcam.start()

success, frame = webcam.read()
if success:
    # Procesar frame...
    pass

webcam.release()

# Opción 2: Uso con context manager (recomendado)
with WebcamCapture() as webcam:
    success, frame = webcam.read()
    if success:
        # Procesar frame...
        pass
```

###x] ~~Implementar captura de webcam~~
- [x] ~~Integrar modelo de reconocimiento emocional (DeepFace)~~
- [ ] Implementar API Flask
```python
from core.emotion import DeepFaceEmotionDetector

# Crear detector
detector = DeepFaceEmotionDetector(enforce_detection=False)

# Predecir emoción en un frame
result = detector.predict(frame)

print(f"Emoción: {result['emotion']}")
print(f"Rostro detectado: {result['face_detected']}")
print(f"Probabilidades: {result['probabilities']}")

# Obtener etiqueta en español
emotion_es = detector.get_emotion_label_spanish(result['emotion'])
```

**Emociones soportadas:**
- `angry` (enfadado)
- `disgust` (asco)
- `fear` (miedo)
- `happy` (feliz)
- `sad` (triste)
- `surprise` (sorpresa)
- `neutral` (neutral)     pass
```

## 📝 Próximos Pasos

- [ ] Implementar API Flask
- [ ] Integrar modelo de reconocimiento emocional
- [ ] Implementar sistema de generación musical
- [ ] Conectar con frontend
- [ ] Añadir tests unitarios

## 👨‍💻 Desarrollo

Este proyecto forma parte de un TFM (Trabajo Fin de Máster) en Inteligencia Artificial.

**Versión actual:** 0.1.0
