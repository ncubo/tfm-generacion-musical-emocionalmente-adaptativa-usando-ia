# Backend - Sistema de Generación Musical Emocional

Backend del sistema de generación musical adaptativa basada en reconocimiento emocional mediante técnicas de Inteligencia Artificial.

## Estructura del Proyecto

```
backend/
├── requirements.txt          # Dependencias del proyecto
├── src/
│   ├── __init__.py
│   ├── app.py               # Aplicación principal (placeholder)
│   └── core/
│       ├── __init__.py
│       ├── camera/          # Captura de webcam
│       ├── emotion/         # Detección emocional facial
│       ├── va/              # Mapeo a Valence-Arousal
│       ├── music/           # Parámetros musicales y generación MIDI
│       ├── pipeline/        # Pipeline integrado
│       └── utils/           # Utilidades matemáticas
├── scripts/
│   ├── run_webcam_demo.py              # Demo de webcam + emociones
│   ├── generate_baseline_from_webcam.py # Generación MIDI desde webcam
│   └── test_midi_generation.py         # Test de generación MIDI
└── .gitignore
```

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- Webcam funcional para captura de video en tiempo real
- Permisos de sistema para acceso a la cámara

### Creación de Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate  # En macOS/Linux
# o
venv\Scripts\activate  # En Windows
```

### Instalación de Dependencias

```bash
cd backend
pip install -r requirements.txt
```

### Dependencias Principales

- **opencv-python**: Captura de video desde webcam
- **deepface**: Framework de reconocimiento emocional facial
- **mido**: Librería para generación de archivos MIDIal

Muestra la emoción detectada y las coordenadas Valence-Arousal en tiempo real:

```bash
python scripts/run_webcam_demo.py
```

Para finalizar la ejecución, presionar la tecla 'q'.

### Demostración 2: Generación de Archivos MIDI desde Captura Emocional

Captura emoción facial

**Controles:** Presiona `q` para salir

### Demo 2: Generación MIDI desde Webcam

Captura tu emoción y genera un archivo MIDI personalizado:

```bash
# Uso básico (10s de captura, 8 compases)
python scripts/generate_baseline_from_webcam.py

# Personalizado
python scripts/generate_baseline_from_webcam.py --duration 15 --bars 16 --output mi_musica.mid
```
#### Parámetros Disponibles

- `--duration SECONDS`: Duración de la captura emocional en segundos (valor por defecto: 10)
- `--bars BARS`: Número de compases a generar en el archivo MIDI (valor por defecto: 8)
- `--output PATH`: Ruta del archivo MIDI de salida (valor por defecto: output/emotion.mid)
- `--seed SEED`: Semilla aleatoria para garantizar reproducibilidad (opcional)

### Demostración 3: Generación MIDI de Prueba
### Demo 3: Test de Generación MIDI

Genera archivos MIDI de prueba sin webcam:

```bash
python scripts/test_midi_generation.py
```

Este script genera archivos MIDI para las siguientes emociones: happy, sad, angry, fear, neutral, excited, calm. No requiere webcam y sirve para validar el funcionamiento del generador MIDI.

## Guía de Uso de los Módulos

### Pipeline de Procesamiento Completo
```python
from core import EmotionPipeline, WebcamCapture, DeepFaceEmotionDetector

# Crear componentes
webcam = WebcamCapture(camera_index=0)
detector = DeepFaceEmotionDetector()

# Crear pipeline integrado
pipeline = EmotionPipeline(
    camera=webcam,
    detector=detector,
    window_size=10  # Suavizado temporal
)

# Iniciar
pipeline.start()

# Procesar frame
result = pipeline.step()
print(result)
# {'emotion': 'happy', 'valence': 0.70, 'arousal': 0.60, 'scores': {...}}

# Detener
pipeline.stop()
```

### Generación MIDI

```python
from core.music import va_to_music_params, generate_midi_baseline

# Coordenadas emocionales
valence = 0.70  # Happy
arousal = 0.60

# Convertir a parámetros musicales
params = va_to_music_params(valence, arousal)

# Generar MIDI
output_path = generate_midi_baseline(
    params=params,
    out_path='output.mid',
    length_bars=8,
    seed=42  # Opcional: para reproducibilidad
)

print(f"MIDI generado: {output_path}")
```
Arquitectura del Sistema de Mapeo Emocional

## Arquitectura del Sistema de Mapeo Emocional

### Flujo de Procesamiento: Emoción a
1. **Detección**: DeepFace detecta emoción facial
2. **Normalización**: Emoción se normaliza al conjunto estándar
3. **Mapeo VA**: Emoción → coordenadas (Valence, Arousal)
4. **Suavizado**: Media móvil temporal para estabilidad
5. **Parámetros**: VA → parámetros musicales (tempo, modo, densidad, etc.)
6. **Generación**: Parámetros → archivo MIDI

### Mapeo Emociones Básicas

| Emoción | Valence | Arousal | Tempo | Modo |
|---------|---------|---------|-------|------|
| Happy | +0.70 | +0.60 | ~132 BPM | Major |
| Sad | -0.70 | -0.40 | ~78 BPM | Minor |
| Angry | -0.60 | +0.70 | ~136 BPM | Minor |
| Fear | -0.70 | +0.60 | ~132 BPM | Minor |
| Neutral | 0.00 | 0.00 | 120 BPM | Major |

## Estructura de Módulos del Core

```python
core/
├── camera/          # WebcamCapture
├── emotion/         # DeepFaceEmotionDetector, normalize_emotion
├── va/              # emotion_to_va, EmotionVAMapper
├── music/           # va_to_music_params, generate_midi_baseline
├── pipeline/        # EmotionPipeline
└── utils/           # Funciones matemáticas (clamp, lerp, etc.)
```

## Notas Importantes

### Primera Ejecución

- DeepFace descarga automáticamente los modelos preentrenados (aproximadamente 100MB)
- El proceso de descarga puede tardar varios minutos dependiendo de la conexión
- Los modelos se almacenan en caché local para ejecuciones posteriores

### Configuración de Permisos

Es necesario otorgar permisos de acceso a la webcam:

- **macOS**: Sistema → Privacidad y Seguridad → Cámara
- **Windows**: Configuración → Privacidad → Cámara
- **Linux**: Verificar permisos del dispositivo /dev/video0
o

- Captura de webcam implementada
- Reconocimiento emocional facial con DeepFace
- Normalización de emociones a conjunto estándar
- Mapeo de emociones a espacio Valence-Arousal
- Conversión de coordenadas VA a parámetros musicales
- Pipeline integrado con suavizado temporal
- Generador MIDI baseline basado en reglas
- Pendiente: API Flask/FastAPI
- Pendiente: Generador MIDI con modelos de aprendizaje automático
**Versión actual:** 0.1.0
