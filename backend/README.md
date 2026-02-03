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

### Demo 1: Reconocimiento Emocional en Tiempo Real

Muestra la emoción detectada y las coordenadas Valence-Arousal en tiempo real con estabilización temporal:

```bash
python scripts/run_webcam_demo.py
```

**Características:**
- Detección emocional facial en tiempo real
- Estabilización temporal con EMA + ventana de mayoría
- Visualización de V/A y confianza
- Parámetros optimizados para robustez

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

### Demo 3: Test de Generación MIDI

Genera archivos MIDI de prueba sin webcam:

```bash
python scripts/test_midi_generation.py
```

Este script genera archivos MIDI para las siguientes emociones: happy, sad, angry, fear, neutral, excited, calm. No requiere webcam y sirve para validar el funcionamiento del generador MIDI.

### Demo 4: Comparación de Estabilidad Temporal

Compara visualmente el sistema con y sin estabilización:

```bash
python scripts/compare_stability.py
```

**Características:**
- Compara dos configuraciones en tiempo real
- Cambia entre modos con tecla `s`
- Observa diferencias en estabilidad y responsividad

**Uso recomendado:** Haz expresiones faciales variadas para ver cómo cada configuración responde.

### Demo 5: Análisis Cuantitativo de Estabilidad

Mide métricas objetivas de estabilidad del sistema:

```bash
# Análisis de 30 segundos por configuración
python scripts/analyze_stability.py --duration 30
```

**Métricas calculadas:**
- Frecuencia de cambios de emoción
- Varianza de valores V/A
- Rate of change de V/A
- Estabilidad de emoción (%)

**Uso:** Mantén expresión neutral 10-15s, luego cambia a otra emoción y mantén 10-15s.

## Guía de Uso de los Módulos

### Pipeline de Procesamiento Completo

El pipeline emocional integra captura de video, detección facial, y estabilización temporal mejorada.

```python
from core import EmotionPipeline, WebcamCapture, DeepFaceEmotionDetector

# Crear componentes
webcam = WebcamCapture(camera_index=0)
detector = DeepFaceEmotionDetector()

# Crear pipeline integrado con estabilización temporal mejorada
pipeline = EmotionPipeline(
    camera=webcam,
    detector=detector,
    window_size=7,       # Ventana de mayoría para emoción discreta
    alpha=0.3,           # Factor EMA para V/A (0.1-1.0)
    min_confidence=60.0  # Umbral de confianza mínima (%)
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

**Parámetros de Estabilización:**
- `window_size`: Tamaño de ventana para mayoría de emoción (5-10 recomendado)
- `alpha`: Factor de suavizado EMA para V/A (menor = más suave)
- `min_confidence`: Confianza mínima para aceptar cambios de emoción

Ver [ESTABILIZACION_TEMPORAL.md](ESTABILIZACION_TEMPORAL.md) para detalles técnicos.

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
## Arquitectura del Sistema de Mapeo Emocional

### Flujo de Procesamiento: Emoción a Música

1. **Captura de Video**: Webcam captura frames en tiempo real
2. **Detección Facial**: DeepFace detecta emoción facial con scores de confianza
3. **Normalización**: Emoción se normaliza al conjunto estándar
4. **Estabilización Temporal**:
   - **Filtro de Confianza**: Solo acepta detecciones con confianza > umbral
   - **Ventana de Mayoría**: Emoción discreta se estabiliza por consenso
   - **EMA para V/A**: Suavizado exponencial de valencia y arousal
5. **Mapeo VA**: Emoción estable → coordenadas continuas (Valence, Arousal)
6. **Parámetros Musicales**: Coordenadas VA → parámetros MIDI
7. **Generación MIDI**: Parámetros → archivo MIDI reproducible

### Mejoras de Estabilidad Temporal

El sistema implementa un **mecanismo dual de estabilización**:

**1. Media Móvil Exponencial (EMA) para Valence-Arousal:**
- Más responsive que media móvil simple
- Balance entre suavizado y capacidad de respuesta
- Configurable mediante parámetro `alpha`

**2. Ventana de Mayoría para Emoción Discreta:**
- Evita "parpadeos" entre emociones
- Requiere consenso en ventana temporal
- Configurable mediante parámetro `window_size`

**3. Filtro de Confianza:**
- Rechaza detecciones de baja calidad
- Reduce falsos positivos
- Configurable mediante parámetro `min_confidence`

**Resultado:** Sistema más robusto y perceptualmente estable sin comprometer latencia.

Ver documentación completa en [ESTABILIZACION_TEMPORAL.md](ESTABILIZACION_TEMPORAL.md).
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
