# Backend - Sistema de Generación Musical Emocional

Backend del sistema de generación musical adaptativa basada en reconocimiento emocional mediante técnicas de Inteligencia Artificial.

## Estructura del Proyecto

```
backend/
├── requirements.txt
├── src/
│   ├── app.py               # API Flask
│   ├── core/
│   │   ├── camera/          # Captura webcam
│   │   ├── emotion/         # Detección emocional
│   │   ├── va/              # Mapeo Valencia-Activación
│   │   ├── music/           # Generación MIDI
│   │   ├── pipeline/        # Pipeline integrado
│   │   └── utils/           # Utilidades
│   └── routes/             # Endpoints API
├── scripts/                 # Demos y análisis
├── data/                    # Datasets
└── metrics/                 # Resultados benchmarks
```

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- Webcam funcional para captura de video en tiempo real
- Permisos de sistema para acceso a la cámara

### Entorno Virtual

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# o
.venv\Scripts\activate  # Windows
```

### Instalación de Dependencias

```bash
cd backend
pip install -r requirements.txt
```

### Dependencias Principales

- **opencv-python**: Captura de video
- **deepface**: Reconocimiento emocional facial
- **mido**: Generación de archivos MIDI
- **miditok**: Tokenización REMI (Maestro-REMI-bpe20k)
- **transformers**: Modelos HuggingFace
- **datasets**: Construcción de datasets fine-tuning

### Demo 1: Reconocimiento Emocional en Tiempo Real

Muestra la emoción detectada y las coordenadas Valencia-Activación en tiempo real con estabilización temporal:

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

### Demo 3: Comparación de Estabilidad Temporal

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

### Flujo de Procesamiento

1. **Captura**: Webcam captura frames en tiempo real
2. **Detección**: DeepFace detecta emoción facial con scores de confianza
3. **Estabilización Temporal**:
   - Filtro de confianza (>60%)
   - Ventana de mayoría (emoción discreta)
   - EMA para V/A (suavizado exponencial)
4. **Mapeo VA**: Emoción → coordenadas Valencia-Activación
5. **Parámetros Musicales**: VA → tempo, modo, densidad, etc.
6. **Generación MIDI**: Parámetros → archivo MIDI

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

## Documentación Adicional

### Dataset y Fine-tuning
- [PREPARACION_DATASET.md](PREPARACION_DATASET.md): Preparación del Lakh MIDI dataset
- [DATASET_FINETUNING.md](DATASET_FINETUNING.md): Construcción de dataset para fine-tuning

### Evaluación y Benchmarks
- [BENCHMARK.md](BENCHMARK.md): Evaluación comparativa de motores
- [COMPARACION_MODELOS.md](COMPARACION_MODELOS.md): Comparación pretrained vs finetuned
- [EVALUACION_RENDIMIENTO.md](EVALUACION_RENDIMIENTO.md): Métricas de rendimiento
- [DOCUMENTACION_API.md](DOCUMENTACION_API.md): Documentación completa de la API REST

## Notas

- DeepFace descarga modelos preentrenados (~100MB) en primera ejecución
- Modelos se cachean localmente
