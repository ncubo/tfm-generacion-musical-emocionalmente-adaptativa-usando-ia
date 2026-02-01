# Evaluación de Rendimiento - TFM

Sistema de instrumentación y benchmarking para evaluar el rendimiento del pipeline de generación musical emocionalmente adaptativa.

## Componentes

### 1. Módulo de Instrumentación
**Ubicación:** `backend/src/core/utils/metrics.py`

Proporciona herramientas para medir tiempos de ejecución:
- Context manager `measure()` para mediciones puntuales
- Decorador `measure_function()` para funciones
- Cálculo automático de estadísticas (media, mediana, desviación estándar, percentiles)
- Exportación a CSV y JSON

### 2. Endpoints Instrumentados
**Ubicación:** `backend/src/routes/emotion.py` y `music.py`

Endpoints modificados para incluir mediciones:
- `/emotion` - Mide latencia de detección emocional
- `/generate-midi` - Mide latencia completa del pipeline (emoción + generación MIDI)

Las mediciones son transparentes y no afectan la funcionalidad.

### 3. Script de Benchmarking
**Ubicación:** `backend/scripts/run_benchmarks.py`

Ejecuta múltiples iteraciones de los endpoints para obtener métricas representativas.

### 4. Análisis y Reportes
**Ubicación:** `backend/scripts/analyze_metrics.py`

## Prerequisitos

### Activar el entorno virtual

Antes de ejecutar cualquier script, asegúrate de activar el entorno virtual de Python:

```bash
cd backend
source .venv/bin/activate  # En macOS/Linux
# o
.venv\Scripts\activate  # En Windows
```

Verifica que el entorno está activo (debería aparecer `(.venv)` en el prompt de la terminal).

## Uso

### Paso 1: Activar entorno virtual e iniciar el servidor

```bash
cd backend
source .venv/bin/activate
python src/app.py
```

### Paso 2: Ejecutar Benchmarks (en otra terminal)

```bash
cd backend
source .venv/bin/activate
python scripts/run_benchmarks.py --iterations 30
```

**Opciones:**
- `--iterations N`: Número de mediciones por endpoint (default: 25)
- `--url URL`: URL del servidor (default: http://localhost:5000)
- `--output DIR`: Directorio de salida (default: metrics)

**Ejemplo:**
```bash
source .venv/bin/activate  # Siempre activar primero
python scripts/run_benchmarks.py --iterations 50 --url http://localhost:5000 --output resultados
```

### Paso 3: Analizar Resultados

```bash
source .venv/bin/activate  # Si no está ya activo
python scripts/analyze_metrics.py --input metrics/benchmark_YYYYMMDD_HHMMSS.json
```

**Genera:**
- Reporte de texto detallado (`.txt`)
- Tabla LaTeX para memoria (`.tex`)
- Tabla Markdown (`.md`)

## Métricas Medidas

### Latencia de Detección Emocional
- Captura de frame desde webcam
- Inferencia del modelo DeepFace
- Mapeo a coordenadas Valence-Arousal
- Suavizado temporal

### Latencia de Generación MIDI
- Detección emocional
- Mapeo VA → parámetros musicales
- Generación de archivo MIDI baseline

### Estadísticas Calculadas
- **Media**: Valor promedio de latencia
- **Mediana**: Valor central (robusto ante outliers)
- **Desviación estándar**: Dispersión de las mediciones
- **Mínimo/Máximo**: Rango de valores observados
- **Percentil 95/99**: Latencia en el 95% y 99% de los casos

## Interpretación de Resultados

### Criterios de Evaluación

| Latencia Total | Categoría | Descripción |
|---------------|-----------|-------------|
| < 500 ms | **Excelente** | Interacción fluida, imperceptible |
| 500-1000 ms | **Bueno** | Tiempo real aceptable |
| > 1000 ms | **Mejorable** | Delay perceptible |

### Ejemplo de Salida

```
[EMOTION DETECTION]
  Mediciones:      30
  Media:           245.32 ms
  Mediana:         242.18 ms
  Desv. Estándar:  28.45 ms
  Mínimo:          198.12 ms
  Máximo:          312.67 ms
  Percentil 95:    289.45 ms
  Percentil 99:    305.23 ms

[TOTAL PIPELINE]
  Mediciones:      30
  Media:           487.56 ms
  Mediana:         481.23 ms
  Desv. Estándar:  42.18 ms
  Mínimo:          412.45 ms
  Máximo:          598.34 ms
  Percentil 95:    556.78 ms
  Percentil 99:    582.12 ms

[EVALUACION DE TIEMPO REAL]
  Latencia promedio: 487.56 ms
  ¿Tiempo real?:     SI
  Criterio:          < 1000 ms para interacción aceptable
```

## Estructura de Archivos Generados

```
metrics/
├── benchmark_YYYYMMDD_HHMMSS.json       # Datos crudos + estadísticas
├── reporte_rendimiento_YYYYMMDD.txt     # Reporte completo
├── tabla_latex_YYYYMMDD.tex             # Tabla para LaTeX
└── tabla_markdown_YYYYMMDD.md           # Tabla Markdown
```

## Configuración Avanzada

### Activar Métricas en Respuestas HTTP

En `backend/src/app.py`:

```python
app.config['INCLUDE_METRICS'] = True
```

Esto añade `processing_time_ms` a las respuestas JSON (útil para debugging).

### Personalizar Directorio de Salida

```python
from src.core.utils.metrics import get_metrics
from pathlib import Path

metrics = get_metrics(output_dir=Path('mi_directorio'))
```

### Acceder a Métricas Programáticamente

```python
from src.core.utils.metrics import get_metrics

metrics = get_metrics()

# Obtener estadísticas
stats = metrics.get_statistics()
print(stats['emotion_detection']['mean'])

# Imprimir resumen
metrics.print_summary()

# Guardar resultados
metrics.save_to_csv('mis_metricas.csv')
metrics.save_to_json('mis_metricas.json')
```

## Troubleshooting

### Error: "El servidor no está disponible"
- Verificar que Flask esté ejecutándose en el puerto correcto
- Comprobar URL con `curl http://localhost:5000/health`

### Latencias muy altas
- Verificar que DeepFace no esté descargando modelos
- Comprobar uso de CPU/GPU
- Revisar logs del servidor

### Resultados inconsistentes
- Ejecutar más iteraciones (30-50 recomendadas)
- Cerrar aplicaciones pesadas en segundo plano
- Verificar que la webcam funciona correctamente

### Error: "ModuleNotFoundError" o "ImportError"
- Asegurarse de que el entorno virtual está activado
- Verificar con: `which python` (debe apuntar a .venv/bin/python)
- Reinstalar dependencias si es necesario: `pip install -r requirements.txt`

## Referencias

- Métricas de rendimiento basadas en: ISO/IEC 25010:2011 (Software Quality)
- Criterios de latencia para sistemas interactivos: Nielsen, J. (1993)
- Estadística: Percentiles para evaluación de sistemas en producción
