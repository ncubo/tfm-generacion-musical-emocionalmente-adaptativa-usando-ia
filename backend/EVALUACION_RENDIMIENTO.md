# Evaluación de Rendimiento

Instrumentación y benchmarking del pipeline de generación musical emocionalmente adaptativa.

## Componentes

- **Instrumentación:** `backend/src/core/utils/metrics.py` (context manager, decorador)
- **Endpoints:** `/emotion` y `/generate-midi` (medición transparente)
- **Benchmark:** `backend/scripts/run_benchmarks.py`
- **Análisis:** `backend/scripts/analyze_metrics.py`

## Uso

### 1. Iniciar servidor

```bash
cd backend
source .venv/bin/activate
python src/app.py
```

### 2. Ejecutar benchmark

```bash
python scripts/run_benchmarks.py --iterations 30
```

**Args:** `--iterations N`, `--url URL`, `--output DIR`

### 3. Analizar resultados

```bash
python scripts/analyze_metrics.py --input metrics/benchmark_YYYYMMDD_HHMMSS.json
```

**Genera:** `.txt`, `.tex`, `.md`

## Métricas

- **Detección emocional:** captura + DeepFace + mapeo VA + suavizado
- **Generación MIDI:** detección + mapeo parámetros + baseline MIDI

**Estadísticas:** mean, median, std, min/max, p95/p99

## Criterios

| Latencia Total | Calidad |
|----------------|---------|
| < 500 ms | Excelente |
| 500-1000 ms | Aceptable |
| > 1000 ms | Mejorable |
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
