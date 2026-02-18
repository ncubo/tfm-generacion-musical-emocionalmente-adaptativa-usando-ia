# Benchmark Final de Motores de Generación Musical

Sistema de evaluación reproducible de los 3 motores de generación musical sobre un grid Valencia-Arousal.

## Descripción

Este benchmark evalúa los siguientes engines:

1. **Baseline** - Generador heurístico basado en reglas
2. **Transformer Pretrained** - Modelo `Natooz/Maestro-REMI-bpe20k` sin fine-tuning
3. **Transformer Finetuned** - Modelo `mmayorga/maestro-remi-finetuned-va` con fine-tuning VA

### Metodología

- Grid VA: 4x4 puntos (valence × arousal) con valores `[-0.8, -0.2, +0.2, +0.8]`
- Seeds reproducibles: Por defecto `[42, 43, 44]`
- Total items: 3 engines × 16 puntos VA × 3 seeds = 144 MIDIs
- Métricas extraídas:
  - `note_density` - Notas por segundo
  - `pitch_range` - Rango tonal (semitonos)
  - `mean_velocity` - Velocity media (intensidad)
  - `mean_note_duration` - Duración media de nota
  - `total_notes` - Total de notas generadas
  - `total_duration_seconds` - Duración total
  - `unique_pitches` - Notas únicas usadas
  - `generation_time_ms` - Latencia de generación

## Ejecución

### 1. Activar entorno virtual

```bash
cd backend
source .venv/bin/activate
```

### 2. Ejecutar benchmark (generación)

Benchmark completo (144 MIDIs, aproximadamente 20-40 minutos dependiendo del hardware):
```bash
python scripts/run_final_benchmark.py
```

Benchmark rápido (3x3 grid, 1 seed = 27 MIDIs):
```bash
python scripts/run_final_benchmark.py --grid 3x3 --seeds "42"
```

Prueba rápida (solo 10 MIDIs):
```bash
python scripts/run_final_benchmark.py --max_items 10
```

Benchmark solo baseline (para debugging):
```bash
python scripts/run_final_benchmark.py --engines baseline --max_items 5
```

No regenerar MIDIs existentes:
```bash
python scripts/run_final_benchmark.py --skip_existing
```

Forzar CPU (si hay problemas con CUDA):
```bash
python scripts/run_final_benchmark.py --device cpu
```

### 3. Analizar resultados

Una vez generado el benchmark, analizar los datos:

```bash
python scripts/analyze_final_benchmark.py results/final_benchmark_YYYYMMDD_HHMMSS
```

Reemplazar `YYYYMMDD_HHMMSS` con el timestamp del directorio generado.

## Outputs Generados

### A) run_final_benchmark.py

Directorio: `backend/results/final_benchmark_YYYYMMDD_HHMMSS/`

- `benchmark_raw.csv` - Datos crudos (1 fila por MIDI)
  - Columnas: engine, valence, arousal, seed, status, generation_time_ms, métricas, midi_path, error
  
- `metadata.json` - Configuración del benchmark
  - Grid usado, seeds, engines evaluados, repos HF
  - Resumen de resultados (éxitos/errores)
  
- `{engine}/v{V}_a{A}/seed{S}.mid` - Archivos MIDI generados
  - Organizados jerárquicamente por engine y coordenadas VA

### B) analyze_final_benchmark.py

En el mismo directorio:

- `benchmark_aggregated.csv` - Estadísticas agregadas por (engine, V, A)
  - Mean y std de cada métrica
  
- `benchmark_table.tex` - Tablas LaTeX listas para copiar al documento
  - Tabla 1: Métricas musicales (density, pitch_range, velocity)
  - Tabla 2: Latencia de generación
  
- `benchmark_summary.txt` - Resumen de hallazgos en español (8-12 bullets)
  - Coherencia A→velocity, A→pitch_range
  - Estabilidad entre seeds
  - Comparación pretrained vs finetuned
  - Comparación vs baseline
  - Latencia por engine
  - Limitaciones
  
- Figuras PNG (5 archivos):
  - `heatmap_velocity_transformer_pretrained.png`
  - `heatmap_velocity_transformer_finetuned.png`
  - `heatmap_pitchrange_transformer_pretrained.png`
  - `heatmap_pitchrange_transformer_finetuned.png`
  - `bar_latency_by_engine.png`

## Opciones Avanzadas

### `run_final_benchmark.py`

```bash
python scripts/run_final_benchmark.py [opciones]

Opciones:
  --output_dir DIR      Directorio de salida personalizado
  --grid {4x4,3x3}      Tamaño del grid VA (default: 4x4)
  --seeds "S1,S2,S3"    Seeds separadas por comas (default: "42,43,44")
  --engines "E1,E2"     Engines a evaluar (default: todos)
                        Opciones: baseline, transformer_pretrained, transformer_finetuned
  --max_items N         Límite de items (para pruebas)
  --skip_existing       No regenerar MIDIs existentes
  --device {cpu,cuda}   Device para transformers (default: auto)
  -h, --help            Mostrar ayuda completa
```

### `analyze_final_benchmark.py`

```bash
python scripts/analyze_final_benchmark.py <results_dir>

Argumentos:
  results_dir           Directorio con benchmark_raw.csv
  -h, --help           Mostrar ayuda
```

## Ejemplo de Flujo Completo

```bash
# 1. Activar entorno
cd backend
source .venv/bin/activate

# 2. Ejecutar benchmark completo
python scripts/run_final_benchmark.py

# Salida esperada:
# [INFO] Directorio de resultados: results/final_benchmark_20260218_143000
# [INFO] Grid VA creado: 16 puntos (4x4)
# [INFO] Total de combinaciones: 144
# [INFO] Generando: baseline | V=-0.8 A=-0.8 | seed=42
# [INFO]   ✓ Generado en 45ms | Notas: 24 | Density: 1.20 n/s
# ...
# [INFO] BENCHMARK COMPLETADO
# [INFO] Tiempo total: 1234.5s (20.6 min)

# 3. Analizar resultados
python scripts/analyze_final_benchmark.py results/final_benchmark_20260218_143000

# Salida esperada:
# [INFO] ANÁLISIS DE BENCHMARK
# [INFO] Datos cargados: 144 filas
# [INFO] Datos agregados: 48 grupos
# [INFO] CSV agregado guardado: benchmark_aggregated.csv
# [INFO] Tabla LaTeX guardada: benchmark_table.tex
# [INFO] Resumen guardado: benchmark_summary.txt
# [INFO] Generando heatmaps...
# [INFO]   Heatmap guardado: heatmap_velocity_transformer_pretrained.png
# ...
# [INFO] ANÁLISIS COMPLETADO

# 4. Revisar resultados
cat results/final_benchmark_20260218_143000/benchmark_summary.txt
open results/final_benchmark_20260218_143000/*.png
```

## Troubleshooting

### Error: "transformers no está instalado"

```bash
pip install transformers>=4.40 miditok>=3.0 torch
```

### Error: "CUDA out of memory"

```bash
python scripts/run_final_benchmark.py --device cpu
```

### Error: "Tokenizer incompatible"

Verifica que `miditok>=3.0` esté instalado (versiones antiguas no son compatibles):

```bash
pip install --upgrade miditok
```

### Benchmark muy lento

Usa grid más pequeño o menos seeds:

```bash
python scripts/run_final_benchmark.py --grid 3x3 --seeds "42"
```

## Notas

- Reproducibilidad: Con las mismas seeds, el benchmark genera MIDIs idénticos
- Cache de modelos: Los transformers se cargan una sola vez y se reutilizan
- Manejo de errores: Si un MIDI falla, se registra el error y continúa con el siguiente
- Skip existing: Útil para reanudar benchmarks interrumpidos sin regenerar todo
- Paralelización: Actualmente secuencial (posible mejora futura: batch generation)

## Referencias

- Mapeo VA→params musicales: `backend/src/core/music/mapping.py`
- Engine baseline: `backend/src/core/music/engines/baseline.py`
- Engine transformers: `backend/src/core/music/engines/hf_maestro_remi.py`
- Extracción de métricas: `backend/src/core/music/analysis/features.py`
