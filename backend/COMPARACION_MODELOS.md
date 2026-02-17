# Comparación Pretrained vs Finetuned

Sistema de benchmarking para comparar modelos Maestro-REMI pretrained vs finetuned en generación musical VA-condicionada.

## Arquitectura

### 1. Engine Refactorizado (hf_maestro_remi.py)

**Cambios principales:**
- Cache multi-modelo: soporta pretrained y finetuned simultáneamente
- Parámetros nuevos en `generate_midi_hf_maestro_remi()`:
  - `model_source`: "pretrained" | "finetuned"
  - `model_id_or_path`: HF Hub ID o path local
- Añade conditioning tokens VA automáticamente desde `dataset_info.json`
- Hace `resize_token_embeddings()` si el modelo no tiene los tokens

**API:**
```python
from core.music.engines.hf_maestro_remi import generate_midi_hf_maestro_remi

generate_midi_hf_maestro_remi(
    params=music_params,
    out_path="output.mid",
    model_source="pretrained",  # o "finetuned"
    model_id_or_path="Natooz/Maestro-REMI-bpe20k"  # o path local
)
```

### 2. Script de Comparación (compare_pretrained_vs_finetuned.py)

Genera MIDIs con ambos modelos sobre un grid VA y extrae métricas.

**Características:**
- Grid configurable: 4x4 (16 combinaciones VA) o 3x3 (9 combinaciones)
- Seeds múltiples para rigor estadístico (default: 3 seeds)
- Control de longitud fijo (`max_tokens=512`) para benchmark justo
- Extrae 9 métricas musicales por MIDI
- Exporta CSV detallado con todos los experimentos

**Uso:**
```bash
# Benchmark completo (16 VA x 3 seeds x 2 modelos = 96 MIDIs)
python scripts/compare_pretrained_vs_finetuned.py \
    --grid 4x4 \
    --seeds "42,43,44" \
    --output_dir results/compare_pre_vs_finetuned

# Benchmark rápido (9 VA x 2 seeds x 2 modelos = 36 MIDIs)
python scripts/compare_pretrained_vs_finetuned.py \
    --grid 3x3 \
    --seeds "42,43" \
    --output_dir results/compare_quick
```

**Salida:** `results/compare_pre_vs_finetuned/{pretrained,finetuned}/v{v}_a{a}/seed{s}.mid` + `compare_raw.csv`

### 3. Script de Análisis (analyze_compare_results.py)

Procesa resultados y genera agregaciones estadísticas.

**Características:**
- Agrega por (model_tag, valence, arousal) con mean ± std
- Genera tabla LaTeX compacta para TFM
- Genera resumen textual en español con hallazgos (5-8 bullets)
- Identifica diferencias significativas entre modelos

**Uso:**
```bash
# Análisis automático (detecta input_csv y output_dir)
python scripts/analyze_compare_results.py

# Análisis con paths custom
python scripts/analyze_compare_results.py \
    --input_csv results/compare_quick/compare_raw.csv \
    --output_dir results/compare_quick
```

**Salida:** `compare_aggregated.csv` (estadísticas), `compare_table.tex` (LaTeX), `compare_summary.txt` (hallazgos)

## Pipeline Completo

```bash
python scripts/compare_pretrained_vs_finetuned.py --grid 4x4
python scripts/analyze_compare_results.py
cat results/compare_pre_vs_finetuned/compare_summary.txt
```

## Métricas Extraídas

Cada MIDI generado se analiza con estas métricas:

1. **note_density**: Notas por segundo (densidad temporal)
2. **pitch_range**: Rango de pitches (max - min en semitonos)
3. **mean_velocity**: Velocidad promedio (dinámica)
4. **mean_note_duration**: Duración promedio de notas (segundos)
5. **total_notes**: Número total de notas
6. **total_duration_seconds**: Duración total del MIDI
7. **unique_pitches**: Número de pitches únicos usados

## Requisitos

- Modelo finetuned en: `backend/models/maestro_finetuned/final/`
- Dataset info en: `backend/data/finetune_dataset/dataset_info.json`
- Dependencies: transformers, miditok, torch, symusic, mido, numpy

## Troubleshooting

**Modelo no encontrado:** Verificar `--finetuned_path models/maestro_finetuned/final`  
**Dataset_info.json faltante:** Modelos usarán vocab base sin conditioning tokens  
**OOM:** Reducir grid (`--grid 3x3`) o seeds (`--seeds "42"`)

## Testing Rápido

```bash
python scripts/compare_pretrained_vs_finetuned.py --grid 3x3 --seeds "42"
python scripts/analyze_compare_results.py
```
