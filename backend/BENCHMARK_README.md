# Benchmark de Motores de Generación Musical

Sistema de evaluación comparativa entre el motor baseline (reglas) y transformer_pretrained (Maestro-REMI) con múltiples seeds para análisis estadístico.

## Uso

### Ejecución básica

```bash
python scripts/run_benchmark_models.py --grid default --seed_base 42
```

### Benchmark con estadísticas (recomendado)

```bash
python scripts/run_benchmark_models.py \
    --grid default \
    --seed_base 42 \
    --num_seeds 5 \
    --max_tokens 512
```

## Argumentos

- `--grid`: Grid de valores V/A (`default` o `custom`)
- `--seed_base`: Semilla inicial (default: 42)
- `--num_seeds`: Réplicas por combinación (default: 1)
- `--max_tokens`: Tokens fijos para Transformer (default: auto)
- `--length_bars`: Compases por MIDI (default: 8)
- `--verbose`: Logs detallados

## Grids

**Default**: 4×4 valores V/A  
**Custom**: 5×5 valores V/A

Total muestras: combinaciones × 2 engines × num_seeds

## Salidas

```
data/benchmark_midis/              # MIDIs generados
results/
  benchmark_raw.json               # Resultados individuales
  benchmark_table.csv              # Tabla CSV individual
  benchmark_table.tex              # Tabla LaTeX individual
  benchmark_aggregated.csv         # Mean ± std (CSV)
  benchmark_aggregated.tex         # Mean ± std (LaTeX)
  benchmark_arousal_summary.tex    # Resumen por arousal
```

## Métricas

- `note_density`: notas/segundo
- `pitch_range`: semitonos (max-min)
- `mean_velocity`: intensidad promedio
- `mean_note_duration`: duración nota (segundos)
- `total_notes`: total de notas
- `unique_pitches`: notas únicas

## Agregación Estadística

Con `--num_seeds > 1` se generan automáticamente:

1. **Tabla agregada** por (engine, V, A) con mean ± std
2. **Tabla resumen** por arousal (promedio sobre valencia)

Las tablas LaTeX están listas para copiar al documento del TFM.

## Notas Técnicas

- **Seeds**: consecutivas desde `seed_base` (42, 43, 44...)
- **Control de longitud**: `max_tokens` fija la generación del Transformer
- **Primer vacío**: Con arousal muy bajo, el sistema garantiza density mínima en el prompt
- **Corrección de Bessel**: Se usa `ddof=1` en el cálculo de std

## Referencias

- Modelo: [Maestro-REMI-bpe20k](https://huggingface.co/Natooz/Maestro-REMI-bpe20k)
- Tokenización: [MidiTok](https://github.com/Natooz/MidiTok)
