# Benchmark Motores de Generación Musical

Evaluación comparativa baseline (reglas) vs transformer_pretrained (Maestro-REMI) con múltiples seeds.

**NOTA:** Este script es LEGACY. Para el benchmark final del TFM, usar `run_final_benchmark.py` (ver [BENCHMARK_FINAL.md](BENCHMARK_FINAL.md)).

## Uso

```bash
python scripts/run_benchmark_models.py --grid default --seed_base 42 --num_seeds 5
```

**Args:** `--grid {default,custom}`, `--seed_base N`, `--num_seeds N`, `--max_tokens N`, `--length_bars N`

**Grids:** Default 4×4, Custom 5×5 valores V/A

## Métricas

**Métricas musicales** (proxy de arousal):
- `note_density` - Notas por segundo (densidad temporal)
- `pitch_range` - Rango tonal en semitonos
- `mean_velocity` - Intensidad dinámica MIDI 0-127

**Métricas de rendimiento** (validación y eficiencia):
- `total_duration_seconds` - Duración total del MIDI
- `generation_time_ms` - Tiempo de generación en milisegundos

## Salidas

```
data/benchmark_midis/
results/
  benchmark_raw.json
  benchmark_table.csv
  benchmark_table.tex
  benchmark_aggregated.csv  # mean ± std
  benchmark_aggregated.tex
  benchmark_arousal_summary.tex
```

Tablas LaTeX listas para TFM.
