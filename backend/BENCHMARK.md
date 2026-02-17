# Benchmark Motores de Generación Musical

Evaluación comparativa baseline (reglas) vs transformer_pretrained (Maestro-REMI) con múltiples seeds.

## Uso

```bash
python scripts/run_benchmark_models.py --grid default --seed_base 42 --num_seeds 5
```

**Args:** `--grid {default,custom}`, `--seed_base N`, `--num_seeds N`, `--max_tokens N`, `--length_bars N`

**Grids:** Default 4×4, Custom 5×5 valores V/A

## Métricas

- `note_density` (notas/seg)
- `pitch_range` (semitonos)
- `mean_velocity`
- `mean_note_duration` (seg)
- `total_notes`
- `unique_pitches`

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
