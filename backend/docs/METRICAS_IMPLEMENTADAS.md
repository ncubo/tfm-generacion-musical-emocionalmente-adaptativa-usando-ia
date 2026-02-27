# Métricas del Sistema

**Última verificación:** 24/02/2026

---

## 1. Métricas Musicales

`src/core/music/analysis/features.py` → `extract_midi_features()`

| Métrica | Fórmula | Rango |
|---------|---------|-------|
| note_density | `len(notes) / total_duration_seconds` | 0.5-10.0 notes/sec |
| pitch_range | `max(pitches) - min(pitches)` | 12-60 semitonos |
| mean_velocity | `sum(velocities) / len(velocities)` | 40-100 MIDI |
| total_duration_seconds | `(total_ticks / ticks_per_beat) * (tempo / 1_000_000)` | variable |

---

## 2. Rendimiento

`src/core/utils/metrics.py` → `PerformanceMetrics.measure(stage_name)` → yields `timing['duration']`

`scripts/run_final_benchmark.py` → `generation_time_ms` → medido con `time.perf_counter()`

Outputs: `benchmark_raw.csv`, `benchmark_aggregated.csv` (mean, std, min, max)

---

## 3. Estabilidad Emocional

`scripts/analyze_stability.py` → `calculate_metrics()`

| Métrica | Fórmula |
|---------|---------|
| n_samples | `len(data)` |
| emotion_changes | `sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])` |
| emotion_stability_pct | `(len(emotions) - emotion_changes) / len(emotions) * 100` |
| valence_variance | `statistics.variance(valences)` |
| arousal_variance | `statistics.variance(arousals)` |
| valence_roc | `mean([abs(valences[i] - valences[i-1]) for i in range(1, len(valences))])` |
| arousal_roc | `mean([abs(arousals[i] - arousals[i-1]) for i in range(1, len(arousals))])` |
| avg_valence | `statistics.mean(valences)` |
| avg_arousal | `statistics.mean(arousals)` |

---

## 4. Correlaciones

### Spearman
`scripts/analyze_final_benchmark.py` → `calculate_spearman_correlations()`

`scipy.stats.spearmanr(arousals, values)` → arousal vs {note_density, pitch_range, mean_velocity}

### Pearson
`scripts/evaluate_dimensional_alignment.py`

`scipy.stats.pearsonr(targets, estimated)` → valence/arousal target vs estimated

Outputs: r, R², p-value, MAE, RMSE

---

## 5. Métricas Compuestas

### System Coherence
`scripts/calculate_system_coherence.py`

`Score = 0.5×EmotionalAlignment + 0.3×MusicalStructure + 0.2×Latency`

- EmotionalAlignment: `mean([abs(spearmanr(arousals, metric_values)[0]) for metric in metrics])`
- MusicalStructure: `1.0 - (avg_std / 127.0)` (std de mean_velocity por VA bin)
- Latency: `1.0 - (mean_latency / 10000.0)`

Outputs: `system_coherence.json`, `system_coherence.tex`

### Dimensional Alignment
`scripts/evaluate_dimensional_alignment.py` → `compute_va_heuristic()`

```python
valence = (mean_velocity / 127.0) * 2.0 - 1.0
arousal_density = np.clip(note_density / 5.0, 0.0, 1.0)
arousal_pitch = np.clip(pitch_range / 48.0, 0.0, 1.0)
arousal = (0.6 * arousal_density + 0.4 * arousal_pitch) * 2.0 - 1.0
```

Outputs: `dimensional_alignment.csv/json/tex`, `valence_alignment_scatter.png`, `arousal_alignment_scatter.png`

### Dataset Comparison
`scripts/compare_dataset_vs_generated.py`

KL Divergence: `scipy.stats.entropy(hist_dataset, hist_generated)` (30 bins, ε=1e-10)

Estadísticas: mean, std, min, max, median, count

Outputs: `comparison_statistics.csv/tex`, `comparison_summary.json`

### Emotion Classifier
`scripts/evaluate_emotion_classifier.py`

sklearn: `confusion_matrix`, `accuracy_score`, `precision_recall_fscore_support`

Outputs: `confusion_matrix.csv/png`, `classification_report.csv/tex`, `metrics_summary.json`

Estado: pendiente ground truth dataset

---

## 6. Uso

```bash
python scripts/run_final_benchmark.py
python scripts/analyze_final_benchmark.py results/final_benchmark_YYYYMMDD_HHMMSS

python scripts/calculate_system_coherence.py --benchmark_results results/final_benchmark_*
python scripts/evaluate_dimensional_alignment.py --benchmark_results results/final_benchmark_*
python scripts/compare_dataset_vs_generated.py --dataset_dir data/lakh_piano_clean --generated_dir results/final_benchmark_*/baseline
python scripts/evaluate_emotion_classifier.py --dataset_csv data/emotion_ground_truth.csv
```

---

## 7. Dependencias

`seaborn>=0.12.0`, `scipy>=1.10.0`, `scikit-learn>=1.3.0`

---

**Total:** 4 musicales + 2 rendimiento + 9 estabilidad + 2 correlación + 4 compuestas = 21 métricas
