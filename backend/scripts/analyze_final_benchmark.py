#!/usr/bin/env python3
"""
Script de análisis de resultados del benchmark final.

Lee benchmark_raw.csv y genera:
- benchmark_aggregated.csv (estadísticas agregadas)
- benchmark_table.tex (tabla LaTeX)
- benchmark_summary.txt (hallazgos en español)
- Figuras PNG (heatmaps y barras)

Uso:
    python analyze_final_benchmark.py <results_dir>
    python analyze_final_benchmark.py results/final_benchmark_20260218_120000
"""

import sys
import argparse
import logging
from pathlib import Path
import csv
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics

import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ===== FUNCIONES DE ANÁLISIS =====

def load_benchmark_data(csv_path: Path) -> List[Dict]:
    """
    Carga datos del CSV crudo.
    
    Args:
        csv_path: Path al benchmark_raw.csv
        
    Returns:
        Lista de diccionarios con los datos
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convertir tipos numéricos
            try:
                row['valence'] = float(row['valence'])
                row['arousal'] = float(row['arousal'])
                row['seed'] = int(row['seed'])
                row['generation_time_ms'] = float(row['generation_time_ms']) if row['generation_time_ms'] else 0
                
                # Parsear bars si existe (para benchmarks con escalabilidad)
                if 'bars' in row and row['bars']:
                    row['bars'] = int(row['bars'])
                
                # Métricas musicales (pueden estar vacías si hubo error)
                if row['note_density']:
                    row['note_density'] = float(row['note_density'])
                    row['pitch_range'] = int(row['pitch_range'])
                    row['mean_velocity'] = float(row['mean_velocity'])
                    row['total_duration_seconds'] = float(row['total_duration_seconds'])
                else:
                    row['note_density'] = None
                    row['pitch_range'] = None
                    row['mean_velocity'] = None
                    row['total_duration_seconds'] = None
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parseando fila: {e} | {row}")
                continue
            
            data.append(row)
    
    logger.info(f"Datos cargados: {len(data)} filas desde {csv_path.name}")
    return data


def aggregate_by_engine_va(data: List[Dict]) -> Dict[Tuple, Dict]:
    """
    Agrega datos por (engine, valence, arousal).
    
    Args:
        data: Lista de filas del benchmark
        
    Returns:
        Dict con key=(engine,v,a) y value=dict con stats
    """
    # Agrupar datos
    groups = defaultdict(lambda: {
        'note_density': [],
        'pitch_range': [],
        'mean_velocity': [],
        'total_duration_seconds': [],
        'generation_time_ms': []
    })
    
    for row in data:
        if row['status'] != 'success':
            continue
        
        key = (row['engine'], row['valence'], row['arousal'])
        
        # Agregar valores (solo si no son None)
        for metric in ['note_density', 'pitch_range', 'mean_velocity',
                      'total_duration_seconds', 'generation_time_ms']:
            value = row.get(metric)
            if value is not None:
                groups[key][metric].append(value)
    
    # Calcular estadísticas
    aggregated = {}
    for key, metrics in groups.items():
        stats = {}
        for metric, values in metrics.items():
            if values:
                stats[f"{metric}_mean"] = statistics.mean(values)
                stats[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                stats[f"{metric}_mean"] = None
                stats[f"{metric}_std"] = None
        
        aggregated[key] = stats
    
    logger.info(f"Datos agregados: {len(aggregated)} grupos (engine, V, A)")
    return aggregated


def aggregate_by_engine_bars(data: List[Dict]) -> Dict[Tuple, Dict]:
    """
    Agrega datos por (engine, length_bars) para análisis de escalabilidad.
    
    Calcula:
    - mean, median, stdev, p95 de generation_time_ms
    - success_rate (% de generaciones exitosas)
    
    Args:
        data: Lista de filas del benchmark
        
    Returns:
        Dict con key=(engine, bars) y value=dict con stats
    """
    # Agrupar datos
    groups = defaultdict(lambda: {
        'generation_time_ms': [],
        'total_count': 0,
        'success_count': 0
    })
    
    for row in data:
        # Verificar si existe bars en el row
        if 'bars' not in row:
            continue
        
        try:
            length_bars = int(row['bars'])
        except (ValueError, KeyError):
            continue
        
        key = (row['engine'], length_bars)
        groups[key]['total_count'] += 1
        
        if row['status'] == 'success':
            groups[key]['success_count'] += 1
            gen_time = row.get('generation_time_ms')
            if gen_time is not None and gen_time > 0:
                groups[key]['generation_time_ms'].append(gen_time)
    
    # Calcular estadísticas
    aggregated = {}
    for key, metrics in groups.items():
        times = metrics['generation_time_ms']
        total = metrics['total_count']
        success = metrics['success_count']
        
        stats = {}
        if times:
            stats['mean_ms'] = statistics.mean(times)
            stats['median_ms'] = statistics.median(times)
            stats['stdev_ms'] = statistics.stdev(times) if len(times) > 1 else 0.0
            stats['p95_ms'] = np.percentile(times, 95)
        else:
            stats['mean_ms'] = None
            stats['median_ms'] = None
            stats['stdev_ms'] = None
            stats['p95_ms'] = None
        
        stats['success_rate'] = (success / total * 100) if total > 0 else 0.0
        stats['total_count'] = total
        stats['success_count'] = success
        
        aggregated[key] = stats
    
    logger.info(f"Datos agregados por bars: {len(aggregated)} grupos (engine, bars)")
    return aggregated


def save_aggregated_csv(aggregated: Dict, output_path: Path):
    """
    Guarda CSV con datos agregados.
    
    Args:
        aggregated: Dict con stats agregadas
        output_path: Path al CSV de salida
    """
    # Preparar filas
    rows = []
    for (engine, v, a), stats in aggregated.items():
        row = {
            'engine': engine,
            'valence': v,
            'arousal': a
        }
        row.update(stats)
        rows.append(row)
    
    # Ordenar por engine, valence, arousal
    rows.sort(key=lambda r: (r['engine'], r['valence'], r['arousal']))
    
    # Escribir CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"CSV agregado guardado: {output_path} ({len(rows)} filas)")
    else:
        logger.warning("No hay datos agregados para guardar")


def save_bars_aggregated_csv(aggregated: Dict, output_path: Path):
    """
    Guarda CSV con datos agregados por (engine, bars).
    
    Args:
        aggregated: Dict con stats agregadas por (engine, bars)
        output_path: Path al CSV de salida
    """
    # Preparar filas
    rows = []
    for (engine, bars), stats in aggregated.items():
        row = {
            'engine': engine,
            'bars': bars
        }
        row.update(stats)
        rows.append(row)
    
    # Ordenar por engine, bars
    rows.sort(key=lambda r: (r['engine'], r['bars']))
    
    # Escribir CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"CSV agregado por bars guardado: {output_path} ({len(rows)} filas)")
    else:
        logger.warning("No hay datos agregados por bars para guardar")


def generate_latex_table(aggregated: Dict, output_path: Path):
    """
    Genera tabla LaTeX compacta para el documento.
    
    Dos tablas:
    1. Métricas musicales (density, pitch_range, velocity)
    2. Latencia de generación
    
    Args:
        aggregated: Dict con stats agregadas
        output_path: Path al archivo .tex
    """
    # Extraer valores únicos de V y A
    engines = sorted(set(k[0] for k in aggregated.keys()))
    
    # Ordenar claves
    keys_sorted = sorted(aggregated.keys(), key=lambda k: (k[0], k[1], k[2]))
    
    latex_lines = []
    
    # Header del documento
    latex_lines.append("% Tabla generada automáticamente por analyze_final_benchmark.py")
    latex_lines.append("% Copiar al documento LaTeX del TFM\n")
    
    # ===== TABLA 1: MÉTRICAS MUSICALES =====
    latex_lines.append("% ===== TABLA 1: MÉTRICAS MUSICALES =====")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Métricas musicales por engine y coordenadas VA (media ± std)}")
    latex_lines.append("\\label{tab:benchmark_musical_metrics}")
    latex_lines.append("\\small")
    latex_lines.append("\\begin{tabular}{l|cc|ccc}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Engine} & \\textbf{V} & \\textbf{A} & \\textbf{Density} & \\textbf{Pitch Range} & \\textbf{Velocity} \\\\")
    latex_lines.append("                &             &             & (notas/s)           & (semitonos)          & (MIDI)             \\\\")
    latex_lines.append("\\hline")
    
    current_engine = None
    for engine, v, a in keys_sorted:
        stats = aggregated[(engine, v, a)]
        
        # Separador entre engines
        if engine != current_engine:
            if current_engine is not None:
                latex_lines.append("\\hline")
            current_engine = engine
        
        # Formatear valores
        density_mean = stats.get('note_density_mean')
        density_std = stats.get('note_density_std')
        pitch_mean = stats.get('pitch_range_mean')
        pitch_std = stats.get('pitch_range_std')
        vel_mean = stats.get('mean_velocity_mean')
        vel_std = stats.get('mean_velocity_std')
        
        density_str = f"{density_mean:.2f}±{density_std:.2f}" if density_mean is not None else "N/A"
        pitch_str = f"{pitch_mean:.1f}±{pitch_std:.1f}" if pitch_mean is not None else "N/A"
        vel_str = f"{vel_mean:.1f}±{vel_std:.1f}" if vel_mean is not None else "N/A"
        
        # Nombre corto del engine
        engine_short = engine.replace("transformer_", "T-").replace("baseline", "Baseline")
        
        latex_lines.append(f"{engine_short} & {v:+.1f} & {a:+.1f} & {density_str} & {pitch_str} & {vel_str} \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}\n")
    
    # ===== TABLA 2: LATENCIA =====
    latex_lines.append("% ===== TABLA 2: LATENCIA DE GENERACIÓN =====")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Latencia de generación por engine (promedio sobre todas las condiciones VA)}")
    latex_lines.append("\\label{tab:benchmark_latency}")
    latex_lines.append("\\begin{tabular}{l|cc}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Engine} & \\textbf{Latencia Media (ms)} & \\textbf{Desviación Std (ms)} \\\\")
    latex_lines.append("\\hline")
    
    # Calcular latencia promedio por engine
    latency_by_engine = defaultdict(list)
    for (engine, v, a), stats in aggregated.items():
        lat_mean = stats.get('generation_time_ms_mean')
        if lat_mean is not None:
            latency_by_engine[engine].append(lat_mean)
    
    for engine in engines:
        latencies = latency_by_engine[engine]
        if latencies:
            mean_lat = statistics.mean(latencies)
            std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            engine_short = engine.replace("transformer_", "T-").replace("baseline", "Baseline")
            latex_lines.append(f"{engine_short} & {mean_lat:.1f} & {std_lat:.1f} \\\\")
        else:
            engine_short = engine.replace("transformer_", "T-").replace("baseline", "Baseline")
            latex_lines.append(f"{engine_short} & N/A & N/A \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}\n")
    
    # Guardar archivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    logger.info(f"Tabla LaTeX guardada: {output_path}")


def calculate_spearman_correlations(data: List[Dict]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Calcula correlaciones de Spearman entre arousal y métricas por engine.
    
    Args:
        data: Datos crudos del benchmark
        
    Returns:
        Dict con estructura {engine: {metric: (rho, p_value)}}
    """
    correlations = {}
    metrics = ['note_density', 'pitch_range', 'mean_velocity']
    
    # Filtrar solo datos exitosos
    data_ok = [row for row in data if row['status'] == 'success']
    
    # Agrupar por engine
    engines = set(row['engine'] for row in data_ok)
    
    for engine in engines:
        engine_data = [row for row in data_ok if row['engine'] == engine]
        
        if len(engine_data) < 3:  # Necesitamos al menos 3 puntos para correlación
            continue
            
        correlations[engine] = {}
        
        for metric in metrics:
            # Extraer pares (arousal, metric)
            pairs = [(row['arousal'], row[metric]) 
                    for row in engine_data 
                    if row.get(metric) is not None]
            
            if len(pairs) >= 3:
                arousals, values = zip(*pairs)
                rho, p_value = scipy.stats.spearmanr(arousals, values)
                correlations[engine][metric] = (rho, p_value)
    
    return correlations


def generate_summary_text(aggregated: Dict, data: List[Dict], output_path: Path):
    """
    Genera resumen de hallazgos en español.
    
    Args:
        aggregated: Dict con stats agregadas
        data: Datos crudos del benchmark
        output_path: Path al archivo .txt
    """
    lines = []
    lines.append("=" * 80)
    lines.append("RESUMEN DE HALLAZGOS - BENCHMARK FINAL")
    lines.append("=" * 80)
    lines.append("")
    
    # 0. Correlaciones de Spearman (validación estadística)
    lines.append("0. CORRELACIONES DE SPEARMAN (Arousal vs Métricas)")
    lines.append("   " + "-" * 40)
    
    correlations = calculate_spearman_correlations(data)
    
    for engine in sorted(correlations.keys()):
        lines.append(f"   {engine}:")
        
        for metric, (rho, p_value) in sorted(correlations[engine].items()):
            # Formatear el nombre de la métrica
            metric_display = metric.replace('_', ' ').title()
            lines.append(f"     - {metric_display:20s}: rho = {rho:+.3f}  (p = {p_value:.3e})")
            
            # Interpretación y validación
            if abs(rho) > 0.7:
                strength = "FUERTE"
            elif abs(rho) > 0.5:
                strength = "MODERADA"
            elif abs(rho) > 0.3:
                strength = "DEBIL"
            else:
                strength = "MUY DEBIL"
            
            direction = "positiva" if rho > 0 else "negativa"
            lines.append(f"       -> Correlacion {strength} {direction}")
            
            # Validación específica para modelo finetuned
            if engine == 'transformer_finetuned' and metric == 'mean_velocity':
                if abs(rho) >= 0.5:
                    lines.append(f"       OK: |rho| >= 0.5 (criterio cumplido)")
                else:
                    lines.append(f"       FALLO: |rho| < 0.5 (revisar entrenamiento)")
        
        lines.append("")
    
    # Interpretación general
    if 'transformer_finetuned' in correlations:
        finetuned_corrs = correlations['transformer_finetuned']
        strong_metrics = [m for m, (rho, _) in finetuned_corrs.items() if abs(rho) >= 0.5]
        
        if len(strong_metrics) >= 2:
            lines.append(f"   RESUMEN: Finetuned tiene {len(strong_metrics)}/3 metricas con |rho| >= 0.5")
            lines.append("   -> Condicionamiento VA funcional")
        elif len(strong_metrics) == 1:
            lines.append(f"   RESUMEN: Finetuned tiene {len(strong_metrics)}/3 metricas con |rho| >= 0.5")
            lines.append("   -> Condicionamiento VA parcial")
        else:
            lines.append("   RESUMEN: Finetuned tiene 0/3 metricas con |rho| >= 0.5")
            lines.append("   -> Condicionamiento VA insuficiente (revisar training)")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    
    # 1. Coherencia A -> velocity
    lines.append("1. COHERENCIA AROUSAL -> VELOCITY")
    lines.append("   " + "-" * 40)
    
    # Agrupar por engine y calcular correlación simple
    for engine in sorted(set(k[0] for k in aggregated.keys())):
        engine_data = [(k[2], v['mean_velocity_mean']) 
                      for k, v in aggregated.items() 
                      if k[0] == engine and v['mean_velocity_mean'] is not None]
        
        if engine_data:
            arousals, velocities = zip(*engine_data)
            # Calcular correlación de Pearson simple
            n = len(arousals)
            if n > 1:
                mean_a = statistics.mean(arousals)
                mean_v = statistics.mean(velocities)
                cov = sum((a - mean_a) * (v - mean_v) for a, v in zip(arousals, velocities)) / n
                std_a = statistics.stdev(arousals)
                std_v = statistics.stdev(velocities)
                corr = cov / (std_a * std_v) if std_a > 0 and std_v > 0 else 0
                
                lines.append(f"   - {engine}: correlación A-velocity = {corr:.3f}")
                
                # Interpretación
                if abs(corr) > 0.7:
                    lines.append(f"     -> Correlacion FUERTE {'positiva' if corr > 0 else 'negativa'}")
                elif abs(corr) > 0.4:
                    lines.append(f"     -> Correlacion MODERADA {'positiva' if corr > 0 else 'negativa'}")
                else:
                    lines.append(f"     -> Correlacion DEBIL")
    
    lines.append("")
    
    # 2. Coherencia A -> pitch_range
    lines.append("2. COHERENCIA AROUSAL -> PITCH RANGE")
    lines.append("   " + "-" * 40)
    
    for engine in sorted(set(k[0] for k in aggregated.keys())):
        engine_data = [(k[2], v['pitch_range_mean']) 
                      for k, v in aggregated.items() 
                      if k[0] == engine and v['pitch_range_mean'] is not None]
        
        if engine_data:
            arousals, pitches = zip(*engine_data)
            n = len(arousals)
            if n > 1:
                mean_a = statistics.mean(arousals)
                mean_p = statistics.mean(pitches)
                cov = sum((a - mean_a) * (p - mean_p) for a, p in zip(arousals, pitches)) / n
                std_a = statistics.stdev(arousals)
                std_p = statistics.stdev(pitches)
                corr = cov / (std_a * std_p) if std_a > 0 and std_p > 0 else 0
                
                lines.append(f"   - {engine}: correlación A-pitch_range = {corr:.3f}")
    
    lines.append("")
    
    # 3. Estabilidad (std)
    lines.append("3. ESTABILIDAD ENTRE SEEDS (promedio de desviaciones estándar)")
    lines.append("   " + "-" * 40)
    
    for engine in sorted(set(k[0] for k in aggregated.keys())):
        engine_stds = [v['mean_velocity_std'] 
                      for k, v in aggregated.items() 
                      if k[0] == engine and v['mean_velocity_std'] is not None]
        
        if engine_stds:
            avg_std = statistics.mean(engine_stds)
            lines.append(f"   - {engine}: std promedio velocity = {avg_std:.2f}")
            
            if avg_std < 5:
                lines.append(f"     -> MUY ESTABLE (variacion <5 MIDI)")
            elif avg_std < 10:
                lines.append(f"     -> ESTABLE (variacion <10 MIDI)")
            else:
                lines.append(f"     -> VARIABILIDAD ALTA")
    
    lines.append("")
    
    # 4. Comparación pretrained vs finetuned
    lines.append("4. COMPARACIÓN TRANSFORMER PRETRAINED vs FINETUNED")
    lines.append("   " + "-" * 40)
    
    pretrained_velocities = [v['mean_velocity_mean'] 
                            for k, v in aggregated.items() 
                            if k[0] == 'transformer_pretrained' and v['mean_velocity_mean'] is not None]
    
    finetuned_velocities = [v['mean_velocity_mean'] 
                           for k, v in aggregated.items() 
                           if k[0] == 'transformer_finetuned' and v['mean_velocity_mean'] is not None]
    
    if pretrained_velocities and finetuned_velocities:
        mean_pre = statistics.mean(pretrained_velocities)
        mean_fine = statistics.mean(finetuned_velocities)
        std_pre = statistics.stdev(pretrained_velocities) if len(pretrained_velocities) > 1 else 0
        std_fine = statistics.stdev(finetuned_velocities) if len(finetuned_velocities) > 1 else 0
        
        lines.append(f"   - Pretrained: velocity = {mean_pre:.1f} ± {std_pre:.1f}")
        lines.append(f"   - Finetuned:  velocity = {mean_fine:.1f} ± {std_fine:.1f}")
        lines.append(f"   - Diferencia: {abs(mean_fine - mean_pre):.1f} MIDI")
    
    lines.append("")
    
    # 5. Comparación vs baseline
    lines.append("5. COMPARACIÓN vs BASELINE")
    lines.append("   " + "-" * 40)
    
    baseline_density = [v['note_density_mean'] 
                       for k, v in aggregated.items() 
                       if k[0] == 'baseline' and v['note_density_mean'] is not None]
    
    transformer_density = [v['note_density_mean'] 
                          for k, v in aggregated.items() 
                          if k[0] in ['transformer_pretrained', 'transformer_finetuned'] 
                          and v['note_density_mean'] is not None]
    
    if baseline_density and transformer_density:
        mean_base = statistics.mean(baseline_density)
        mean_trans = statistics.mean(transformer_density)
        
        lines.append(f"   - Baseline density:     {mean_base:.2f} notas/s")
        lines.append(f"   - Transformers density: {mean_trans:.2f} notas/s")
        
        if mean_trans > mean_base * 1.2:
            lines.append(f"     -> Transformers generan musica MAS DENSA (+{((mean_trans/mean_base - 1)*100):.1f}%)")
        elif mean_trans / mean_base < 0.8:
            lines.append(f"     -> Transformers generan musica MAS ESPACIADA ({((mean_trans/mean_base - 1)*100):.1f}%)")
        else:
            lines.append(f"     -> Densidades SIMILARES")
    
    lines.append("")
    
    # 6. Latencia por engine
    lines.append("6. LATENCIA DE GENERACIÓN")
    lines.append("   " + "-" * 40)
    
    latency_by_engine = {}
    for engine in sorted(set(k[0] for k in aggregated.keys())):
        engine_latencies = [v['generation_time_ms_mean'] 
                           for k, v in aggregated.items() 
                           if k[0] == engine and v['generation_time_ms_mean'] is not None]
        
        if engine_latencies:
            mean_lat = statistics.mean(engine_latencies)
            latency_by_engine[engine] = mean_lat
            lines.append(f"   - {engine}: {mean_lat:.0f} ms")
    
    # Comparar
    if len(latency_by_engine) > 1:
        fastest = min(latency_by_engine.items(), key=lambda x: x[1])
        slowest = max(latency_by_engine.items(), key=lambda x: x[1])
        
        lines.append(f"   - Más rápido: {fastest[0]} ({fastest[1]:.0f} ms)")
        lines.append(f"   - Más lento: {slowest[0]} ({slowest[1]:.0f} ms)")
        lines.append(f"   - Factor: {slowest[1] / fastest[1]:.1f}x")
    
    lines.append("")
    
    # 7. Tasa de éxito
    lines.append("7. TASA DE ÉXITO")
    lines.append("   " + "-" * 40)
    
    for engine in sorted(set(k[0] for k in aggregated.keys())):
        engine_rows = [r for r in data if r['engine'] == engine]
        total = len(engine_rows)
        success = sum(1 for r in engine_rows if r['status'] == 'success')
        errors = sum(1 for r in engine_rows if r['status'] == 'error')
        
        if total > 0:
            success_rate = (success / total) * 100
            lines.append(f"   - {engine}: {success}/{total} ({success_rate:.1f}% éxito, {errors} errores)")
    
    lines.append("")
    
    # 8. Limitaciones
    lines.append("8. LIMITACIONES")
    lines.append("   " + "-" * 40)
    lines.append("   - Mapeo VA heurístico (no aprendido)")
    lines.append("   - Dataset solo piano (Maestro)")
    lines.append("   - Finetuning sin optimización exhaustiva de hiperparámetros/epochs")
    lines.append("   - Grid VA reducido (no cubre todo el espacio continuo)")
    lines.append("   - Métricas objetivas (no evaluación perceptual humana)")
    lines.append("   - Longitud fija (8 compases)")
    
    lines.append("")
    lines.append("=" * 80)
    
    # Guardar archivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Resumen guardado: {output_path}")


def generate_heatmaps(aggregated: Dict, output_dir: Path):
    """
    Genera heatmaps de velocity y pitch_range para transformers.
    
    Args:
        aggregated: Dict con stats agregadas
        output_dir: Directorio donde guardar las figuras
    """
    logger.info("Generando heatmaps...")
    
    # Extraer valores únicos de V y A (ordenados)
    valences = sorted(set(k[1] for k in aggregated.keys()))
    arousals = sorted(set(k[2] for k in aggregated.keys()))
    
    # Engines que existen en los datos (excluir baseline)
    all_engines = sorted(set(k[0] for k in aggregated.keys()))
    transformer_engines = [e for e in all_engines if e != 'baseline']
    
    if not transformer_engines:
        logger.info("No hay engines transformer en los datos, skipping heatmaps")
        return
    
    # Métricas a graficar
    metrics = [
        ('mean_velocity_mean', 'velocity', 'Velocity (MIDI)'),
        ('pitch_range_mean', 'pitchrange', 'Pitch Range (semitonos)')
    ]
    
    for engine in transformer_engines:
        for metric_key, metric_name, metric_label in metrics:
            # Crear matriz
            matrix = np.zeros((len(arousals), len(valences)))
            
            for i, a in enumerate(arousals):
                for j, v in enumerate(valences):
                    key = (engine, v, a)
                    if key in aggregated:
                        value = aggregated[key].get(metric_key)
                        if value is not None:
                            matrix[i, j] = value
                        else:
                            matrix[i, j] = np.nan
                    else:
                        matrix[i, j] = np.nan
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(8, 6))
            
            im = ax.imshow(matrix, cmap='viridis', aspect='auto', origin='lower')
            
            # Configurar ejes
            ax.set_xticks(range(len(valences)))
            ax.set_yticks(range(len(arousals)))
            ax.set_xticklabels([f"{v:+.1f}" for v in valences])
            ax.set_yticklabels([f"{a:+.1f}" for a in arousals])
            ax.set_xlabel('Valence', fontsize=12)
            ax.set_ylabel('Arousal', fontsize=12)
            
            # Título
            engine_title = engine.replace('_', ' ').title()
            ax.set_title(f'{metric_label} - {engine_title}', fontsize=14, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric_label, fontsize=11)
            
            # Anotar valores en celdas
            for i in range(len(arousals)):
                for j in range(len(valences)):
                    value = matrix[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if value < (np.nanmin(matrix) + np.nanmax(matrix)) / 2 else 'black'
                        ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                               color=text_color, fontsize=9)
            
            plt.tight_layout()
            
            # Guardar
            filename = f"heatmap_{metric_name}_{engine}.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Heatmap guardado: {filename}")


def generate_latency_barplot(aggregated: Dict, output_dir: Path):
    """
    Genera gráfico de barras de latencia por engine.
    
    Args:
        aggregated: Dict con stats agregadas
        output_dir: Directorio donde guardar la figura
    """
    logger.info("Generando gráfico de latencia...")
    
    # Calcular latencia promedio y std por engine
    latency_by_engine = {}
    for engine in sorted(set(k[0] for k in aggregated.keys())):
        engine_latencies = [v['generation_time_ms_mean'] 
                           for k, v in aggregated.items() 
                           if k[0] == engine and v['generation_time_ms_mean'] is not None]
        
        if engine_latencies:
            mean_lat = statistics.mean(engine_latencies)
            std_lat = statistics.stdev(engine_latencies) if len(engine_latencies) > 1 else 0.0
            latency_by_engine[engine] = (mean_lat, std_lat)
    
    if not latency_by_engine:
        logger.warning("No hay datos de latencia para graficar")
        return
    
    # Preparar datos para el gráfico
    engines = list(latency_by_engine.keys())
    means = [latency_by_engine[e][0] for e in engines]
    stds = [latency_by_engine[e][1] for e in engines]
    
    # Nombres cortos
    engine_labels = [e.replace('transformer_', 'T-').replace('baseline', 'Baseline') for e in engines]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(engines))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8)
    
    # Configurar ejes
    ax.set_xticks(x_pos)
    ax.set_xticklabels(engine_labels, rotation=15, ha='right')
    ax.set_ylabel('Generation Time (ms)', fontsize=12)
    ax.set_title('Latencia de Generación por Engine', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Anotar valores encima de barras
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std, f'{mean:.0f}±{std:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Guardar
    filepath = output_dir / "bar_latency_by_engine.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Gráfico de latencia guardado: bar_latency_by_engine.png")


# ===== FUNCIÓN PRINCIPAL =====

def analyze_benchmark(results_dir: Path):
    """
    Analiza los resultados del benchmark y genera todos los outputs.
    
    Args:
        results_dir: Directorio con benchmark_raw.csv
    """
    logger.info("=" * 60)
    logger.info(f"ANÁLISIS DE BENCHMARK: {results_dir.name}")
    logger.info("=" * 60)
    
    # 1. Cargar datos
    csv_path = results_dir / "benchmark_raw.csv"
    data = load_benchmark_data(csv_path)
    
    if not data:
        logger.error("No hay datos para analizar")
        return
    
    # 2. Agregar datos
    aggregated = aggregate_by_engine_va(data)
    
    if not aggregated:
        logger.error("No se pudo agregar datos (todos los items fallaron?)")
        return
    
    # 3. Guardar CSV agregado
    agg_csv_path = results_dir / "benchmark_aggregated.csv"
    save_aggregated_csv(aggregated, agg_csv_path)
    
    # 3b. Agregar datos por (engine, bars) para escalabilidad
    aggregated_bars = aggregate_by_engine_bars(data)
    if aggregated_bars:
        bars_csv_path = results_dir / "benchmark_bars_aggregated.csv"
        save_bars_aggregated_csv(aggregated_bars, bars_csv_path)
    else:
        logger.info("No se encontraron datos con length_bars (benchmark antiguo sin escalabilidad)")
    
    # 4. Generar tabla LaTeX
    latex_path = results_dir / "benchmark_table.tex"
    generate_latex_table(aggregated, latex_path)
    
    # 5. Generar resumen de texto
    summary_path = results_dir / "benchmark_summary.txt"
    generate_summary_text(aggregated, data, summary_path)
    
    # 6. Generar heatmaps
    generate_heatmaps(aggregated, results_dir)
    
    # 7. Generar gráfico de latencia
    generate_latency_barplot(aggregated, results_dir)
    
    logger.info("=" * 60)
    logger.info("ANÁLISIS COMPLETADO")
    logger.info("=" * 60)
    logger.info(f"Archivos generados en: {results_dir}")
    logger.info("  - benchmark_aggregated.csv")
    logger.info("  - benchmark_bars_aggregated.csv (si hay datos con length_bars)")
    logger.info("  - benchmark_table.tex")
    logger.info("  - benchmark_summary.txt")
    logger.info("  - heatmap_velocity_transformer_pretrained.png")
    logger.info("  - heatmap_velocity_transformer_finetuned.png")
    logger.info("  - heatmap_pitchrange_transformer_pretrained.png")
    logger.info("  - heatmap_pitchrange_transformer_finetuned.png")
    logger.info("  - bar_latency_by_engine.png")
    logger.info("=" * 60)


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Analiza resultados del benchmark final",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo:
  python analyze_final_benchmark.py results/final_benchmark_20260218_120000
        """
    )
    
    parser.add_argument(
        'results_dir',
        type=str,
        help='Directorio con benchmark_raw.csv'
    )
    
    args = parser.parse_args()
    
    # Validar directorio
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Directorio no encontrado: {results_dir}")
        sys.exit(1)
    
    if not results_dir.is_dir():
        logger.error(f"No es un directorio: {results_dir}")
        sys.exit(1)
    
    csv_path = results_dir / "benchmark_raw.csv"
    if not csv_path.exists():
        logger.error(f"benchmark_raw.csv no encontrado en {results_dir}")
        sys.exit(1)
    
    # Ejecutar análisis
    try:
        analyze_benchmark(results_dir)
    except Exception as e:
        logger.error(f"Error en análisis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
