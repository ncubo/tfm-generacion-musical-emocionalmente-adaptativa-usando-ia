#!/usr/bin/env python3
"""
Script para analizar resultados de comparación pretrained vs finetuned.

Lee compare_raw.csv generado por compare_pretrained_vs_finetuned.py, agrega resultados
por (model_tag, valence, arousal) calculando mean ± std, y exporta en múltiples formatos.

Usage:
    python analyze_compare_results.py [options]
    
    Options:
        --input_csv PATH              CSV de entrada (default: results/compare_pre_vs_finetuned/compare_raw.csv)
        --output_dir PATH             Directorio de salida (default: deriva del input_csv)

Output:
    - compare_aggregated.csv          Tabla agregada con mean ± std por (model, V, A)
    - compare_table.tex               Tabla LaTeX compacta para TFM
    - compare_summary.txt             Resumen textual en español (5-8 bullets)
"""

import sys
import csv
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import logging
import numpy as np

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_results_csv(csv_path: Path) -> List[Dict]:
    """
    Carga resultados desde CSV.
    
    Args:
        csv_path: Path al CSV de entrada
        
    Returns:
        Lista de dicts con resultados
    """
    logger.info(f"Cargando resultados desde: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convertir campos numéricos
            for key in row:
                if key in ['valence', 'arousal', 'seed', 'note_density', 'pitch_range',
                          'mean_velocity', 'mean_note_duration', 'total_notes',
                          'total_duration_seconds', 'min_pitch', 'max_pitch', 'unique_pitches']:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, KeyError):
                        pass
            results.append(row)
    
    logger.info(f"Cargados {len(results)} resultados")
    return results


def aggregate_results(results: List[Dict]) -> List[Dict]:
    """
    Agrega resultados por (model_tag, valence, arousal) con mean ± std.
    
    Args:
        results: Resultados crudos
        
    Returns:
        Lista de dicts agregados con mean y std por cada métrica
    """
    # Filtrar resultados válidos (sin errores)
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        logger.warning("No hay resultados válidos para agregar")
        return []
    
    # Agrupar por (model_tag, valence, arousal)
    groups = defaultdict(list)
    
    for result in valid_results:
        key = (result['model_tag'], result['valence'], result['arousal'])
        groups[key].append(result)
    
    # Métricas a agregar
    metrics = [
        'note_density', 'pitch_range', 'mean_velocity', 'mean_note_duration',
        'total_notes', 'total_duration_seconds', 'unique_pitches'
    ]
    
    # Calcular estadísticas para cada grupo
    aggregated = []
    
    for (model_tag, valence, arousal), group_results in sorted(groups.items()):
        agg_row = {
            'model_tag': model_tag,
            'valence': valence,
            'arousal': arousal,
            'num_samples': len(group_results)
        }
        
        # Calcular mean y std para cada métrica
        for metric in metrics:
            values = [r[metric] for r in group_results if metric in r]
            
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                
                agg_row[f'{metric}_mean'] = round(mean_val, 3)
                agg_row[f'{metric}_std'] = round(std_val, 3)
            else:
                agg_row[f'{metric}_mean'] = 0.0
                agg_row[f'{metric}_std'] = 0.0
        
        aggregated.append(agg_row)
    
    logger.info(f"Resultados agregados: {len(aggregated)} combinaciones (model, V, A)")
    
    return aggregated


def save_aggregated_csv(aggregated: List[Dict], output_path: Path):
    """
    Guarda resultados agregados en CSV.
    
    Args:
        aggregated: Resultados agregados
        output_path: Path de salida
    """
    if not aggregated:
        logger.warning("No hay resultados agregados para guardar")
        return
    
    logger.info(f"Guardando CSV agregado: {output_path}")
    
    # Obtener todas las columnas
    fieldnames = list(aggregated[0].keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregated)
    
    logger.info(f"CSV agregado guardado: {output_path} ({len(aggregated)} filas)")


def generate_latex_table(aggregated: List[Dict], output_path: Path):
    """
    Genera tabla LaTeX compacta para TFM.
    
    Formato: Comparación lado a lado pretrained vs finetuned
    
    Args:
        aggregated: Resultados agregados
        output_path: Path de salida .tex
    """
    logger.info(f"Generando tabla LaTeX: {output_path}")
    
    # Agrupar por (valence, arousal) para comparar modelos lado a lado
    va_groups = defaultdict(dict)
    
    for row in aggregated:
        key = (row['valence'], row['arousal'])
        model = row['model_tag']
        va_groups[key][model] = row
    
    # Generar tabla LaTeX
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparación Pretrained vs Finetuned: Métricas Musicales}")
    lines.append(r"\label{tab:compare_models}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{cc|ccc|ccc}")
    lines.append(r"\hline")
    lines.append(r"\multicolumn{2}{c|}{\textbf{VA}} & \multicolumn{3}{c|}{\textbf{Pretrained}} & \multicolumn{3}{c}{\textbf{Finetuned}} \\")
    lines.append(r"\textbf{V} & \textbf{A} & \textbf{Density} & \textbf{Pitch} & \textbf{Vel} & \textbf{Density} & \textbf{Pitch} & \textbf{Vel} \\")
    lines.append(r"\hline")
    
    # Ordenar por arousal y luego valence
    for (valence, arousal) in sorted(va_groups.keys(), key=lambda x: (x[1], x[0])):
        models = va_groups[(valence, arousal)]
        
        pretrained = models.get('pretrained', {})
        finetuned = models.get('finetuned', {})
        
        # Formatear valores
        v_str = f"{valence:+.1f}"
        a_str = f"{arousal:+.1f}"
        
        # Pretrained
        pre_density = f"{pretrained.get('note_density_mean', 0):.2f}"
        pre_pitch = f"{pretrained.get('pitch_range_mean', 0):.1f}"
        pre_vel = f"{pretrained.get('mean_velocity_mean', 0):.1f}"
        
        # Finetuned
        fin_density = f"{finetuned.get('note_density_mean', 0):.2f}"
        fin_pitch = f"{finetuned.get('pitch_range_mean', 0):.1f}"
        fin_vel = f"{finetuned.get('mean_velocity_mean', 0):.1f}"
        
        line = f"{v_str} & {a_str} & {pre_density} & {pre_pitch} & {pre_vel} & {fin_density} & {fin_pitch} & {fin_vel} \\\\"
        lines.append(line)
    
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    # Guardar
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Tabla LaTeX guardada: {output_path}")


def generate_summary_text(aggregated: List[Dict], raw_results: List[Dict], output_path: Path):
    """
    Genera resumen textual en español con hallazgos clave.
    
    Args:
        aggregated: Resultados agregados
        raw_results: Resultados crudos
        output_path: Path de salida .txt
    """
    logger.info(f"Generando resumen textual: {output_path}")
    
    lines = []
    lines.append("=" * 80)
    lines.append("RESUMEN: COMPARACIÓN PRETRAINED VS FINETUNED")
    lines.append("=" * 80)
    lines.append("")
    
    # Estadísticas generales
    pretrained_results = [r for r in aggregated if r['model_tag'] == 'pretrained']
    finetuned_results = [r for r in aggregated if r['model_tag'] == 'finetuned']
    
    lines.append(f"Total de combinaciones VA evaluadas: {len(pretrained_results)}")
    lines.append(f"Muestras por combinación: {aggregated[0]['num_samples'] if aggregated else 0}")
    lines.append("")
    
    # Comparación de métricas clave
    lines.append("HALLAZGOS PRINCIPALES:")
    lines.append("")
    
    # 1. Note Density
    pre_density = [r['note_density_mean'] for r in pretrained_results]
    fin_density = [r['note_density_mean'] for r in finetuned_results]
    
    pre_density_avg = np.mean(pre_density)
    fin_density_avg = np.mean(fin_density)
    density_diff = ((fin_density_avg - pre_density_avg) / pre_density_avg) * 100
    
    lines.append(f"1. DENSIDAD DE NOTAS:")
    lines.append(f"   - Pretrained: {pre_density_avg:.2f} notas/s (promedio)")
    lines.append(f"   - Finetuned: {fin_density_avg:.2f} notas/s (promedio)")
    lines.append(f"   - Diferencia: {density_diff:+.1f}%")
    if abs(density_diff) > 5:
        lines.append(f"   → {'Finetuned genera MIDIs más densos' if density_diff > 0 else 'Pretrained genera MIDIs más densos'}")
    else:
        lines.append(f"   → Densidad similar entre ambos modelos")
    lines.append("")
    
    # 2. Pitch Range
    pre_pitch = [r['pitch_range_mean'] for r in pretrained_results]
    fin_pitch = [r['pitch_range_mean'] for r in finetuned_results]
    
    pre_pitch_avg = np.mean(pre_pitch)
    fin_pitch_avg = np.mean(fin_pitch)
    pitch_diff = fin_pitch_avg - pre_pitch_avg
    
    lines.append(f"2. RANGO DE PITCH:")
    lines.append(f"   - Pretrained: {pre_pitch_avg:.1f} semitonos (promedio)")
    lines.append(f"   - Finetuned: {fin_pitch_avg:.1f} semitonos (promedio)")
    lines.append(f"   - Diferencia: {pitch_diff:+.1f} semitonos")
    if abs(pitch_diff) > 3:
        lines.append(f"   → {'Finetuned usa rangos más amplios' if pitch_diff > 0 else 'Pretrained usa rangos más amplios'}")
    else:
        lines.append(f"   → Rango de pitch similar entre ambos modelos")
    lines.append("")
    
    # 3. Mean Velocity
    pre_vel = [r['mean_velocity_mean'] for r in pretrained_results]
    fin_vel = [r['mean_velocity_mean'] for r in finetuned_results]
    
    pre_vel_avg = np.mean(pre_vel)
    fin_vel_avg = np.mean(fin_vel)
    vel_diff = fin_vel_avg - pre_vel_avg
    
    lines.append(f"3. VELOCIDAD MEDIA:")
    lines.append(f"   - Pretrained: {pre_vel_avg:.1f} (promedio)")
    lines.append(f"   - Finetuned: {fin_vel_avg:.1f} (promedio)")
    lines.append(f"   - Diferencia: {vel_diff:+.1f}")
    if abs(vel_diff) > 5:
        lines.append(f"   → {'Finetuned genera dinámicas más fuertes' if vel_diff > 0 else 'Pretrained genera dinámicas más fuertes'}")
    else:
        lines.append(f"   → Velocidades similares entre ambos modelos")
    lines.append("")
    
    # 4. Variabilidad (std promedio)
    pre_density_stds = [r['note_density_std'] for r in pretrained_results]
    fin_density_stds = [r['note_density_std'] for r in finetuned_results]
    
    pre_std_avg = np.mean(pre_density_stds)
    fin_std_avg = np.mean(fin_density_stds)
    
    lines.append(f"4. CONSISTENCIA (variabilidad entre seeds):")
    lines.append(f"   - Pretrained: std promedio = {pre_std_avg:.3f}")
    lines.append(f"   - Finetuned: std promedio = {fin_std_avg:.3f}")
    if fin_std_avg < pre_std_avg * 0.8:
        lines.append(f"   → Finetuned es más consistente (menor variabilidad)")
    elif fin_std_avg > pre_std_avg * 1.2:
        lines.append(f"   → Pretrained es más consistente (menor variabilidad)")
    else:
        lines.append(f"   → Ambos modelos tienen variabilidad similar")
    lines.append("")
    
    # 5. Total de notas
    pre_notes = [r['total_notes_mean'] for r in pretrained_results]
    fin_notes = [r['total_notes_mean'] for r in finetuned_results]
    
    pre_notes_avg = np.mean(pre_notes)
    fin_notes_avg = np.mean(fin_notes)
    
    lines.append(f"5. TOTAL DE NOTAS POR MIDI:")
    lines.append(f"   - Pretrained: {pre_notes_avg:.1f} notas (promedio)")
    lines.append(f"   - Finetuned: {fin_notes_avg:.1f} notas (promedio)")
    notes_diff_pct = ((fin_notes_avg - pre_notes_avg) / pre_notes_avg) * 100
    lines.append(f"   - Diferencia: {notes_diff_pct:+.1f}%")
    lines.append("")
    
    # Conclusión
    lines.append("=" * 80)
    lines.append("CONCLUSIÓN:")
    lines.append("")
    
    # Determinar modelo "ganador" basado en métricas clave
    if abs(density_diff) < 5 and abs(pitch_diff) < 3 and abs(vel_diff) < 5:
        lines.append("Los modelos pretrained y finetuned muestran características musicales muy")
        lines.append("similares, indicando que el fine-tuning preserva el comportamiento base del")
        lines.append("modelo sin introducir cambios drásticos.")
    else:
        lines.append("Se observan diferencias significativas entre pretrained y finetuned en:")
        if abs(density_diff) > 10:
            lines.append(f"  - Densidad de notas ({abs(density_diff):.1f}% diferencia)")
        if abs(pitch_diff) > 5:
            lines.append(f"  - Rango de pitch ({abs(pitch_diff):.1f} semitonos diferencia)")
        if abs(vel_diff) > 8:
            lines.append(f"  - Velocidad media ({abs(vel_diff):.1f} puntos diferencia)")
    
    lines.append("")
    lines.append("=" * 80)
    
    # Guardar
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Resumen guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analizar resultados de comparación pretrained vs finetuned'
    )
    
    parser.add_argument(
        '--input_csv',
        default='results/compare_pre_vs_finetuned/compare_raw.csv',
        help='CSV de entrada (default: results/compare_pre_vs_finetuned/compare_raw.csv)'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Directorio de salida (default: deriva del input_csv)'
    )
    
    args = parser.parse_args()
    
    # Paths
    backend_dir = Path(__file__).parent.parent
    input_csv = backend_dir / args.input_csv
    
    if args.output_dir:
        output_dir = backend_dir / args.output_dir
    else:
        output_dir = input_csv.parent
    
    # Cargar resultados
    results = load_results_csv(input_csv)
    
    # Agregar por (model, V, A)
    aggregated = aggregate_results(results)
    
    # Guardar CSV agregado
    aggregated_csv = output_dir / "compare_aggregated.csv"
    save_aggregated_csv(aggregated, aggregated_csv)
    
    # Generar tabla LaTeX
    table_tex = output_dir / "compare_table.tex"
    generate_latex_table(aggregated, table_tex)
    
    # Generar resumen textual
    summary_txt = output_dir / "compare_summary.txt"
    generate_summary_text(aggregated, results, summary_txt)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANÁLISIS COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"Archivos generados en: {output_dir}")
    logger.info(f"  - compare_aggregated.csv: Tabla agregada con mean ± std")
    logger.info(f"  - compare_table.tex: Tabla LaTeX para TFM")
    logger.info(f"  - compare_summary.txt: Resumen en español con hallazgos")
    logger.info("")
    
    return 0


if __name__ == '__main__':
    exit(main())
