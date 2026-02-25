#!/usr/bin/env python3
"""
Comparación de estadísticas: Dataset Original vs Música Generada.

Este script compara las distribuciones de características musicales entre
el dataset de entrenamiento/referencia y los MIDIs generados por el sistema.

Uso:
    python compare_dataset_vs_generated.py --dataset_dir data/lakh_piano_clean --generated_dir output
    python compare_dataset_vs_generated.py --dataset_dir data/lakh_piano_clean --generated_dir results/final_benchmark_20260223/baseline

Output:
    - comparison_statistics.csv (tabla comparativa)
    - comparison_statistics.tex (tabla LaTeX)
    - comparison_summary.json (resumen con KL divergence opcional)
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import csv
import json
from datetime import datetime

# Añadir backend/src al path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

import numpy as np
from scipy.stats import entropy

# Imports internos
from core.music.analysis.features import extract_midi_features

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_features_from_directory(midi_dir: Path, max_files: int = None) -> Dict[str, List[float]]:
    """
    Extrae características de todos los MIDIs en un directorio.
    
    Args:
        midi_dir: Directorio con archivos MIDI
        max_files: Número máximo de archivos a procesar (None = todos)
        
    Returns:
        Dict con listas de valores por métrica
    """
    midi_paths = list(midi_dir.glob('**/*.mid')) + list(midi_dir.glob('**/*.midi'))
    
    if max_files:
        midi_paths = midi_paths[:max_files]
    
    logger.info(f"Extrayendo features de {len(midi_paths)} MIDIs desde {midi_dir}...")
    
    features = {
        'note_density': [],
        'pitch_range': [],
        'mean_velocity': []
    }
    
    processed = 0
    errors = 0
    
    for i, midi_path in enumerate(midi_paths):
        try:
            result = extract_midi_features(str(midi_path))
            
            features['note_density'].append(result['note_density'])
            features['pitch_range'].append(result['pitch_range'])
            features['mean_velocity'].append(result['mean_velocity'])
            
            processed += 1
            
            if (i + 1) % 100 == 0 or (i + 1) == len(midi_paths):
                logger.info(f"  Procesados: {i+1}/{len(midi_paths)}")
                
        except Exception as e:
            errors += 1
            if errors <= 5:  # Solo mostrar primeros 5 errores
                logger.warning(f"  Error en {midi_path.name}: {e}")
    
    logger.info(f"[OK] Features extraídas: {processed} OK, {errors} errores\n")
    
    return features


def calculate_statistics(features: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Calcula media y desviación estándar para cada métrica.
    
    Args:
        features: Dict con listas de valores
        
    Returns:
        Dict con estadísticas por métrica
    """
    stats = {}
    
    for metric_name, values in features.items():
        if not values:
            stats[metric_name] = {'mean': 0.0, 'std': 0.0}
            continue
        
        stats[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'count': len(values)
        }
    
    return stats


def calculate_kl_divergence(dataset_values: List[float], generated_values: List[float], bins: int = 30) -> float:
    """
    Calcula KL divergence entre dos distribuciones.
    
    KL(P||Q) mide cuánta información se pierde al aproximar P con Q.
    
    Args:
        dataset_values: Valores del dataset
        generated_values: Valores generados
        bins: Número de bins para el histograma
        
    Returns:
        KL divergence (float)
    """
    if not dataset_values or not generated_values:
        return float('inf')
    
    # Determinar rango común
    min_val = min(min(dataset_values), min(generated_values))
    max_val = max(max(dataset_values), max(generated_values))
    
    # Crear histogramas normalizados
    hist_dataset, bin_edges = np.histogram(dataset_values, bins=bins, range=(min_val, max_val), density=True)
    hist_generated, _ = np.histogram(generated_values, bins=bins, range=(min_val, max_val), density=True)
    
    # Normalizar a probabilidades (sumar 1)
    hist_dataset = hist_dataset / np.sum(hist_dataset)
    hist_generated = hist_generated / np.sum(hist_generated)
    
    # Añadir epsilon para evitar log(0)
    epsilon = 1e-10
    hist_dataset = hist_dataset + epsilon
    hist_generated = hist_generated + epsilon
    
    # Calcular KL divergence
    kl_div = entropy(hist_dataset, hist_generated)
    
    return float(kl_div)


def calculate_percentage_difference(dataset_mean: float, generated_mean: float) -> float:
    """
    Calcula diferencia porcentual entre dos valores.
    
    Args:
        dataset_mean: Media del dataset
        generated_mean: Media generada
        
    Returns:
        Diferencia porcentual (%)
    """
    if dataset_mean == 0:
        return 0.0
    
    return ((generated_mean - dataset_mean) / dataset_mean) * 100


def save_comparison_csv(
    dataset_stats: Dict,
    generated_stats: Dict,
    kl_divergences: Dict[str, float],
    output_path: Path
):
    """
    Guarda tabla comparativa en CSV.
    
    Args:
        dataset_stats: Estadísticas del dataset
        generated_stats: Estadísticas de generados
        kl_divergences: KL divergences por métrica
        output_path: Path al archivo CSV
    """
    rows = []
    
    for metric in ['note_density', 'pitch_range', 'mean_velocity']:
        dataset = dataset_stats.get(metric, {})
        generated = generated_stats.get(metric, {})
        
        diff_pct = calculate_percentage_difference(
            dataset.get('mean', 0),
            generated.get('mean', 0)
        )
        
        rows.append({
            'metric': metric,
            'dataset_mean': round(dataset.get('mean', 0), 3),
            'dataset_std': round(dataset.get('std', 0), 3),
            'generated_mean': round(generated.get('mean', 0), 3),
            'generated_std': round(generated.get('std', 0), 3),
            'diff_percent': round(diff_pct, 2),
            'kl_divergence': round(kl_divergences.get(metric, 0), 4)
        })
    
    # Escribir CSV
    fieldnames = ['metric', 'dataset_mean', 'dataset_std', 'generated_mean', 
                  'generated_std', 'diff_percent', 'kl_divergence']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"CSV guardado: {output_path}")


def save_comparison_latex(
    dataset_stats: Dict,
    generated_stats: Dict,
    kl_divergences: Dict[str, float],
    output_path: Path
):
    """
    Genera tabla LaTeX comparativa.
    
    Args:
        dataset_stats: Estadísticas del dataset
        generated_stats: Estadísticas de generados
        kl_divergences: KL divergences por métrica
        output_path: Path al archivo .tex
    """
    lines = []
    
    lines.append("% Tabla generada automáticamente por compare_dataset_vs_generated.py")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Comparación: Dataset Original vs Música Generada}")
    lines.append("\\label{tab:dataset_vs_generated_comparison}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l|cc|cc|c|c}")
    lines.append("\\hline")
    lines.append("\\multirow{2}{*}{\\textbf{Metric}} & \\multicolumn{2}{c|}{\\textbf{Dataset}} & \\multicolumn{2}{c|}{\\textbf{Generated}} & \\textbf{Diff} & \\textbf{KL} \\\\")
    lines.append("                                     & Mean & Std & Mean & Std & (\\%) & Div \\\\")
    lines.append("\\hline")
    
    # Nombres legibles de métricas
    metric_names = {
        'note_density': 'Note Density (n/s)',
        'pitch_range': 'Pitch Range (semitones)',
        'mean_velocity': 'Mean Velocity (MIDI)'
    }
    
    for metric in ['note_density', 'pitch_range', 'mean_velocity']:
        dataset = dataset_stats.get(metric, {})
        generated = generated_stats.get(metric, {})
        
        dataset_mean = dataset.get('mean', 0)
        dataset_std = dataset.get('std', 0)
        generated_mean = generated.get('mean', 0)
        generated_std = generated.get('std', 0)
        
        diff_pct = calculate_percentage_difference(dataset_mean, generated_mean)
        kl_div = kl_divergences.get(metric, 0)
        
        metric_label = metric_names[metric]
        
        lines.append(
            f"{metric_label} & {dataset_mean:.2f} & {dataset_std:.2f} & "
            f"{generated_mean:.2f} & {generated_std:.2f} & {diff_pct:+.1f} & {kl_div:.3f} \\\\"
        )
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Tabla LaTeX guardada: {output_path}")


def save_comparison_summary_json(
    dataset_stats: Dict,
    generated_stats: Dict,
    kl_divergences: Dict[str, float],
    output_path: Path
):
    """
    Guarda resumen completo en JSON.
    
    Args:
        dataset_stats: Estadísticas del dataset
        generated_stats: Estadísticas de generados
        kl_divergences: KL divergences por métrica
        output_path: Path al archivo JSON
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset_statistics': dataset_stats,
        'generated_statistics': generated_stats,
        'comparison': {}
    }
    
    for metric in ['note_density', 'pitch_range', 'mean_velocity']:
        dataset_mean = dataset_stats.get(metric, {}).get('mean', 0)
        generated_mean = generated_stats.get(metric, {}).get('mean', 0)
        
        summary['comparison'][metric] = {
            'difference_percent': round(
                calculate_percentage_difference(dataset_mean, generated_mean),
                2
            ),
            'kl_divergence': round(kl_divergences.get(metric, 0), 4)
        }
    
    # Añadir interpretación
    summary['interpretation'] = {
        'note': 'KL divergence mide cuánta información se pierde al aproximar el dataset con generated',
        'values': {
            'close': 'KL < 0.1 (distribuciones muy similares)',
            'moderate': '0.1 <= KL < 0.5 (diferencias moderadas)',
            'distant': 'KL >= 0.5 (distribuciones distintas)'
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resumen JSON guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Comparación Dataset Original vs Música Generada'
    )
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        required=True,
        help='Directorio con MIDIs del dataset original'
    )
    parser.add_argument(
        '--generated_dir',
        type=Path,
        required=True,
        help='Directorio con MIDIs generados'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('results/dataset_comparison'),
        help='Directorio de salida para resultados'
    )
    parser.add_argument(
        '--max_dataset_files',
        type=int,
        default=None,
        help='Máximo de archivos del dataset a procesar (None = todos)'
    )
    parser.add_argument(
        '--max_generated_files',
        type=int,
        default=None,
        help='Máximo de archivos generados a procesar (None = todos)'
    )
    
    args = parser.parse_args()
    
    # Validar directorios
    if not args.dataset_dir.exists():
        logger.error(f"Directorio dataset no existe: {args.dataset_dir}")
        sys.exit(1)
    
    if not args.generated_dir.exists():
        logger.error(f"Directorio generated no existe: {args.generated_dir}")
        sys.exit(1)
    
    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("COMPARACIÓN: DATASET ORIGINAL vs MÚSICA GENERADA")
    logger.info("="*70)
    logger.info(f"Dataset dir:   {args.dataset_dir}")
    logger.info(f"Generated dir: {args.generated_dir}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info("="*70 + "\n")
    
    try:
        # 1. Extraer features del dataset
        logger.info("[1/5] Extrayendo features del dataset...")
        dataset_features = extract_features_from_directory(
            args.dataset_dir,
            max_files=args.max_dataset_files
        )
        
        # 2. Extraer features de generados
        logger.info("[2/5] Extrayendo features de generados...")
        generated_features = extract_features_from_directory(
            args.generated_dir,
            max_files=args.max_generated_files
        )
        
        # 3. Calcular estadísticas
        logger.info("[3/5] Calculando estadísticas...")
        dataset_stats = calculate_statistics(dataset_features)
        generated_stats = calculate_statistics(generated_features)
        
        # Mostrar estadísticas
        logger.info("\nESTADÍSTICAS DATASET:")
        for metric, stats in dataset_stats.items():
            logger.info(f"  {metric:20s}: mean={stats['mean']:.3f} ± std={stats['std']:.3f} (n={stats['count']})")
        
        logger.info("\nESTADÍSTICAS GENERATED:")
        for metric, stats in generated_stats.items():
            logger.info(f"  {metric:20s}: mean={stats['mean']:.3f} ± std={stats['std']:.3f} (n={stats['count']})")
        
        # 4. Calcular KL divergences
        logger.info("\n[4/5] Calculando KL divergences...")
        kl_divergences = {}
        
        for metric in ['note_density', 'pitch_range', 'mean_velocity']:
            kl_div = calculate_kl_divergence(
                dataset_features[metric],
                generated_features[metric]
            )
            kl_divergences[metric] = kl_div
            logger.info(f"  {metric:20s}: KL = {kl_div:.4f}")
        
        # 5. Exportar resultados
        logger.info("\n[5/5] Guardando resultados...")
        
        save_comparison_csv(
            dataset_stats, generated_stats, kl_divergences,
            args.output_dir / 'comparison_statistics.csv'
        )
        
        save_comparison_latex(
            dataset_stats, generated_stats, kl_divergences,
            args.output_dir / 'comparison_statistics.tex'
        )
        
        save_comparison_summary_json(
            dataset_stats, generated_stats, kl_divergences,
            args.output_dir / 'comparison_summary.json'
        )
        
        logger.info("\n" + "="*70)
        logger.info("COMPARACIÓN COMPLETADA")
        logger.info("="*70)
        logger.info(f"Resultados guardados en: {args.output_dir}")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Error durante la comparación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
