#!/usr/bin/env python3
"""
Evaluación Dimensional: Alineación entre VA Target y VA Estimado desde Música.

Este script evalúa qué tan bien el sistema genera música cuyas características
musicales se alinean con los parámetros emocionales (valence, arousal) objetivo.

Proceso:
1. Cargar benchmark_raw.csv con VA targets y paths a MIDIs generados
2. Extraer características musicales de cada MIDI
3. Estimar VA desde características musicales usando compute_va_heuristic()
4. Calcular correlación de Pearson entre VA target y VA estimado
5. Generar scatter plots y reportes

Uso:
    python evaluate_dimensional_alignment.py --benchmark_results results/final_benchmark_20260223_120200
    python evaluate_dimensional_alignment.py --benchmark_results results/final_benchmark_20260223_120200 --target_engine transformer_finetuned

Output:
    - dimensional_alignment.csv (tabla de correlaciones)
    - dimensional_alignment.tex (tabla LaTeX)
    - dimensional_alignment.json (detalles completos)
    - valence_alignment_scatter.png
    - arousal_alignment_scatter.png
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import json
from datetime import datetime

# Añadir backend/src al path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from core.music.analysis.features import extract_midi_features

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def compute_va_heuristic(features: Dict) -> Tuple[float, float]:
    """
    Estima valence y arousal desde características musicales.
    
    Implementación basada en scripts/generate_va_labels.py.
    
    Heurística:
    - Valence: Correlaciona con mean_velocity (mayor velocity = más positivo)
    - Arousal: Correlaciona con note_density (mayor densidad = mayor arousal)
    
    Args:
        features: Dict con note_density, mean_velocity, etc.
        
    Returns:
        Tupla (valence_estimated, arousal_estimated) normalizados a [-1, 1]
    """
    # Extraer métricas
    note_density = features.get('note_density', 0.0)
    mean_velocity = features.get('mean_velocity', 64.0)
    pitch_range = features.get('pitch_range', 12)
    
    # Valence: basado en velocity
    # Rango MIDI: 0-127, normalizar a [-1, 1]
    valence = (mean_velocity / 127.0) * 2.0 - 1.0
    
    # Arousal: basado en note_density y pitch_range
    # Densidad típica: 0.5-5.0 notes/sec
    # Pitch range típico: 12-36 semitonos
    arousal_density = np.clip(note_density / 5.0, 0.0, 1.0)
    arousal_pitch = np.clip(pitch_range / 48.0, 0.0, 1.0)
    
    # Combinar (60% densidad, 40% pitch range)
    arousal_raw = 0.6 * arousal_density + 0.4 * arousal_pitch
    arousal = arousal_raw * 2.0 - 1.0  # Normalizar a [-1, 1]
    
    return valence, arousal


def load_benchmark_raw_csv(csv_path: Path) -> List[Dict]:
    """
    Carga benchmark_raw.csv con resultados de generación.
    
    Args:
        csv_path: Path al benchmark_raw.csv
        
    Returns:
        Lista de dicts con resultados
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
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parseando fila: {e}")
                continue
            
            data.append(row)
    
    logger.info(f"Benchmark raw cargado: {len(data)} filas")
    return data


def evaluate_dimensional_alignment(
    data: List[Dict],
    target_engine: str,
    base_dir: Path
) -> Dict:
    """
    Evalúa alineación dimensional entre VA target y VA estimado.
    
    Args:
        data: Datos del benchmark_raw.csv
        target_engine: Engine a evaluar
        base_dir: Directorio base para resolver paths relativos
        
    Returns:
        Dict con resultados de evaluación
    """
    logger.info(f"Evaluando {target_engine}...")
    
    # Filtrar datos del engine target con status success
    engine_data = [
        row for row in data
        if row.get('engine') == target_engine 
        and row.get('status') == 'success'
        and row.get('midi_path')
    ]
    
    if not engine_data:
        raise ValueError(f"No hay datos para {target_engine}")
    
    logger.info(f"  Procesando {len(engine_data)} generaciones exitosas...")
    
    # Listas para almacenar VA target y VA estimated
    valence_targets = []
    valence_estimated = []
    arousal_targets = []
    arousal_estimated = []
    
    skipped = 0
    for i, row in enumerate(engine_data, 1):
        # Resolver path al MIDI
        midi_rel_path = row['midi_path']
        midi_path = base_dir / midi_rel_path
        
        if not midi_path.exists():
            logger.warning(f"  MIDI no encontrado: {midi_path}")
            skipped += 1
            continue
        
        # Extraer características musicales
        try:
            features = extract_midi_features(str(midi_path))
        except Exception as e:
            logger.warning(f"  Error extrayendo features de {midi_path.name}: {e}")
            skipped += 1
            continue
        
        # Estimar VA desde características
        valence_est, arousal_est = compute_va_heuristic(features)
        
        # Almacenar
        valence_targets.append(row['valence'])
        valence_estimated.append(valence_est)
        arousal_targets.append(row['arousal'])
        arousal_estimated.append(arousal_est)
        
        if i % 10 == 0:
            logger.info(f"  Procesados: {i}/{len(engine_data)}")
    
    if skipped > 0:
        logger.warning(f"  Archivos omitidos: {skipped}")
    
    if len(valence_targets) < 3:
        raise ValueError(f"Insuficientes datos válidos (n={len(valence_targets)})")
    
    logger.info(f"  Datos válidos: {len(valence_targets)}")
    
    # Calcular correlaciones de Pearson
    logger.info("  Calculando correlaciones de Pearson...")
    
    valence_r, valence_p = pearsonr(valence_targets, valence_estimated)
    arousal_r, arousal_p = pearsonr(arousal_targets, arousal_estimated)
    
    # Calcular errores
    valence_mae = np.mean(np.abs(np.array(valence_targets) - np.array(valence_estimated)))
    valence_rmse = np.sqrt(np.mean((np.array(valence_targets) - np.array(valence_estimated))**2))
    
    arousal_mae = np.mean(np.abs(np.array(arousal_targets) - np.array(arousal_estimated)))
    arousal_rmse = np.sqrt(np.mean((np.array(arousal_targets) - np.array(arousal_estimated))**2))
    
    results = {
        'engine': target_engine,
        'num_samples': len(valence_targets),
        'valence': {
            'pearson_r': float(valence_r),
            'pearson_p': float(valence_p),
            'mae': float(valence_mae),
            'rmse': float(valence_rmse),
            'r_squared': float(valence_r**2),
            'targets': valence_targets,
            'estimated': valence_estimated,
            'interpretation': _interpret_correlation(valence_r)
        },
        'arousal': {
            'pearson_r': float(arousal_r),
            'pearson_p': float(arousal_p),
            'mae': float(arousal_mae),
            'rmse': float(arousal_rmse),
            'r_squared': float(arousal_r**2),
            'targets': arousal_targets,
            'estimated': arousal_estimated,
            'interpretation': _interpret_correlation(arousal_r)
        }
    }
    
    logger.info(f"\n  VALENCE:")
    logger.info(f"    Pearson r = {valence_r:+.3f} (p = {valence_p:.3e})")
    logger.info(f"    R²        = {valence_r**2:.3f}")
    logger.info(f"    MAE       = {valence_mae:.3f}")
    logger.info(f"    RMSE      = {valence_rmse:.3f}")
    logger.info(f"    {results['valence']['interpretation']}")
    
    logger.info(f"\n  AROUSAL:")
    logger.info(f"    Pearson r = {arousal_r:+.3f} (p = {arousal_p:.3e})")
    logger.info(f"    R²        = {arousal_r**2:.3f}")
    logger.info(f"    MAE       = {arousal_mae:.3f}")
    logger.info(f"    RMSE      = {arousal_rmse:.3f}")
    logger.info(f"    {results['arousal']['interpretation']}\n")
    
    return results


def _interpret_correlation(r: float) -> str:
    """Interpreta la correlación de Pearson."""
    r_abs = abs(r)
    if r_abs >= 0.7:
        strength = "STRONG"
    elif r_abs >= 0.5:
        strength = "MODERATE"
    elif r_abs >= 0.3:
        strength = "WEAK"
    else:
        strength = "VERY WEAK"
    
    direction = "positive" if r >= 0 else "negative"
    return f"{strength} {direction} correlation"


def save_alignment_csv(results: Dict, output_path: Path):
    """
    Guarda tabla de correlaciones en CSV.
    
    Args:
        results: Dict con resultados de evaluación
        output_path: Path al archivo CSV
    """
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Dimension', 'Pearson_r', 'p_value', 'R_squared',
            'MAE', 'RMSE', 'Interpretation'
        ])
        
        # Valence
        v = results['valence']
        writer.writerow([
            'Valence',
            f"{v['pearson_r']:.3f}",
            f"{v['pearson_p']:.3e}",
            f"{v['r_squared']:.3f}",
            f"{v['mae']:.3f}",
            f"{v['rmse']:.3f}",
            v['interpretation']
        ])
        
        # Arousal
        a = results['arousal']
        writer.writerow([
            'Arousal',
            f"{a['pearson_r']:.3f}",
            f"{a['pearson_p']:.3e}",
            f"{a['r_squared']:.3f}",
            f"{a['mae']:.3f}",
            f"{a['rmse']:.3f}",
            a['interpretation']
        ])
    
    logger.info(f"CSV guardado: {output_path}")


def save_alignment_latex(results: Dict, output_path: Path):
    """
    Guarda tabla LaTeX con correlaciones dimensionales.
    
    Args:
        results: Dict con resultados de evaluación
        output_path: Path al archivo .tex
    """
    lines = []
    
    lines.append("% Tabla generada automáticamente por evaluate_dimensional_alignment.py")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Dimensional Alignment - Correlaciones VA Target vs Estimated}")
    lines.append("\\label{tab:dimensional_alignment}")
    lines.append("\\begin{tabular}{l|c|c|c|c|c}")
    lines.append("\\hline")
    lines.append("\\textbf{Dimension} & \\textbf{Pearson r} & \\textbf{$R^2$} & \\textbf{p-value} & \\textbf{MAE} & \\textbf{RMSE} \\\\")
    lines.append("\\hline")
    
    # Valence
    v = results['valence']
    lines.append(
        f"Valence & {v['pearson_r']:+.3f} & {v['r_squared']:.3f} & "
        f"{v['pearson_p']:.2e} & {v['mae']:.3f} & {v['rmse']:.3f} \\\\"
    )
    
    # Arousal
    a = results['arousal']
    lines.append(
        f"Arousal & {a['pearson_r']:+.3f} & {a['r_squared']:.3f} & "
        f"{a['pearson_p']:.2e} & {a['mae']:.3f} & {a['rmse']:.3f} \\\\"
    )
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Tabla LaTeX guardada: {output_path}")


def save_alignment_json(results: Dict, output_path: Path):
    """
    Guarda resultados completos en JSON.
    
    Args:
        results: Dict con resultados de evaluación
        output_path: Path al archivo JSON
    """
    # Crear copia sin las listas de valores (demasiado verbose)
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'engine': results['engine'],
        'num_samples': results['num_samples'],
        'valence': {
            k: v for k, v in results['valence'].items()
            if k not in ['targets', 'estimated']
        },
        'arousal': {
            k: v for k, v in results['arousal'].items()
            if k not in ['targets', 'estimated']
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"JSON guardado: {output_path}")


def plot_scatter(
    targets: List[float],
    estimated: List[float],
    dimension: str,
    r: float,
    p_value: float,
    output_path: Path
):
    """
    Genera scatter plot de target vs estimated.
    
    Args:
        targets: Lista de valores target
        estimated: Lista de valores estimados
        dimension: Nombre de la dimensión (Valence/Arousal)
        r: Correlación de Pearson
        p_value: p-value de la correlación
        output_path: Path al archivo PNG
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(targets, estimated, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    
    # Línea de identidad (ideal)
    lim_min = min(min(targets), min(estimated)) - 0.1
    lim_max = max(max(targets), max(estimated)) + 0.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect alignment')
    
    # Línea de regresión
    z = np.polyfit(targets, estimated, 1)
    p = np.poly1d(z)
    x_line = np.linspace(lim_min, lim_max, 100)
    ax.plot(x_line, p(x_line), 'b-', linewidth=2, alpha=0.7, label=f'Linear fit (r={r:.3f})')
    
    # Títulos y labels
    ax.set_xlabel(f'{dimension} Target', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{dimension} Estimated from Music', fontsize=12, fontweight='bold')
    ax.set_title(
        f'{dimension} Alignment\n'
        f'Pearson r = {r:+.3f}, p = {p_value:.3e}, R² = {r**2:.3f}',
        fontsize=14, fontweight='bold'
    )
    
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Scatter plot guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluación Dimensional: Alineación entre VA Target y VA Estimado'
    )
    parser.add_argument(
        '--benchmark_results',
        type=Path,
        required=True,
        help='Directorio con benchmark_raw.csv'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Directorio de salida (default: mismo que benchmark_results)'
    )
    parser.add_argument(
        '--target_engine',
        type=str,
        default='transformer_finetuned',
        help='Engine a evaluar (default: transformer_finetuned)'
    )
    parser.add_argument(
        '--base_dir',
        type=Path,
        default=None,
        help='Directorio base para resolver paths relativos de MIDIs (default: benchmark_results)'
    )
    
    args = parser.parse_args()
    
    # Determinar output_dir y base_dir
    if args.output_dir is None:
        args.output_dir = args.benchmark_results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.base_dir is None:
        args.base_dir = args.benchmark_results
    
    # Path al CSV
    csv_path = args.benchmark_results / 'benchmark_raw.csv'
    
    logger.info("="*70)
    logger.info("EVALUACIÓN DIMENSIONAL: ALINEACIÓN VA TARGET VS ESTIMATED")
    logger.info("="*70)
    logger.info(f"Benchmark results: {args.benchmark_results}")
    logger.info(f"Target engine:     {args.target_engine}")
    logger.info(f"Base dir (MIDIs):  {args.base_dir}")
    logger.info(f"Output dir:        {args.output_dir}")
    logger.info("="*70 + "\n")
    
    try:
        # Cargar datos
        data = load_benchmark_raw_csv(csv_path)
        
        # Evaluar alineación dimensional
        results = evaluate_dimensional_alignment(
            data, args.target_engine, args.base_dir
        )
        
        # Guardar resultados
        logger.info("="*70)
        logger.info("GUARDANDO RESULTADOS")
        logger.info("="*70)
        
        save_alignment_csv(
            results,
            args.output_dir / 'dimensional_alignment.csv'
        )
        
        save_alignment_latex(
            results,
            args.output_dir / 'dimensional_alignment.tex'
        )
        
        save_alignment_json(
            results,
            args.output_dir / 'dimensional_alignment.json'
        )
        
        # Generar scatter plots
        logger.info("\nGenerando scatter plots...")
        
        plot_scatter(
            results['valence']['targets'],
            results['valence']['estimated'],
            'Valence',
            results['valence']['pearson_r'],
            results['valence']['pearson_p'],
            args.output_dir / 'valence_alignment_scatter.png'
        )
        
        plot_scatter(
            results['arousal']['targets'],
            results['arousal']['estimated'],
            'Arousal',
            results['arousal']['pearson_r'],
            results['arousal']['pearson_p'],
            args.output_dir / 'arousal_alignment_scatter.png'
        )
        
        logger.info("\n" + "="*70)
        logger.info("EVALUACIÓN COMPLETADA")
        logger.info("="*70)
        logger.info(f"Engine:          {args.target_engine}")
        logger.info(f"Samples:         {results['num_samples']}")
        logger.info(f"Valence r:       {results['valence']['pearson_r']:+.3f} (R² = {results['valence']['r_squared']:.3f})")
        logger.info(f"Arousal r:       {results['arousal']['pearson_r']:+.3f} (R² = {results['arousal']['r_squared']:.3f})")
        logger.info(f"Resultados en:   {args.output_dir}")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
