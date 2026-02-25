#!/usr/bin/env python3
"""
Métrica Compuesta del Sistema Completo: SystemCoherenceScore.

Este script calcula una métrica unificada que evalúa el rendimiento global
del sistema de generación musical emocionalmente adaptativa.

SystemCoherenceScore = w1*EmotionalAlignment + w2*MusicalStructure + w3*Latency

Componentes:
- EmotionalAlignment: Correlación Spearman promedio del modelo finetuned (0-1)
- MusicalStructure: Estabilidad entre seeds basada en std de velocity (0-1)
- Latency: Inversa normalizada del tiempo de generación (0-1)

Uso:
    python calculate_system_coherence.py --benchmark_results results/final_benchmark_20260223_120200
    python calculate_system_coherence.py --benchmark_results results/final_benchmark_20260223_120200 --weights 0.5,0.3,0.2

Output:
    - system_coherence.json (score final y componentes)
    - system_coherence.tex (tabla LaTeX)
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import csv
import json
from datetime import datetime

# Añadir backend/src al path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

import numpy as np
from scipy.stats import spearmanr

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_benchmark_raw_csv(csv_path: Path) -> list:
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
                
                if row.get('generation_time_ms'):
                    row['generation_time_ms'] = float(row['generation_time_ms'])
                
                if row.get('note_density'):
                    row['note_density'] = float(row['note_density'])
                if row.get('pitch_range'):
                    row['pitch_range'] = int(row['pitch_range'])
                if row.get('mean_velocity'):
                    row['mean_velocity'] = float(row['mean_velocity'])
                    
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parseando fila: {e}")
                continue
            
            data.append(row)
    
    logger.info(f"Benchmark raw cargado: {len(data)} filas")
    return data


def calculate_emotional_alignment_score(data: list, target_engine: str = 'transformer_finetuned') -> Tuple[float, Dict]:
    """
    Calcula EmotionalAlignmentScore basado en correlaciones de Spearman.
    
    Score = promedio de |rho| para las 3 métricas musicales del modelo finetuned.
    
    Args:
        data: Datos crudos del benchmark
        target_engine: Engine a evaluar (default: transformer_finetuned)
        
    Returns:
        Tupla (score normalizado 0-1, dict con detalles)
    """
    logger.info(f"[1/3] Calculando EmotionalAlignmentScore ({target_engine})...")
    
    # Filtrar datos del engine target y con status success
    engine_data = [
        row for row in data
        if row.get('engine') == target_engine and row.get('status') == 'success'
    ]
    
    if len(engine_data) < 3:
        logger.warning(f"  Insuficientes datos para {target_engine} (n={len(engine_data)})")
        return 0.0, {'error': 'insufficient_data'}
    
    # Calcular correlaciones de Spearman para cada métrica
    metrics = ['note_density', 'pitch_range', 'mean_velocity']
    correlations = {}
    
    for metric in metrics:
        # Extraer pares (arousal, metric_value)
        pairs = [
            (row['arousal'], row[metric])
            for row in engine_data
            if row.get(metric) is not None
        ]
        
        if len(pairs) < 3:
            correlations[metric] = {'rho': 0.0, 'p_value': 1.0}
            continue
        
        arousals, values = zip(*pairs)
        rho, p_value = spearmanr(arousals, values)
        
        correlations[metric] = {
            'rho': float(rho),
            'p_value': float(p_value),
            'abs_rho': float(abs(rho))
        }
        
        logger.info(f"  {metric:20s}: rho = {rho:+.3f} (p = {p_value:.3e})")
    
    # Score = promedio de |rho|
    abs_rhos = [corr['abs_rho'] for corr in correlations.values()]
    score = np.mean(abs_rhos)
    
    details = {
        'score': float(score),
        'correlations': correlations,
        'num_samples': len(engine_data),
        'interpretation': _interpret_correlation_strength(score)
    }
    
    logger.info(f"  -> EmotionalAlignmentScore: {score:.3f}")
    
    return score, details


def calculate_musical_structure_score(data: list, target_engine: str = 'transformer_finetuned') -> Tuple[float, Dict]:
    """
    Calcula MusicalStructureScore basado en estabilidad entre seeds.
    
    Score = 1 - (std_promedio_velocity / 127) normalizado.
    Menor std = mayor estabilidad = mayor score.
    
    Args:
        data: Datos crudos del benchmark
        target_engine: Engine a evaluar
        
    Returns:
        Tupla (score normalizado 0-1, dict con detalles)
    """
    logger.info(f"[2/3] Calculando MusicalStructureScore ({target_engine})...")
    
    # Filtrar datos del engine target y con status success
    engine_data = [
        row for row in data
        if row.get('engine') == target_engine and row.get('status') == 'success'
    ]
    
    if len(engine_data) < 3:
        logger.warning(f"  Insuficientes datos para {target_engine}")
        return 0.0, {'error': 'insufficient_data'}
    
    # Agrupar por (valence, arousal) y calcular std de velocity
    from collections import defaultdict
    groups = defaultdict(list)
    
    for row in engine_data:
        if row.get('mean_velocity') is not None:
            key = (row['valence'], row['arousal'])
            groups[key].append(row['mean_velocity'])
    
    # Calcular std para cada grupo
    stds = []
    for key, velocities in groups.items():
        if len(velocities) > 1:
            std = np.std(velocities, ddof=1)
            stds.append(std)
    
    if not stds:
        logger.warning("  No hay suficientes repeticiones para calcular std")
        return 0.0, {'error': 'no_repetitions'}
    
    # Std promedio
    avg_std = np.mean(stds)
    
    # Normalizar: std_max teórico = 127 (rango MIDI velocity)
    # Score = 1 - (std / 127) para que menor std = mayor score
    # Clamp a [0, 1]
    score = max(0.0, min(1.0, 1.0 - (avg_std / 127.0)))
    
    details = {
        'score': float(score),
        'avg_std_velocity': float(avg_std),
        'num_va_groups': len(groups),
        'interpretation': _interpret_stability(avg_std)
    }
    
    logger.info(f"  Average std velocity: {avg_std:.2f} MIDI")
    logger.info(f"  -> MusicalStructureScore: {score:.3f}")
    
    return score, details


def calculate_latency_score(data: list, target_engine: str = 'transformer_finetuned') -> Tuple[float, Dict]:
    """
    Calcula LatencyScore basado en tiempo de generación.
    
    Score = 1 - (latency_mean / latency_max_acceptable)
    Menor latencia = mayor score.
    
    Args:
        data: Datos crudos del benchmark
        target_engine: Engine a evaluar
        
    Returns:
        Tupla (score normalizado 0-1, dict con detalles)
    """
    logger.info(f"[3/3] Calculando LatencyScore ({target_engine})...")
    
    # Filtrar datos del engine target y con status success
    engine_data = [
        row for row in data
        if row.get('engine') == target_engine and row.get('status') == 'success'
        and row.get('generation_time_ms') is not None
    ]
    
    if not engine_data:
        logger.warning(f"  No hay datos de latencia para {target_engine}")
        return 0.0, {'error': 'no_latency_data'}
    
    # Calcular latencia promedio
    latencies = [row['generation_time_ms'] for row in engine_data]
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies, ddof=1) if len(latencies) > 1 else 0.0
    
    # Latencia máxima aceptable: 10 segundos (10000 ms)
    # En la práctica, transformers suelen estar en 500-3000ms
    max_acceptable_latency = 10000.0
    
    # Score = 1 - (latency / max_acceptable), clamped a [0, 1]
    score = max(0.0, min(1.0, 1.0 - (mean_latency / max_acceptable_latency)))
    
    details = {
        'score': float(score),
        'mean_latency_ms': float(mean_latency),
        'std_latency_ms': float(std_latency),
        'max_acceptable_ms': max_acceptable_latency,
        'num_samples': len(latencies),
        'interpretation': _interpret_latency(mean_latency)
    }
    
    logger.info(f"  Mean latency: {mean_latency:.1f} ms")
    logger.info(f"  -> LatencyScore: {score:.3f}")
    
    return score, details


def _interpret_correlation_strength(score: float) -> str:
    """Interpreta la fuerza de correlación."""
    if score >= 0.7:
        return "STRONG (excellent emotional alignment)"
    elif score >= 0.5:
        return "MODERATE (acceptable emotional alignment)"
    elif score >= 0.3:
        return "WEAK (limited emotional alignment)"
    else:
        return "VERY WEAK (poor emotional alignment)"


def _interpret_stability(std: float) -> str:
    """Interpreta la estabilidad del sistema."""
    if std < 5:
        return "VERY STABLE (excellent consistency)"
    elif std < 10:
        return "STABLE (good consistency)"
    elif std < 20:
        return "MODERATE (acceptable variability)"
    else:
        return "UNSTABLE (high variability)"


def _interpret_latency(latency_ms: float) -> str:
    """Interpreta latencia."""
    if latency_ms < 500:
        return "EXCELLENT (real-time capable)"
    elif latency_ms < 1500:
        return "GOOD (acceptable for interactive use)"
    elif latency_ms < 5000:
        return "ACCEPTABLE (near real-time)"
    else:
        return "SLOW (batch processing only)"


def calculate_system_coherence_score(
    emotional_score: float,
    musical_score: float,
    latency_score: float,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
) -> Tuple[float, Dict]:
    """
    Calcula SystemCoherenceScore unificado.
    
    Args:
        emotional_score: EmotionalAlignmentScore (0-1)
        musical_score: MusicalStructureScore (0-1)
        latency_score: LatencyScore (0-1)
        weights: Tupla (w1, w2, w3) con pesos que suman 1.0
        
    Returns:
        Tupla (score final 0-1, dict con breakdown)
    """
    w1, w2, w3 = weights
    
    # Normalizar pesos a sumar 1.0
    total_weight = w1 + w2 + w3
    w1, w2, w3 = w1/total_weight, w2/total_weight, w3/total_weight
    
    # Calcular score ponderado
    final_score = (
        w1 * emotional_score +
        w2 * musical_score +
        w3 * latency_score
    )
    
    breakdown = {
        'final_score': float(final_score),
        'weights': {
            'emotional_alignment': float(w1),
            'musical_structure': float(w2),
            'latency': float(w3)
        },
        'components': {
            'emotional_alignment': {
                'score': float(emotional_score),
                'weighted_contribution': float(w1 * emotional_score)
            },
            'musical_structure': {
                'score': float(musical_score),
                'weighted_contribution': float(w2 * musical_score)
            },
            'latency': {
                'score': float(latency_score),
                'weighted_contribution': float(w3 * latency_score)
            }
        },
        'interpretation': _interpret_system_coherence(final_score)
    }
    
    return final_score, breakdown


def _interpret_system_coherence(score: float) -> str:
    """Interpreta el score del sistema completo."""
    if score >= 0.8:
        return "EXCELLENT (production-ready system)"
    elif score >= 0.6:
        return "GOOD (functional system with room for improvement)"
    elif score >= 0.4:
        return "ACCEPTABLE (research prototype with limitations)"
    else:
        return "POOR (significant improvements needed)"


def save_coherence_json(
    final_score: float,
    breakdown: Dict,
    emotional_details: Dict,
    musical_details: Dict,
    latency_details: Dict,
    output_path: Path
):
    """
    Guarda resultados completos en JSON.
    
    Args:
        final_score: Score final del sistema
        breakdown: Desglose de componentes
        emotional_details: Detalles de EmotionalAlignment
        musical_details: Detalles de MusicalStructure
        latency_details: Detalles de Latency
        output_path: Path al archivo JSON
    """
    result = {
        'timestamp': datetime.now().isoformat(),
        'system_coherence_score': final_score,
        'breakdown': breakdown,
        'component_details': {
            'emotional_alignment': emotional_details,
            'musical_structure': musical_details,
            'latency': latency_details
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"JSON guardado: {output_path}")


def save_coherence_latex(
    final_score: float,
    breakdown: Dict,
    output_path: Path
):
    """
    Genera tabla LaTeX con el SystemCoherenceScore.
    
    Args:
        final_score: Score final del sistema
        breakdown: Desglose de componentes
        output_path: Path al archivo .tex
    """
    lines = []
    
    lines.append("% Tabla generada automáticamente por calculate_system_coherence.py")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{System Coherence Score - Métrica Compuesta del Sistema}")
    lines.append("\\label{tab:system_coherence_score}")
    lines.append("\\begin{tabular}{l|c|c|c}")
    lines.append("\\hline")
    lines.append("\\textbf{Component} & \\textbf{Score} & \\textbf{Weight} & \\textbf{Contribution} \\\\")
    lines.append("\\hline")
    
    # Componentes
    components = [
        ('Emotional Alignment', 'emotional_alignment'),
        ('Musical Structure', 'musical_structure'),
        ('Latency', 'latency')
    ]
    
    for label, key in components:
        comp = breakdown['components'][key]
        weight = breakdown['weights'][key]
        score = comp['score']
        contribution = comp['weighted_contribution']
        
        lines.append(f"{label} & {score:.3f} & {weight:.2f} & {contribution:.3f} \\\\")
    
    lines.append("\\hline")
    lines.append(f"\\textbf{{System Coherence}} & \\textbf{{{final_score:.3f}}} & 1.00 & {final_score:.3f} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Tabla LaTeX guardada: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calcula System Coherence Score - Métrica Compuesta del Sistema'
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
        '--weights',
        type=str,
        default='0.5,0.3,0.2',
        help='Pesos w1,w2,w3 separados por comas (default: 0.5,0.3,0.2)'
    )
    
    args = parser.parse_args()
    
    # Parsear pesos
    try:
        weights = tuple(float(w) for w in args.weights.split(','))
        if len(weights) != 3:
            raise ValueError("Se requieren exactamente 3 pesos")
    except Exception as e:
        logger.error(f"Error en formato de pesos: {e}")
        logger.error("Usa formato: w1,w2,w3 (ejemplo: 0.5,0.3,0.2)")
        sys.exit(1)
    
    # Determinar output_dir
    if args.output_dir is None:
        args.output_dir = args.benchmark_results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Path al CSV
    csv_path = args.benchmark_results / 'benchmark_raw.csv'
    
    logger.info("="*70)
    logger.info("SYSTEM COHERENCE SCORE - MÉTRICA COMPUESTA")
    logger.info("="*70)
    logger.info(f"Benchmark results: {args.benchmark_results}")
    logger.info(f"Target engine:     {args.target_engine}")
    logger.info(f"Weights:           {weights}")
    logger.info(f"Output dir:        {args.output_dir}")
    logger.info("="*70 + "\n")
    
    try:
        # Cargar datos
        data = load_benchmark_raw_csv(csv_path)
        
        # Calcular componentes
        emotional_score, emotional_details = calculate_emotional_alignment_score(
            data, args.target_engine
        )
        
        musical_score, musical_details = calculate_musical_structure_score(
            data, args.target_engine
        )
        
        latency_score, latency_details = calculate_latency_score(
            data, args.target_engine
        )
        
        # Calcular score final
        logger.info("\n" + "="*70)
        logger.info("CALCULANDO SYSTEM COHERENCE SCORE")
        logger.info("="*70)
        
        final_score, breakdown = calculate_system_coherence_score(
            emotional_score, musical_score, latency_score, weights
        )
        
        logger.info(f"\nSystemCoherenceScore = {final_score:.3f}")
        logger.info(f"  Emotional Alignment: {emotional_score:.3f} × {weights[0]:.2f} = {breakdown['components']['emotional_alignment']['weighted_contribution']:.3f}")
        logger.info(f"  Musical Structure:   {musical_score:.3f} × {weights[1]:.2f} = {breakdown['components']['musical_structure']['weighted_contribution']:.3f}")
        logger.info(f"  Latency:             {latency_score:.3f} × {weights[2]:.2f} = {breakdown['components']['latency']['weighted_contribution']:.3f}")
        logger.info(f"\nInterpretation: {breakdown['interpretation']}")
        
        # Guardar resultados
        logger.info("\n" + "="*70)
        logger.info("GUARDANDO RESULTADOS")
        logger.info("="*70)
        
        save_coherence_json(
            final_score, breakdown,
            emotional_details, musical_details, latency_details,
            args.output_dir / 'system_coherence.json'
        )
        
        save_coherence_latex(
            final_score, breakdown,
            args.output_dir / 'system_coherence.tex'
        )
        
        logger.info("\n" + "="*70)
        logger.info("CÁLCULO COMPLETADO")
        logger.info("="*70)
        logger.info(f"SystemCoherenceScore: {final_score:.3f}")
        logger.info(f"Resultados guardados en: {args.output_dir}")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Error durante el cálculo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
