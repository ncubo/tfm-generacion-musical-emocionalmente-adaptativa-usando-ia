#!/usr/bin/env python3
"""
Evaluación de Coherencia Emocional: Matriz de Confusión por Cuadrantes VA.

Clasifica cada muestra generada en un cuadrante VA (objetivo y estimado)
y calcula matriz de confusión, precision, recall y F1 por cuadrante.

Cuadrantes:
    Q1 (Happy/Excited): valence >= 0, arousal >= 0
    Q2 (Angry/Tense):   valence <  0, arousal >= 0
    Q3 (Sad/Depressed): valence <  0, arousal <  0
    Q4 (Calm/Relaxed):  valence >= 0, arousal <  0

Entrada:
    benchmark_raw.csv con columnas: engine, valence, arousal, note_density,
    pitch_range, mean_velocity, status

Salidas (en la misma carpeta del benchmark):
    music_emotion_confusion_matrix.csv
    music_emotion_classification_report.csv
    music_emotion_confusion_matrix.png
    music_emotion_summary.json

Uso:
    python evaluate_emotional_confusion_music.py --benchmark_results results/final_benchmark_20260226_213840
    python evaluate_emotional_confusion_music.py --benchmark_results results/final_benchmark_20260226_213840 --engine transformer_finetuned
    python evaluate_emotional_confusion_music.py --benchmark_results results/final_benchmark_20260226_213840 --engine all

Dependencias:
    - compute_va_heuristic() de evaluate_dimensional_alignment.py
    - load_benchmark_raw_csv() de evaluate_dimensional_alignment.py
    - sklearn.metrics (confusion_matrix, precision_recall_fscore_support)
    - matplotlib (para heatmap)
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import json
from datetime import datetime

# Añadir backend/src y backend/scripts al path
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(BACKEND_ROOT / "scripts"))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)

# Reusar funciones existentes
from evaluate_dimensional_alignment import compute_va_heuristic, load_benchmark_raw_csv

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Definición de cuadrantes VA
QUADRANT_LABELS = ['Q1', 'Q2', 'Q3', 'Q4']
QUADRANT_NAMES = {
    'Q1': 'Q1 (V+, A+)',
    'Q2': 'Q2 (V-, A+)',
    'Q3': 'Q3 (V-, A-)',
    'Q4': 'Q4 (V+, A-)'
}


def va_to_quadrant(valence: float, arousal: float) -> str:
    """
    Asigna un cuadrante VA según signo de valence y arousal.

    Q1: valence >= 0, arousal >= 0
    Q2: valence <  0, arousal >= 0
    Q3: valence <  0, arousal <  0
    Q4: valence >= 0, arousal <  0

    Args:
        valence: valor de valencia [-1, 1]
        arousal: valor de arousal [-1, 1]

    Returns:
        Etiqueta del cuadrante ('Q1', 'Q2', 'Q3', 'Q4')
    """
    if valence >= 0 and arousal >= 0:
        return 'Q1'
    elif valence < 0 and arousal >= 0:
        return 'Q2'
    elif valence < 0 and arousal < 0:
        return 'Q3'
    else:
        return 'Q4'


def estimate_va_from_csv_row(row: Dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Calcula VA estimado a partir de las columnas de features del CSV.

    Usa compute_va_heuristic() de evaluate_dimensional_alignment.py
    pasándole las features del CSV (no requiere leer MIDI).

    Args:
        row: Fila del benchmark_raw.csv (dict)

    Returns:
        Tupla (valence_estimated, arousal_estimated) o (None, None) si faltan datos
    """
    try:
        features = {
            'note_density': float(row['note_density']),
            'pitch_range': float(row['pitch_range']),
            'mean_velocity': float(row['mean_velocity'])
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.debug(f"No se pudieron extraer features de la fila: {e}")
        return None, None

    return compute_va_heuristic(features)


def compute_quadrant_metrics(
    data: List[Dict],
    engine: str
) -> Dict:
    """
    Calcula matriz de confusión y métricas de clasificación por cuadrante.

    Args:
        data: Lista de filas del benchmark_raw.csv
        engine: Nombre del engine a evaluar

    Returns:
        Dict con confusion_matrix, classification_report, accuracy, y listas
    """
    # Filtrar por engine y status
    engine_data = [
        row for row in data
        if row.get('engine') == engine
        and row.get('status') == 'success'
    ]

    if not engine_data:
        raise ValueError(
            f"No hay datos con status='success' para engine='{engine}'. "
            f"Engines disponibles: {sorted(set(r.get('engine', '?') for r in data))}"
        )

    logger.info(f"  Procesando {len(engine_data)} muestras de '{engine}'...")

    y_true = []
    y_pred = []
    skipped = 0

    for row in engine_data:
        # Cuadrante objetivo (del VA target)
        q_target = va_to_quadrant(row['valence'], row['arousal'])

        # VA estimado desde features del CSV
        v_est, a_est = estimate_va_from_csv_row(row)
        if v_est is None or a_est is None:
            skipped += 1
            continue

        q_estimated = va_to_quadrant(v_est, a_est)

        y_true.append(q_target)
        y_pred.append(q_estimated)

    if skipped > 0:
        logger.warning(f"  Filas omitidas por datos faltantes: {skipped}")

    n_valid = len(y_true)
    if n_valid == 0:
        raise ValueError(f"No hay muestras válidas para engine='{engine}'")

    logger.info(f"  Muestras válidas: {n_valid}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=QUADRANT_LABELS)

    # Precision, recall, F1 por cuadrante
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=QUADRANT_LABELS,
        average=None,
        zero_division=0
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=QUADRANT_LABELS,
        average='macro',
        zero_division=0
    )

    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=QUADRANT_LABELS,
        average='weighted',
        zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)

    results = {
        'engine': engine,
        'num_samples': n_valid,
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'labels': QUADRANT_LABELS,
        'per_quadrant': {},
        'macro_avg': {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1': float(f1_macro)
        },
        'weighted_avg': {
            'precision': float(precision_weighted),
            'recall': float(recall_weighted),
            'f1': float(f1_weighted)
        },
        'y_true': y_true,
        'y_pred': y_pred
    }

    for i, label in enumerate(QUADRANT_LABELS):
        results['per_quadrant'][label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    # Log
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  Macro F1: {f1_macro:.3f}")
    for label in QUADRANT_LABELS:
        q = results['per_quadrant'][label]
        logger.info(
            f"    {QUADRANT_NAMES[label]}: "
            f"P={q['precision']:.3f} R={q['recall']:.3f} F1={q['f1']:.3f} "
            f"(n={q['support']})"
        )

    return results


def save_confusion_matrix_csv(cm: List[List[int]], labels: List[str], output_path: Path):
    """
    Guarda la matriz de confusión en CSV.

    Args:
        cm: Matriz de confusión (lista de listas)
        labels: Etiquetas de cuadrantes
        output_path: Path al CSV
    """
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true\\predicted'] + [QUADRANT_NAMES[l] for l in labels])
        for i, label in enumerate(labels):
            writer.writerow([QUADRANT_NAMES[label]] + [str(v) for v in cm[i]])

    logger.info(f"Confusion matrix CSV guardado: {output_path}")


def save_classification_report_csv(results: Dict, output_path: Path):
    """
    Guarda classification report en CSV.

    Args:
        results: Dict con per_quadrant, macro_avg, weighted_avg, accuracy
        output_path: Path al CSV
    """
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Quadrant', 'Precision', 'Recall', 'F1-Score', 'Support'])

        for label in QUADRANT_LABELS:
            q = results['per_quadrant'][label]
            writer.writerow([
                QUADRANT_NAMES[label],
                f"{q['precision']:.4f}",
                f"{q['recall']:.4f}",
                f"{q['f1']:.4f}",
                q['support']
            ])

        # Separador
        writer.writerow([])

        # Macro avg
        m = results['macro_avg']
        writer.writerow([
            'macro avg',
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            results['num_samples']
        ])

        # Weighted avg
        w = results['weighted_avg']
        writer.writerow([
            'weighted avg',
            f"{w['precision']:.4f}",
            f"{w['recall']:.4f}",
            f"{w['f1']:.4f}",
            results['num_samples']
        ])

        # Accuracy
        writer.writerow([])
        writer.writerow(['accuracy', f"{results['accuracy']:.4f}", '', '', results['num_samples']])

    logger.info(f"Classification report CSV guardado: {output_path}")


def save_summary_json(all_results: Dict, output_path: Path):
    """
    Guarda resumen completo en JSON.

    Args:
        all_results: Dict con resultados por engine
        output_path: Path al JSON
    """
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Evaluación de coherencia emocional musical por cuadrantes VA',
        'quadrant_definitions': {
            'Q1': 'valence >= 0, arousal >= 0 (Happy/Excited)',
            'Q2': 'valence <  0, arousal >= 0 (Angry/Tense)',
            'Q3': 'valence <  0, arousal <  0 (Sad/Depressed)',
            'Q4': 'valence >= 0, arousal <  0 (Calm/Relaxed)'
        },
        'va_estimation_method': (
            'compute_va_heuristic() de evaluate_dimensional_alignment.py: '
            'valence_est = (mean_velocity / 127.0) * 2.0 - 1.0; '
            'arousal_est = (0.6 * clip(note_density/5.0) + 0.4 * clip(pitch_range/48.0)) * 2.0 - 1.0'
        ),
        'engines': {}
    }

    for engine_name, results in all_results.items():
        output_data['engines'][engine_name] = {
            'num_samples': results['num_samples'],
            'accuracy': results['accuracy'],
            'confusion_matrix': results['confusion_matrix'],
            'labels': results['labels'],
            'per_quadrant': results['per_quadrant'],
            'macro_avg': results['macro_avg'],
            'weighted_avg': results['weighted_avg']
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Summary JSON guardado: {output_path}")


def plot_confusion_matrix(
    cm: List[List[int]],
    labels: List[str],
    engine: str,
    accuracy: float,
    output_path: Path
):
    """
    Genera heatmap de la matriz de confusión con matplotlib puro.

    Args:
        cm: Matriz de confusión (lista de listas)
        labels: Etiquetas de cuadrantes
        engine: Nombre del engine
        accuracy: Accuracy global
        output_path: Path al PNG
    """
    cm_array = np.array(cm)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Heatmap manual con imshow
    im = ax.imshow(cm_array, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Anotaciones en celdas
    thresh = cm_array.max() / 2.0
    for i in range(n):
        for j in range(n):
            color = 'white' if cm_array[i, j] > thresh else 'black'
            ax.text(
                j, i, str(cm_array[i, j]),
                ha='center', va='center',
                fontsize=16, fontweight='bold',
                color=color
            )

    # Etiquetas
    display_labels = [QUADRANT_NAMES[l] for l in labels]
    ax.set_xticks(range(n))
    ax.set_xticklabels(display_labels, fontsize=10, rotation=30, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(display_labels, fontsize=10)

    ax.set_xlabel('Cuadrante Estimado (desde features musicales)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cuadrante Objetivo (VA target)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Matriz de Confusión Emocional — {engine}\n'
        f'Accuracy = {accuracy:.1%} (n={cm_array.sum()})',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix PNG guardado: {output_path}")


def find_latest_benchmark(results_dir: Path) -> Path:
    """
    Encuentra la carpeta final_benchmark_* más reciente.

    Args:
        results_dir: Directorio results/

    Returns:
        Path a la carpeta más reciente

    Raises:
        FileNotFoundError: Si no hay carpetas de benchmark
    """
    benchmark_dirs = sorted(
        results_dir.glob('final_benchmark_*'),
        key=lambda p: p.name,
        reverse=True
    )
    if not benchmark_dirs:
        raise FileNotFoundError(
            f"No se encontraron carpetas final_benchmark_* en {results_dir}"
        )
    return benchmark_dirs[0]


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Evaluación de Coherencia Emocional Musical: '
            'Matriz de Confusión por Cuadrantes VA'
        )
    )
    parser.add_argument(
        '--benchmark_results',
        type=Path,
        default=None,
        help=(
            'Directorio con benchmark_raw.csv '
            '(default: carpeta final_benchmark_* más reciente en results/)'
        )
    )
    parser.add_argument(
        '--engine',
        type=str,
        default='transformer_finetuned',
        help=(
            'Engine a evaluar. Usar "all" para evaluar todos los engines. '
            '(default: transformer_finetuned)'
        )
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Directorio de salida (default: mismo que benchmark_results)'
    )

    args = parser.parse_args()

    # Resolver benchmark_results
    if args.benchmark_results is None:
        results_base = BACKEND_ROOT / 'results'
        args.benchmark_results = find_latest_benchmark(results_base)
        logger.info(f"Usando benchmark más reciente: {args.benchmark_results.name}")

    # Resolver output_dir
    if args.output_dir is None:
        args.output_dir = args.benchmark_results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.benchmark_results / 'benchmark_raw.csv'

    logger.info("=" * 70)
    logger.info("EVALUACIÓN DE COHERENCIA EMOCIONAL MUSICAL")
    logger.info("Matriz de Confusión por Cuadrantes VA")
    logger.info("=" * 70)
    logger.info(f"Benchmark:   {args.benchmark_results}")
    logger.info(f"Engine:      {args.engine}")
    logger.info(f"Output dir:  {args.output_dir}")
    logger.info("=" * 70 + "\n")

    # Cargar datos
    data = load_benchmark_raw_csv(csv_path)

    # Determinar engines a evaluar
    available_engines = sorted(set(row.get('engine', '') for row in data))
    logger.info(f"Engines en el CSV: {available_engines}")

    if args.engine == 'all':
        engines_to_eval = available_engines
    else:
        if args.engine not in available_engines:
            logger.error(
                f"Engine '{args.engine}' no encontrado en CSV. "
                f"Disponibles: {available_engines}"
            )
            sys.exit(1)
        engines_to_eval = [args.engine]

    # Evaluar cada engine
    all_results = {}

    for engine in engines_to_eval:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Evaluando engine: {engine}")
        logger.info(f"{'─' * 50}")

        try:
            results = compute_quadrant_metrics(data, engine)
            all_results[engine] = results
        except ValueError as e:
            logger.error(f"  Error: {e}")
            continue

    if not all_results:
        logger.error("No se obtuvieron resultados para ningún engine.")
        sys.exit(1)

    # Guardar outputs
    logger.info(f"\n{'=' * 70}")
    logger.info("GUARDANDO RESULTADOS")
    logger.info("=" * 70)

    for engine, results in all_results.items():
        # Prefijo para archivos si hay múltiples engines
        prefix = f"music_emotion_{engine}_" if len(all_results) > 1 else "music_emotion_"

        # Confusion matrix CSV
        save_confusion_matrix_csv(
            results['confusion_matrix'],
            results['labels'],
            args.output_dir / f"{prefix}confusion_matrix.csv"
        )

        # Classification report CSV
        save_classification_report_csv(
            results,
            args.output_dir / f"{prefix}classification_report.csv"
        )

        # Confusion matrix PNG
        plot_confusion_matrix(
            results['confusion_matrix'],
            results['labels'],
            engine,
            results['accuracy'],
            args.output_dir / f"{prefix}confusion_matrix.png"
        )

    # Summary JSON (todos los engines)
    save_summary_json(
        all_results,
        args.output_dir / "music_emotion_summary.json"
    )

    # Resumen final
    logger.info(f"\n{'=' * 70}")
    logger.info("EVALUACIÓN COMPLETADA")
    logger.info("=" * 70)

    for engine, results in all_results.items():
        logger.info(f"\n  Engine: {engine}")
        logger.info(f"    Samples:    {results['num_samples']}")
        logger.info(f"    Accuracy:   {results['accuracy']:.3f}")
        logger.info(f"    Macro F1:   {results['macro_avg']['f1']:.3f}")
        logger.info(f"    Weighted F1:{results['weighted_avg']['f1']:.3f}")
        for label in QUADRANT_LABELS:
            q = results['per_quadrant'][label]
            logger.info(
                f"      {QUADRANT_NAMES[label]}: "
                f"P={q['precision']:.3f} R={q['recall']:.3f} "
                f"F1={q['f1']:.3f} (n={q['support']})"
            )

    logger.info(f"\nResultados en: {args.output_dir}")
    logger.info("=" * 70 + "\n")


if __name__ == '__main__':
    main()
