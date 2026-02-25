#!/usr/bin/env python3
"""
Evaluación del clasificador emocional con matriz de confusión.

Este script evalúa el rendimiento del detector DeepFace sobre un dataset
de imágenes con ground truth emocional, calculando métricas de clasificación.

Uso:
    python evaluate_emotion_classifier.py --dataset_csv data/emotion_ground_truth.csv
    python evaluate_emotion_classifier.py --dataset_csv data/emotion_ground_truth.csv --output_dir results/emotion_eval

Formato CSV de entrada:
    image_path,true_emotion
    images/happy_001.jpg,happy
    images/sad_002.jpg,sad
    ...

Output:
    - confusion_matrix.csv (matriz cruda)
    - confusion_matrix.png (heatmap)
    - classification_report.csv (precision/recall/f1 por clase)
    - classification_report.tex (tabla LaTeX)
    - metrics_summary.json (accuracy, macro F1, etc.)
"""

import sys
import os
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

import cv2
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# Imports internos
from core.emotion import DeepFaceEmotionDetector, STANDARD_EMOTIONS

# Matplotlib para heatmap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_ground_truth_dataset(csv_path: Path) -> List[Dict[str, str]]:
    """
    Carga dataset con ground truth emocional.
    
    Args:
        csv_path: Path al CSV con columnas: image_path, true_emotion
        
    Returns:
        Lista de dicts con image_path y true_emotion
        
    Raises:
        FileNotFoundError: Si el CSV no existe
        ValueError: Si el formato es inválido
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    
    dataset = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        if 'image_path' not in reader.fieldnames or 'true_emotion' not in reader.fieldnames:
            raise ValueError("CSV debe tener columnas: image_path, true_emotion")
        
        for row in reader:
            dataset.append({
                'image_path': row['image_path'],
                'true_emotion': row['true_emotion'].lower()
            })
    
    logger.info(f"Dataset cargado: {len(dataset)} imágenes desde {csv_path.name}")
    return dataset


def evaluate_detector(
    dataset: List[Dict[str, str]],
    detector: DeepFaceEmotionDetector,
    base_dir: Path
) -> Tuple[List[str], List[str]]:
    """
    Ejecuta el detector sobre el dataset y recopila predicciones.
    
    Args:
        dataset: Lista de dicts con image_path y true_emotion
        detector: Instancia de DeepFaceEmotionDetector
        base_dir: Directorio base para resolver rutas relativas
        
    Returns:
        Tupla (y_true, y_pred) con listas de emociones
    """
    y_true = []
    y_pred = []
    
    total = len(dataset)
    logger.info(f"Evaluando detector sobre {total} imágenes...\n")
    
    for i, sample in enumerate(dataset):
        # Resolver path absoluto
        image_path = sample['image_path']
        if not Path(image_path).is_absolute():
            image_path = base_dir / image_path
        
        true_emotion = sample['true_emotion']
        
        # Cargar imagen
        try:
            frame = cv2.imread(str(image_path))
            if frame is None:
                logger.warning(f"  [{i+1}/{total}] No se pudo cargar: {image_path}")
                continue
            
            # Detectar emoción
            result = detector.detect(frame)
            predicted_emotion = result['emotion']
            confidence = result['confidence']
            
            y_true.append(true_emotion)
            y_pred.append(predicted_emotion)
            
            # Log cada 10 imágenes
            if (i + 1) % 10 == 0 or (i + 1) == total:
                match = "✓" if predicted_emotion == true_emotion else "✗"
                logger.info(f"  [{i+1}/{total}] {match} True: {true_emotion:8s} | Pred: {predicted_emotion:8s} | Conf: {confidence:.1f}%")
                
        except Exception as e:
            logger.error(f"  [{i+1}/{total}] Error procesando {image_path}: {e}")
            continue
    
    logger.info(f"\n[OK] Evaluación completada: {len(y_pred)}/{total} imágenes procesadas")
    return y_true, y_pred


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """
    Calcula todas las métricas de clasificación.
    
    Args:
        y_true: Lista de emociones verdaderas
        y_pred: Lista de emociones predichas
        
    Returns:
        Dict con todas las métricas
    """
    # Obtener labels únicos presentes en los datos
    labels = sorted(set(y_true + y_pred))
    
    # Accuracy global
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    # Macro F1
    macro_f1 = np.mean(f1)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'labels': labels,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        'support': support.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"MÉTRICAS DE CLASIFICACIÓN")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:  {accuracy:.3f}")
    logger.info(f"Macro F1:  {macro_f1:.3f}")
    logger.info(f"{'='*60}\n")
    
    return metrics


def save_confusion_matrix_csv(cm: np.ndarray, labels: List[str], output_path: Path):
    """
    Guarda matriz de confusión en CSV con headers de fila/columna.
    
    Args:
        cm: Matriz de confusión (numpy array)
        labels: Lista de nombres de clases
        output_path: Path al archivo CSV
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header con labels
        writer.writerow(['True \\ Predicted'] + labels)
        
        # Filas con label + valores
        for i, label in enumerate(labels):
            writer.writerow([label] + cm[i].tolist())
    
    logger.info(f"Matriz de confusión guardada: {output_path}")


def save_confusion_matrix_heatmap(cm: np.ndarray, labels: List[str], output_path: Path):
    """
    Genera heatmap de la matriz de confusión.
    
    Args:
        cm: Matriz de confusión (numpy array)
        labels: Lista de nombres de clases
        output_path: Path al archivo PNG
    """
    plt.figure(figsize=(10, 8))
    
    # Crear heatmap con seaborn
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.title('Confusion Matrix - Emotion Classifier', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Heatmap guardado: {output_path}")


def save_classification_report_csv(metrics: Dict, output_path: Path):
    """
    Guarda reporte de clasificación en CSV.
    
    Args:
        metrics: Dict con métricas calculadas
        output_path: Path al archivo CSV
    """
    rows = []
    
    for i, label in enumerate(metrics['labels']):
        rows.append({
            'emotion': label,
            'precision': round(metrics['precision'][i], 3),
            'recall': round(metrics['recall'][i], 3),
            'f1_score': round(metrics['f1_score'][i], 3),
            'support': metrics['support'][i]
        })
    
    # Agregar fila de promedios
    rows.append({
        'emotion': 'macro_avg',
        'precision': round(np.mean(metrics['precision']), 3),
        'recall': round(np.mean(metrics['recall']), 3),
        'f1_score': round(metrics['macro_f1'], 3),
        'support': sum(metrics['support'])
    })
    
    # Escribir CSV
    fieldnames = ['emotion', 'precision', 'recall', 'f1_score', 'support']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"Reporte CSV guardado: {output_path}")


def save_classification_report_latex(metrics: Dict, output_path: Path):
    """
    Genera tabla LaTeX con el reporte de clasificación.
    
    Args:
        metrics: Dict con métricas calculadas
        output_path: Path al archivo .tex
    """
    lines = []
    
    lines.append("% Tabla generada automáticamente por evaluate_emotion_classifier.py")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Reporte de Clasificación - Detector Emocional}")
    lines.append("\\label{tab:emotion_classification_report}")
    lines.append("\\begin{tabular}{l|ccc|c}")
    lines.append("\\hline")
    lines.append("\\textbf{Emotion} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{Support} \\\\")
    lines.append("\\hline")
    
    # Filas por emoción
    for i, label in enumerate(metrics['labels']):
        precision = metrics['precision'][i]
        recall = metrics['recall'][i]
        f1 = metrics['f1_score'][i]
        support = metrics['support'][i]
        
        lines.append(f"{label.capitalize()} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {support} \\\\")
    
    lines.append("\\hline")
    
    # Fila de promedios
    macro_precision = np.mean(metrics['precision'])
    macro_recall = np.mean(metrics['recall'])
    macro_f1 = metrics['macro_f1']
    total_support = sum(metrics['support'])
    
    lines.append(f"\\textbf{{Macro Avg}} & \\textbf{{{macro_precision:.3f}}} & \\textbf{{{macro_recall:.3f}}} & \\textbf{{{macro_f1:.3f}}} & {total_support} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Tabla LaTeX guardada: {output_path}")


def save_metrics_summary_json(metrics: Dict, output_path: Path):
    """
    Guarda resumen de métricas en JSON.
    
    Args:
        metrics: Dict con métricas calculadas
        output_path: Path al archivo JSON
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'accuracy': round(metrics['accuracy'], 4),
        'macro_f1': round(metrics['macro_f1'], 4),
        'macro_precision': round(np.mean(metrics['precision']), 4),
        'macro_recall': round(np.mean(metrics['recall']), 4),
        'num_samples': sum(metrics['support']),
        'num_classes': len(metrics['labels']),
        'per_class_metrics': {}
    }
    
    for i, label in enumerate(metrics['labels']):
        summary['per_class_metrics'][label] = {
            'precision': round(metrics['precision'][i], 4),
            'recall': round(metrics['recall'][i], 4),
            'f1_score': round(metrics['f1_score'][i], 4),
            'support': int(metrics['support'][i])
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resumen JSON guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluación del clasificador emocional con matriz de confusión'
    )
    parser.add_argument(
        '--dataset_csv',
        type=Path,
        required=True,
        help='CSV con ground truth (columnas: image_path, true_emotion)'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('results/emotion_evaluation'),
        help='Directorio de salida para resultados'
    )
    parser.add_argument(
        '--base_dir',
        type=Path,
        default=Path('.'),
        help='Directorio base para resolver paths relativos de imágenes'
    )
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("EVALUACIÓN DEL CLASIFICADOR EMOCIONAL")
    logger.info("="*60)
    logger.info(f"Dataset CSV:   {args.dataset_csv}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info(f"Base dir:      {args.base_dir}")
    logger.info("="*60 + "\n")
    
    try:
        # 1. Cargar dataset
        dataset = load_ground_truth_dataset(args.dataset_csv)
        
        # 2. Inicializar detector
        logger.info("Inicializando detector DeepFace...")
        detector = DeepFaceEmotionDetector(enforce_detection=False)
        logger.info("[OK] Detector inicializado\n")
        
        # 3. Evaluar detector
        y_true, y_pred = evaluate_detector(dataset, detector, args.base_dir)
        
        if len(y_true) == 0:
            logger.error("No se pudo procesar ninguna imagen. Abortando.")
            return
        
        # 4. Calcular métricas
        metrics = calculate_metrics(y_true, y_pred)
        
        # 5. Exportar resultados
        logger.info("Guardando resultados...\n")
        
        # Confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        labels = metrics['labels']
        
        save_confusion_matrix_csv(cm, labels, args.output_dir / 'confusion_matrix.csv')
        save_confusion_matrix_heatmap(cm, labels, args.output_dir / 'confusion_matrix.png')
        
        # Classification report
        save_classification_report_csv(metrics, args.output_dir / 'classification_report.csv')
        save_classification_report_latex(metrics, args.output_dir / 'classification_report.tex')
        
        # Summary JSON
        save_metrics_summary_json(metrics, args.output_dir / 'metrics_summary.json')
        
        logger.info("\n" + "="*60)
        logger.info("EVALUACIÓN COMPLETADA")
        logger.info("="*60)
        logger.info(f"Resultados guardados en: {args.output_dir}")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
