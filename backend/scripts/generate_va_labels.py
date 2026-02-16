#!/usr/bin/env python3
"""
Generador de labels VA heurísticas para el subset Lakh Piano.

Este script toma el archivo de metadata musical y genera labels de Valence y Arousal
usando heurísticas basadas en características musicales:
- Arousal: basado en tempo, velocity, density
- Valence: basado en pitch range, velocity, duration

Entrada: lakh_piano_subset_metadata.csv
Salida: lakh_piano_subset_labeled.csv (con columnas valence, arousal)

Uso:
    python scripts/generate_va_labels.py
    python scripts/generate_va_labels.py --output data/lakh_piano_metadata/custom_labels.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def compute_va_heuristic(row: pd.Series) -> tuple[float, float]:
    """
    Calcula valence y arousal heurístico desde características musicales.
    
    Heurísticas:
    - Arousal: mayor tempo, velocity y density → mayor arousal
    - Valence: mayor velocity, pitch medio-alto, menor varianza → mayor valence
    
    Args:
        row: Fila del DataFrame con metadata musical
        
    Returns:
        Tupla (valence, arousal) en rango [-1, 1]
    """
    # Extraer características (con defaults seguros)
    mean_velocity = row.get('mean_velocity', 64.0)
    pitch_range = row.get('pitch_range', 36)
    min_pitch = row.get('min_pitch', 48)
    max_pitch = row.get('max_pitch', 84)
    total_notes = row.get('total_notes', 100)
    duration_seconds = row.get('duration_seconds', 60.0)
    
    # Evitar división por cero
    if duration_seconds < 1.0:
        duration_seconds = 1.0
    
    # Calcular density (notas por segundo)
    density = total_notes / duration_seconds
    
    # Estimar tempo desde duration y total_notes (asumiendo 4/4)
    # Heurística: más notas por segundo ~ mayor tempo
    estimated_tempo = min(180, max(60, 60 + density * 10))
    
    # AROUSAL: energía musical (0-1 antes de normalizar)
    # - Tempo: 60-180 bpm → 0-1
    # - Velocity: 0-127 → 0-1
    # - Density: 0-10 notas/s → 0-1
    tempo_norm = (estimated_tempo - 60) / 120
    velocity_norm = mean_velocity / 127.0
    density_norm = min(1.0, density / 10.0)
    
    arousal_raw = 0.4 * tempo_norm + 0.3 * velocity_norm + 0.3 * density_norm
    
    # VALENCE: "positividad" musical (0-1 antes de normalizar)
    # - Velocity alta (energía) correlaciona con valence positivo
    # - Pitch medio-alto (60-80) correlaciona con valence positivo
    # - Pitch range moderado (ni muy estrecho ni muy amplio) → más estable → más valence
    pitch_mean = (min_pitch + max_pitch) / 2
    pitch_mean_norm = (pitch_mean - 48) / 48  # 48-96 → 0-1
    
    # Range ideal ~24 semitonos (2 octavas)
    range_deviation = abs(pitch_range - 24) / 24
    range_score = max(0, 1.0 - range_deviation)
    
    valence_raw = 0.4 * velocity_norm + 0.3 * pitch_mean_norm + 0.3 * range_score
    
    # Normalizar a [-1, 1] con algo de varianza
    # Añadir pequeño ruido gaussiano para evitar valores idénticos
    arousal = np.clip(arousal_raw * 2 - 1 + np.random.normal(0, 0.05), -1, 1)
    valence = np.clip(valence_raw * 2 - 1 + np.random.normal(0, 0.05), -1, 1)
    
    return float(valence), float(arousal)


def main():
    parser = argparse.ArgumentParser(description='Generar labels VA heurísticas')
    parser.add_argument(
        '--input',
        default='data/lakh_piano_metadata/lakh_piano_subset_metadata.csv',
        help='CSV de entrada con metadata musical'
    )
    parser.add_argument(
        '--output',
        default='data/lakh_piano_metadata/lakh_piano_subset_labeled.csv',
        help='CSV de salida con labels VA'
    )
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    
    args = parser.parse_args()
    
    # Configurar semilla
    np.random.seed(args.seed)
    
    # Paths
    backend_dir = Path(__file__).parent.parent
    input_path = backend_dir / args.input
    output_path = backend_dir / args.output
    
    logger.info(f"Leyendo metadata desde: {input_path}")
    
    # Leer CSV
    if not input_path.exists():
        logger.error(f"Archivo no encontrado: {input_path}")
        return 1
    
    df = pd.read_csv(input_path)
    logger.info(f"Cargadas {len(df)} filas")
    
    # Filtrar solo archivos válidos (sin discarded_reason)
    df_valid = df[df['discarded_reason'].isna() | (df['discarded_reason'] == '')].copy()
    logger.info(f"Archivos válidos (sin discarded_reason): {len(df_valid)}")
    
    # Generar labels VA
    logger.info("Generando labels VA heurísticas...")
    va_labels = df_valid.apply(compute_va_heuristic, axis=1)
    
    # Separar en columnas
    df_valid['valence'] = va_labels.apply(lambda x: x[0])
    df_valid['arousal'] = va_labels.apply(lambda x: x[1])
    
    # Estadísticas
    logger.info(f"Estadísticas VA:")
    logger.info(f"  Valence: mean={df_valid['valence'].mean():.3f}, std={df_valid['valence'].std():.3f}")
    logger.info(f"  Arousal: mean={df_valid['arousal'].mean():.3f}, std={df_valid['arousal'].std():.3f}")
    
    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_valid.to_csv(output_path, index=False)
    
    logger.info(f"Labels VA guardadas en: {output_path}")
    logger.info(f"  Total archivos etiquetados: {len(df_valid)}")
    
    return 0


if __name__ == '__main__':
    exit(main())
