"""
Script para analizar estadísticas del subset Lakh MIDI preparado.

Lee el CSV de metadata y genera análisis estadístico y visualizaciones.

Uso:
    python scripts/analyze_lakh_subset.py --metadata_csv data/lakh_metadata/lakh_subset_metadata.csv

Author: Miguel (TFM Generación Musical Adaptativa)
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_metadata(csv_path: Path) -> List[Dict]:
    """Carga metadata desde CSV (compatible con cualquier formato)."""
    metadata = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Solo cargar archivos aceptados (sin discarded_reason o vacío)
            if row.get('discarded_reason', '').strip():
                continue
            
            # Convertir strings a números (con manejo de campos opcionales)
            entry = {
                'filename': row['filename'],
                'duration_seconds': float(row['duration_seconds']),
                'total_notes': int(row['total_notes']),
                'mean_velocity': float(row['mean_velocity']),
                'pitch_range': int(row['pitch_range']),
                'min_pitch': int(row['min_pitch']),
                'max_pitch': int(row['max_pitch']),
            }
            
            # Campos opcionales (dependen del tipo de subset)
            if 'num_tracks' in row:
                entry['num_tracks'] = int(row['num_tracks'])
            if 'original_tracks' in row:
                entry['original_tracks'] = int(row['original_tracks'])
            if 'piano_tracks_found' in row:
                entry['piano_tracks_found'] = int(row['piano_tracks_found'])
            
            metadata.append(entry)
    
    return metadata


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Calcula estadísticas descriptivas."""
    arr = np.array(values)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr, ddof=1),
        'min': np.min(arr),
        'max': np.max(arr),
        'median': np.median(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75)
    }


def analyze_subset(metadata: List[Dict]):
    """Analiza y muestra estadísticas del subset."""
    
    n_files = len(metadata)
    
    # Extraer features
    durations = [m['duration_seconds'] for m in metadata]
    total_notes = [m['total_notes'] for m in metadata]
    velocities = [m['mean_velocity'] for m in metadata]
    pitch_ranges = [m['pitch_range'] for m in metadata]
    
    # Features opcionales
    has_num_tracks = 'num_tracks' in metadata[0]
    has_piano_info = 'piano_tracks_found' in metadata[0]
    
    if has_num_tracks:
        num_tracks = [m['num_tracks'] for m in metadata]
    if has_piano_info:
        original_tracks = [m['original_tracks'] for m in metadata]
        piano_tracks = [m['piano_tracks_found'] for m in metadata]
    
    # Calcular estadísticas
    duration_stats = compute_statistics(durations)
    notes_stats = compute_statistics(total_notes)
    velocity_stats = compute_statistics(velocities)
    pitch_range_stats = compute_statistics(pitch_ranges)
    
    if has_num_tracks:
        tracks_stats = compute_statistics(num_tracks)
    if has_piano_info:
        original_tracks_stats = compute_statistics(original_tracks)
        piano_tracks_stats = compute_statistics(piano_tracks)
    
    # Imprimir resumen
    logger.info("\n" + "="*70)
    logger.info("ANÁLISIS ESTADÍSTICO - LAKH MIDI SUBSET")
    logger.info("="*70)
    logger.info(f"Total de archivos: {n_files}")
    logger.info("")
    
    # Duración
    logger.info("DURACIÓN (segundos)")
    logger.info("-" * 70)
    logger.info(f"  Media:    {duration_stats['mean']:>8.2f}s  ±  {duration_stats['std']:.2f}")
    logger.info(f"  Mediana:  {duration_stats['median']:>8.2f}s")
    logger.info(f"  Rango:    {duration_stats['min']:>8.2f}s  -  {duration_stats['max']:.2f}s")
    logger.info(f"  Q25-Q75:  {duration_stats['q25']:>8.2f}s  -  {duration_stats['q75']:.2f}s")
    logger.info("")
    
    # Número de notas
    logger.info("NÚMERO DE NOTAS")
    logger.info("-" * 70)
    logger.info(f"  Media:    {notes_stats['mean']:>8.1f}  ±  {notes_stats['std']:.1f}")
    logger.info(f"  Mediana:  {notes_stats['median']:>8.0f}")
    logger.info(f"  Rango:    {notes_stats['min']:>8.0f}  -  {notes_stats['max']:.0f}")
    logger.info(f"  Q25-Q75:  {notes_stats['q25']:>8.0f}  -  {notes_stats['q75']:.0f}")
    logger.info("")
    
    # Tracks (si existe)
    if has_num_tracks:
        logger.info("NÚMERO DE TRACKS")
        logger.info("-" * 70)
        logger.info(f"  Media:    {tracks_stats['mean']:>8.1f}  ±  {tracks_stats['std']:.1f}")
        logger.info(f"  Mediana:  {tracks_stats['median']:>8.0f}")
        logger.info(f"  Rango:    {tracks_stats['min']:>8.0f}  -  {tracks_stats['max']:.0f}")
        logger.info("")
    
    # Info piano (si existe)
    if has_piano_info:
        logger.info("TRACKS ORIGINALES (antes de filtrar)")
        logger.info("-" * 70)
        logger.info(f"  Media:    {original_tracks_stats['mean']:>8.1f}  ±  {original_tracks_stats['std']:.1f}")
        logger.info(f"  Mediana:  {original_tracks_stats['median']:>8.0f}")
        logger.info("")
        logger.info("TRACKS DE PIANO ENCONTRADOS")
        logger.info("-" * 70)
        logger.info(f"  Media:    {piano_tracks_stats['mean']:>8.1f}  ±  {piano_tracks_stats['std']:.1f}")
        logger.info(f"  Mediana:  {piano_tracks_stats['median']:>8.0f}")
        logger.info("")
    
    # Velocity
    logger.info("VELOCITY PROMEDIO")
    logger.info("-" * 70)
    logger.info(f"  Media:    {velocity_stats['mean']:>8.1f}  ±  {velocity_stats['std']:.1f}")
    logger.info(f"  Mediana:  {velocity_stats['median']:>8.1f}")
    logger.info(f"  Rango:    {velocity_stats['min']:>8.1f}  -  {velocity_stats['max']:.1f}")
    logger.info("")
    
    # Pitch range
    logger.info("RANGO TONAL (semitonos)")
    logger.info("-" * 70)
    logger.info(f"  Media:    {pitch_range_stats['mean']:>8.1f}  ±  {pitch_range_stats['std']:.1f}")
    logger.info(f"  Mediana:  {pitch_range_stats['median']:>8.0f}")
    logger.info(f"  Rango:    {pitch_range_stats['min']:>8.0f}  -  {pitch_range_stats['max']:.0f}")
    logger.info("")
    
    # Duración total del corpus
    total_duration_hours = sum(durations) / 3600
    logger.info("CORPUS COMPLETO")
    logger.info("-" * 70)
    logger.info(f"  Duración total:   {total_duration_hours:.2f} horas")
    logger.info(f"  Notas totales:    {sum(total_notes):,}")
    logger.info("")
    
    # Distribuciones
    logger.info("DISTRIBUCIÓN DE DURACIÓN")
    logger.info("-" * 70)
    duration_bins = {
        '10-30s': len([d for d in durations if 10 <= d < 30]),
        '30-60s': len([d for d in durations if 30 <= d < 60]),
        '1-2min': len([d for d in durations if 60 <= d < 120]),
        '2-3min': len([d for d in durations if 120 <= d < 180]),
        '3-5min': len([d for d in durations if 180 <= d <= 300]),
    }
    
    for bin_name, count in duration_bins.items():
        pct = (count / n_files * 100)
        bar = '█' * int(pct / 2)
        logger.info(f"  {bin_name:>10}:  {count:>5} ({pct:>5.1f}%)  {bar}")
    
    logger.info("")
    logger.info("="*70)
    
    # Evaluación de calidad
    logger.info("\nEVALUACIÓN DE CALIDAD")
    logger.info("="*70)
    
    quality_checks = []
    
    # Check 1: Variedad de duración
    duration_cv = duration_stats['std'] / duration_stats['mean']
    if duration_cv > 0.5:
        quality_checks.append(("OK", "Buena variedad de duraciones"))
    else:
        quality_checks.append(("⚠", "Poca variedad de duraciones"))
    
    # Check 2: Rango tonal
    if pitch_range_stats['mean'] > 40:
        quality_checks.append(("OK", "Buen rango tonal promedio"))
    else:
        quality_checks.append(("⚠", "Rango tonal limitado"))
    
    # Check 3: Densidad de notas
    avg_density = notes_stats['mean'] / duration_stats['mean']
    if 1.0 < avg_density < 10.0:
        quality_checks.append(("OK", f"Densidad razonable ({avg_density:.2f} notas/s)"))
    else:
        quality_checks.append(("⚠", f"Densidad inusual ({avg_density:.2f} notas/s)"))
    
    # Check 4: Tamaño del corpus
    if n_files >= 1000:
        quality_checks.append(("OK", f"Corpus suficientemente grande ({n_files} archivos)"))
    else:
        quality_checks.append(("⚠", f"Corpus pequeño ({n_files} archivos, recomendado: >1000)"))
    
    # Check 5: Duración total
    if total_duration_hours > 5:
        quality_checks.append(("OK", f"Duración total adecuada ({total_duration_hours:.1f}h)"))
    else:
        quality_checks.append(("⚠", f"Duración total limitada ({total_duration_hours:.1f}h)"))
    
    for symbol, message in quality_checks:
        logger.info(f"{symbol} {message}")
    
    logger.info("="*70 + "\n")
    
    # Recomendaciones
    logger.info("RECOMENDACIONES PARA FINE-TUNING")
    logger.info("="*70)
    
    if n_files < 1000:
        logger.info("⚠ Considera aumentar el subset a al menos 1000-2000 archivos")
    
    if total_duration_hours < 10:
        logger.info("⚠ Para mejor fine-tuning, apunta a 10-20 horas de audio total")
    
    if pitch_range_stats['mean'] < 36:
        logger.info("⚠ Rango tonal bajo. Considera filtrar por pitch_range > 36")
    
    logger.info("Dataset listo para etiquetado heurístico V/A")
    logger.info("Dataset listo para tokenización REMI")
    logger.info("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analizar estadísticas del subset Lakh MIDI preparado'
    )
    
    parser.add_argument(
        '--metadata_csv',
        type=str,
        default='data/lakh_piano_metadata/lakh_piano_subset_metadata.csv',
        help='Ruta al CSV de metadata (default: data/lakh_piano_metadata/lakh_piano_subset_metadata.csv)'
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.metadata_csv)
    
    if not csv_path.exists():
        logger.error(f"ERROR: {csv_path} no existe")
        logger.info("\nPrimero ejecuta:")
        logger.info("  python scripts/prepare_lakh_piano_subset.py --input_dir data/lakh_raw --output_dir data/lakh_piano_clean")
        return
    
    logger.info(f"Cargando metadata desde: {csv_path}")
    metadata = load_metadata(csv_path)
    
    if not metadata:
        logger.error("ERROR: CSV vacío o sin datos")
        return
    
    analyze_subset(metadata)


if __name__ == '__main__':
    main()
