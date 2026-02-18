"""
Script para preparar subconjunto piano-only del Lakh MIDI Dataset.

Filtra y extrae solo pistas de piano, mergea en un solo track,
y genera subset limpio compatible con Maestro-REMI-bpe20k.

Uso:
    python scripts/prepare_lakh_piano_subset.py \
        --input_dir data/lakh_raw \
        --output_dir data/lakh_piano_clean \
        --max_files 5000 \
        --seed 42

Author: Miguel (TFM Generación Musical Adaptativa)
"""

import argparse
import csv
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pretty_midi

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Program numbers de piano en General MIDI (0-7)
PIANO_PROGRAMS = list(range(8))  # 0-7: Acoustic Grand Piano to Clavinet


# ============================================================================
# FUNCIONES DE DETECCIÓN DE PIANO
# ============================================================================

def is_piano_instrument(instrument: pretty_midi.Instrument) -> bool:
    """
    Determina si un instrumento es piano.
    
    Criterios:
    - No es percusión
    - Program number 0-7 (pianos en GM)
    - O nombre contiene "piano"
    
    Args:
        instrument: Instrumento de PrettyMIDI
    
    Returns:
        True si es piano
    """
    # Descartar percusión
    if instrument.is_drum:
        return False
    
    # Check program number (0-7 son pianos)
    if instrument.program in PIANO_PROGRAMS:
        return True
    
    # Check nombre
    if instrument.name and 'piano' in instrument.name.lower():
        return True
    
    return False


def extract_piano_only_midi(midi_path: Path) -> Tuple[Optional[pretty_midi.PrettyMIDI], Dict, Optional[str]]:
    """
    Extrae solo pistas de piano de un MIDI y las mergea en un solo track.
    
    Args:
        midi_path: Ruta al MIDI original
    
    Returns:
        Tupla (piano_midi, metadata, error_reason)
        - piano_midi: PrettyMIDI con solo piano o None si error
        - metadata: Dict con info del procesamiento
        - error_reason: Razón de descarte o None si OK
    """
    metadata = {
        'filename': midi_path.name,
        'original_tracks': 0,
        'piano_tracks_found': 0,
        'total_notes': 0,
        'duration_seconds': 0.0,
        'mean_velocity': 0.0,
        'pitch_range': 0,
        'min_pitch': 0,
        'max_pitch': 0,
        'discarded_reason': ''
    }
    
    # Intentar cargar MIDI
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        logger.debug(f"Error al cargar {midi_path.name}: {e}")
        metadata['discarded_reason'] = 'corrupted'
        return None, metadata, 'corrupted'
    
    metadata['original_tracks'] = len(midi.instruments)
    
    # Encontrar pistas de piano
    piano_instruments = [inst for inst in midi.instruments if is_piano_instrument(inst)]
    metadata['piano_tracks_found'] = len(piano_instruments)
    
    if not piano_instruments:
        metadata['discarded_reason'] = 'no_piano'
        return None, metadata, 'no_piano'
    
    # Crear nuevo MIDI con solo piano
    piano_midi = pretty_midi.PrettyMIDI()
    
    # Crear instrumento piano único (program 0: Acoustic Grand Piano)
    merged_piano = pretty_midi.Instrument(program=0, name='Piano')
    
    # Mergear todas las notas de piano
    all_notes = []
    for piano_inst in piano_instruments:
        all_notes.extend(piano_inst.notes)
    
    # Ordenar notas por tiempo de inicio
    all_notes.sort(key=lambda n: n.start)
    
    # Asignar al instrumento mergeado
    merged_piano.notes = all_notes
    
    # Añadir al nuevo MIDI
    piano_midi.instruments.append(merged_piano)
    
    # Calcular metadata del MIDI resultante
    if not all_notes:
        metadata['discarded_reason'] = 'no_notes'
        return None, metadata, 'no_notes'
    
    metadata['total_notes'] = len(all_notes)
    metadata['duration_seconds'] = piano_midi.get_end_time()
    
    # Features adicionales
    pitches = [note.pitch for note in all_notes]
    velocities = [note.velocity for note in all_notes]
    
    metadata['min_pitch'] = min(pitches)
    metadata['max_pitch'] = max(pitches)
    metadata['pitch_range'] = metadata['max_pitch'] - metadata['min_pitch']
    metadata['mean_velocity'] = sum(velocities) / len(velocities)
    
    return piano_midi, metadata, None


# ============================================================================
# FUNCIONES DE FILTRADO
# ============================================================================

def should_accept_piano_midi(
    metadata: Dict,
    min_duration: float,
    max_duration: float,
    min_notes: int
) -> Tuple[bool, Optional[str]]:
    """
    Determina si un MIDI piano debe ser aceptado.
    
    Args:
        metadata: Metadata del MIDI
        min_duration: Duración mínima en segundos
        max_duration: Duración máxima en segundos
        min_notes: Número mínimo de notas
    
    Returns:
        Tupla (should_accept, rejection_reason)
    """
    duration = metadata['duration_seconds']
    total_notes = metadata['total_notes']
    
    # Verificar duración
    if duration < min_duration:
        return False, f"too_short_{duration:.1f}s"
    
    if duration > max_duration:
        return False, f"too_long_{duration:.1f}s"
    
    # Verificar número de notas
    if total_notes < min_notes:
        return False, f"too_few_notes_{total_notes}"
    
    # Verificar rango tonal mínimo (al menos 1 octava)
    if metadata['pitch_range'] < 12:
        return False, f"narrow_range_{metadata['pitch_range']}"
    
    return True, None


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def prepare_lakh_piano_subset(
    input_dir: Path,
    output_dir: Path,
    metadata_dir: Path,
    max_files: int = 5000,
    min_duration: float = 10.0,
    max_duration: float = 300.0,
    min_notes: int = 50,
    seed: int = 42
) -> Dict[str, int]:
    """
    Prepara subset piano-only de Lakh MIDI.
    
    Args:
        input_dir: Directorio con MIDIs originales
        output_dir: Directorio de salida para MIDIs piano
        metadata_dir: Directorio para metadata CSV
        max_files: Número máximo de archivos a aceptar
        min_duration: Duración mínima en segundos
        max_duration: Duración máxima en segundos
        min_notes: Número mínimo de notas
        seed: Seed para reproducibilidad
    
    Returns:
        Diccionario con estadísticas del proceso
    """
    # Crear directorios
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Estadísticas
    stats = {
        'total_scanned': 0,
        'corrupted': 0,
        'no_piano': 0,
        'no_notes': 0,
        'too_short': 0,
        'too_long': 0,
        'too_few_notes': 0,
        'narrow_range': 0,
        'accepted': 0
    }
    
    # Lista para metadata
    all_metadata = []
    
    # Buscar todos los MIDIs
    logger.info(f"Buscando MIDIs en {input_dir}...")
    midi_files = list(input_dir.rglob('*.mid')) + list(input_dir.rglob('*.midi'))
    
    if not midi_files:
        logger.error(f"No se encontraron archivos MIDI en {input_dir}")
        return stats
    
    # Reproducibilidad: ordenar y samplear si necesario
    midi_files.sort(key=lambda p: str(p))  # Orden alfabético
    
    random.seed(seed)
    if len(midi_files) > max_files * 10:  # Si hay muchos más, samplear para eficiencia
        logger.info(f"Sampleando {max_files * 10} de {len(midi_files)} archivos (seed={seed})")
        midi_files = random.sample(midi_files, max_files * 10)
        midi_files.sort(key=lambda p: str(p))  # Re-ordenar después de samplear
    
    stats['total_scanned'] = len(midi_files)
    logger.info(f"Procesando {stats['total_scanned']} archivos...")
    logger.info(f"Límite de aceptación: {max_files}")
    logger.info("")
    
    # Procesar archivos
    for idx, midi_path in enumerate(midi_files, 1):
        # Verificar si ya alcanzamos el límite
        if stats['accepted'] >= max_files:
            logger.info(f"Límite de {max_files} archivos alcanzado. Deteniendo.")
            break
        
        # Log de progreso
        if idx % 100 == 0:
            logger.info(f"Progreso: {idx}/{stats['total_scanned']} | Aceptados: {stats['accepted']}/{max_files}")
        
        # Extraer piano-only MIDI
        piano_midi, metadata, initial_error = extract_piano_only_midi(midi_path)
        
        # Registrar metadata inicial (aunque se descarte)
        all_metadata.append(metadata)
        
        # Si hubo error en extracción
        if initial_error:
            if initial_error == 'corrupted':
                stats['corrupted'] += 1
            elif initial_error == 'no_piano':
                stats['no_piano'] += 1
            elif initial_error == 'no_notes':
                stats['no_notes'] += 1
            
            logger.debug(f"Descartado: {midi_path.name} - {initial_error}")
            continue
        
        # Aplicar filtros de calidad
        should_accept, reject_reason = should_accept_piano_midi(
            metadata, min_duration, max_duration, min_notes
        )
        
        if not should_accept:
            metadata['discarded_reason'] = reject_reason
            
            # Categorizar rechazo
            if 'too_short' in reject_reason:
                stats['too_short'] += 1
            elif 'too_long' in reject_reason:
                stats['too_long'] += 1
            elif 'too_few_notes' in reject_reason:
                stats['too_few_notes'] += 1
            elif 'narrow_range' in reject_reason:
                stats['narrow_range'] += 1
            
            logger.debug(f"Rechazado: {midi_path.name} - {reject_reason}")
            continue
        
        # Aceptar: guardar MIDI piano
        try:
            output_path = output_dir / midi_path.name
            
            # Si hay conflicto de nombre, añadir sufijo
            if output_path.exists():
                stem = midi_path.stem
                output_path = output_dir / f"{stem}_piano_{stats['accepted']}.mid"
            
            piano_midi.write(str(output_path))
            
            # Actualizar metadata
            metadata['discarded_reason'] = ''  # Aceptado
            
            stats['accepted'] += 1
            logger.debug(
                f"Aceptado: {midi_path.name} "
                f"({metadata['duration_seconds']:.1f}s, "
                f"{metadata['total_notes']} notas, "
                f"{metadata['piano_tracks_found']} piano tracks)"
            )
        
        except Exception as e:
            logger.error(f"Error al guardar {midi_path.name}: {e}")
            metadata['discarded_reason'] = 'save_error'
            continue
    
    # Guardar metadata a CSV
    csv_path = metadata_dir / 'lakh_piano_subset_metadata.csv'
    logger.info(f"\nGuardando metadata en {csv_path}...")
    
    if all_metadata:
        fieldnames = [
            'filename', 'duration_seconds', 'total_notes', 'mean_velocity',
            'pitch_range', 'min_pitch', 'max_pitch', 'original_tracks',
            'piano_tracks_found', 'discarded_reason'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metadata)
    
    # Imprimir resumen
    logger.info("\n" + "="*70)
    logger.info("RESUMEN - SUBSET PIANO-ONLY")
    logger.info("="*70)
    logger.info(f"Total escaneados:         {stats['total_scanned']}")
    logger.info(f"Corruptos:                {stats['corrupted']}")
    logger.info(f"Sin piano:                {stats['no_piano']}")
    logger.info(f"Sin notas:                {stats['no_notes']}")
    logger.info(f"Demasiado cortos:         {stats['too_short']}")
    logger.info(f"Demasiado largos:         {stats['too_long']}")
    logger.info(f"Pocas notas:              {stats['too_few_notes']}")
    logger.info(f"Rango tonal estrecho:     {stats['narrow_range']}")
    logger.info(f"ACEPTADOS:                {stats['accepted']}")
    logger.info("="*70)
    
    if stats['total_scanned'] > 0:
        acceptance_rate = (stats['accepted'] / stats['total_scanned'] * 100)
        logger.info(f"Tasa de aceptación:       {acceptance_rate:.1f}%")
    
    logger.info(f"Archivos en {output_dir}: {stats['accepted']}")
    logger.info(f"Metadata guardada en:     {csv_path}")
    logger.info("="*70 + "\n")
    
    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preparar subset piano-only del Lakh MIDI Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    # Procesamiento estándar
    python scripts/prepare_lakh_piano_subset.py \\
        --input_dir data/lakh_raw \\
        --output_dir data/lakh_piano_clean

    # Con parámetros personalizados
    python scripts/prepare_lakh_piano_subset.py \\
        --input_dir data/lakh_raw/lmd_matched \\
        --output_dir data/lakh_piano_clean \\
        --max_files 3000 \\
        --min_notes 100 \\
        --seed 42

Detección de piano:
    - Program 0-7 (General MIDI piano sounds)
    - O nombre de track contiene "piano"
    
Procesamiento:
    - Extrae todas las pistas de piano
    - Mergea en un solo track (program 0)
    - Filtra por duración y cantidad de notas
    - Compatible con Maestro-REMI-bpe20k
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directorio con archivos MIDI originales'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directorio de salida para MIDIs piano-only'
    )
    
    parser.add_argument(
        '--metadata_dir',
        type=str,
        default=None,
        help='Directorio para metadata CSV (default: data/lakh_piano_metadata)'
    )
    
    parser.add_argument(
        '--max_files',
        type=int,
        default=5000,
        help='Número máximo de archivos a aceptar (default: 5000)'
    )
    
    parser.add_argument(
        '--min_duration',
        type=float,
        default=10.0,
        help='Duración mínima en segundos (default: 10.0)'
    )
    
    parser.add_argument(
        '--max_duration',
        type=float,
        default=300.0,
        help='Duración máxima en segundos (default: 300.0)'
    )
    
    parser.add_argument(
        '--min_notes',
        type=int,
        default=50,
        help='Número mínimo de notas (default: 50)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed para reproducibilidad (default: 42)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Logs detallados (DEBUG level)'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convertir a Path
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if args.metadata_dir:
        metadata_dir = Path(args.metadata_dir)
    else:
        metadata_dir = Path('data/lakh_piano_metadata')
    
    # Validar input_dir existe
    if not input_dir.exists():
        logger.error(f"ERROR: {input_dir} no existe")
        logger.info(f"\nPara obtener el Lakh MIDI Dataset:")
        logger.info(f"Ver: DATASET_PREPARATION.md")
        return
    
    # Ejecutar preparación
    logger.info("Iniciando preparación de Lakh Piano subset...")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Límite: {args.max_files} archivos")
    logger.info(f"Duración: {args.min_duration}s - {args.max_duration}s")
    logger.info(f"Notas mínimas: {args.min_notes}")
    logger.info(f"Seed: {args.seed}\n")
    
    stats = prepare_lakh_piano_subset(
        input_dir=input_dir,
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        max_files=args.max_files,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_notes=args.min_notes,
        seed=args.seed
    )
    
    if stats['accepted'] > 0:
        logger.info("Proceso completado exitosamente")
        logger.info(f"\nPróximos pasos:")
        logger.info(f"1. Analizar subset: python scripts/analyze_lakh_subset.py --metadata_csv {metadata_dir / 'lakh_piano_subset_metadata.csv'}")
        logger.info(f"2. Etiquetar V/A heurísticamente")
        logger.info(f"3. Tokenizar con REMI")
        logger.info(f"4. Fine-tuning Maestro-REMI-bpe20k")
    else:
        logger.warning("⚠ No se encontraron archivos piano válidos")


if __name__ == '__main__':
    main()
