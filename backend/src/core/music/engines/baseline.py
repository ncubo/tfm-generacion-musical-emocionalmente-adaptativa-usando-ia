"""
Generador MIDI baseline basado en reglas.

Este módulo implementa un generador simple de música MIDI que convierte
parámetros musicales (derivados de emociones) en secuencias MIDI monofónicas
siguiendo reglas deterministas.

El generador es un baseline para evaluación posterior, sin usar modelos
de aprendizaje automático, solo lógica composicional explícita.
"""

import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage


# Definición de escalas musicales (intervalos en semitonos desde la tónica)
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],      # Escala mayor (Jónico)
    'minor': [0, 2, 3, 5, 7, 8, 10]       # Escala menor natural (Eólico)
}

# Subdivisiones rítmicas (en ticks por nota, 480 ticks = 1 beat)
RHYTHM_PATTERNS = {
    'simple': [960, 480, 480],                    # Blancas y negras
    'moderate': [480, 480, 240, 240],             # Negras y corcheas
    'complex': [240, 240, 240, 120, 120, 240]     # Corcheas y semicorcheas
}


def _select_rhythm_pattern(complexity: float) -> List[int]:
    """
    Selecciona un patrón rítmico basado en la complejidad.
    
    Args:
        complexity (float): Complejidad rítmica en [0, 1]
    
    Returns:
        List[int]: Lista de duraciones en ticks
    """
    if complexity < 0.33:
        return RHYTHM_PATTERNS['simple']
    elif complexity < 0.67:
        return RHYTHM_PATTERNS['moderate']
    else:
        return RHYTHM_PATTERNS['complex']


def _build_scale_notes(
    mode: str,
    pitch_low: int,
    pitch_high: int,
    root_note: int = 60
) -> List[int]:
    """
    Construye una lista de notas MIDI válidas según la escala y rango.
    
    Args:
        mode (str): Modo musical ('major' o 'minor')
        pitch_low (int): Nota MIDI más grave permitida
        pitch_high (int): Nota MIDI más aguda permitida
        root_note (int): Tónica de la escala (default C4 = 60)
    
    Returns:
        List[int]: Lista de notas MIDI válidas ordenadas
    """
    scale_intervals = SCALES.get(mode, SCALES['major'])
    notes = []
    
    # Generar notas en múltiples octavas
    for octave_offset in range(-3, 4):  # 7 octavas
        for interval in scale_intervals:
            note = root_note + octave_offset * 12 + interval
            if pitch_low <= note <= pitch_high:
                notes.append(note)
    
    return sorted(list(set(notes)))


def _generate_melody(
    scale_notes: List[int],
    density: float,
    rhythm_pattern: List[int],
    velocity_mean: int,
    velocity_spread: int,
    length_ticks: int,
    rng: random.Random
) -> List[Tuple[int, int, int]]:
    """
    Genera una secuencia melódica simple.
    
    La generación sigue un random walk en la escala con restricciones:
    - Los saltos grandes son menos probables
    - La densidad controla el porcentaje de notas vs silencios
    
    Args:
        scale_notes (List[int]): Notas MIDI disponibles
        density (float): Densidad de notas en [0, 1]
        rhythm_pattern (List[int]): Duraciones rítmicas en ticks
        velocity_mean (int): Velocity media
        velocity_spread (int): Variación de velocity
        length_ticks (int): Duración total en ticks
        rng (random.Random): Generador aleatorio
    
    Returns:
        List[Tuple[int, int, int]]: Lista de (nota, velocity, duración)
    """
    melody = []
    current_tick = 0
    current_note_idx = len(scale_notes) // 2  # Empezar en el centro
    
    while current_tick < length_ticks:
        # Seleccionar duración del patrón rítmico
        duration = rng.choice(rhythm_pattern)
        
        # Decidir si poner nota o silencio según densidad
        if rng.random() < density:
            # Generar nota
            note = scale_notes[current_note_idx]
            
            # Calcular velocity con variación
            velocity = int(velocity_mean + rng.randint(-velocity_spread, velocity_spread))
            velocity = max(1, min(127, velocity))  # Clamp [1, 127]
            
            melody.append((note, velocity, duration))
            
            # Random walk: mover índice de nota (preferir saltos pequeños)
            step = rng.choices(
                [-2, -1, 0, 1, 2],
                weights=[0.1, 0.3, 0.2, 0.3, 0.1]
            )[0]
            current_note_idx = max(0, min(len(scale_notes) - 1, current_note_idx + step))
        else:
            # Silencio: no añadir nota, solo avanzar tiempo
            melody.append((None, 0, duration))
        
        current_tick += duration
    
    return melody


def generate_midi_baseline(
    params: Dict[str, any],
    out_path: str,
    length_bars: int = 8,
    seed: Optional[int] = None
) -> str:
    """
    Genera un archivo MIDI baseline a partir de parámetros musicales.
    
    Este generador implementa un sistema simple basado en reglas que traduce
    parámetros emocionales a música MIDI reproducible. El resultado es
    monofónico (una sola voz melódica) y determinista si se proporciona seed.
    
    Args:
        params (Dict[str, any]): Parámetros musicales con las claves:
            - tempo_bpm (int): Tempo en beats por minuto
            - mode (str): Modo musical 'major' o 'minor'
            - density (float): Densidad de notas [0, 1]
            - pitch_low (int): Nota MIDI más grave
            - pitch_high (int): Nota MIDI más aguda
            - rhythm_complexity (float): Complejidad rítmica [0, 1]
            - velocity_mean (int): Velocity media [1, 127]
            - velocity_spread (int): Variación de velocity [0, 127]
        out_path (str): Ruta donde guardar el archivo MIDI
        length_bars (int): Número de compases a generar (default: 8)
        seed (Optional[int]): Semilla para reproducibilidad (default: None)
    
    Returns:
        str: Ruta absoluta del archivo MIDI generado
    
    Raises:
        ValueError: Si los parámetros son inválidos
        IOError: Si no se puede escribir el archivo
    
    Example:
        >>> params = {
        ...     'tempo_bpm': 120,
        ...     'mode': 'major',
        ...     'density': 0.8,
        ...     'pitch_low': 60,
        ...     'pitch_high': 72,
        ...     'rhythm_complexity': 0.5,
        ...     'velocity_mean': 80,
        ...     'velocity_spread': 20
        ... }
        >>> path = generate_midi_baseline(params, 'output.mid', length_bars=4)
        >>> print(f"MIDI generado: {path}")
    """
    # Validar parámetros obligatorios
    required_keys = [
        'tempo_bpm', 'mode', 'density', 'pitch_low', 'pitch_high',
        'rhythm_complexity', 'velocity_mean', 'velocity_spread'
    ]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Parámetro obligatorio faltante: {key}")
    
    # Extraer parámetros
    tempo_bpm = params['tempo_bpm']
    mode = params['mode']
    density = params['density']
    pitch_low = params['pitch_low']
    pitch_high = params['pitch_high']
    rhythm_complexity = params['rhythm_complexity']
    velocity_mean = params['velocity_mean']
    velocity_spread = params['velocity_spread']
    
    # Validar rangos
    if not (20 <= tempo_bpm <= 300):
        raise ValueError(f"tempo_bpm fuera de rango: {tempo_bpm}")
    if mode not in SCALES:
        raise ValueError(f"Modo no reconocido: {mode}")
    if not (0 <= pitch_low < pitch_high <= 127):
        raise ValueError(f"Rango de pitch inválido: [{pitch_low}, {pitch_high}]")
    
    # Inicializar generador aleatorio
    rng = random.Random(seed)
    
    # Configuración MIDI
    ticks_per_beat = 480  # Resolución estándar
    beats_per_bar = 4     # Compás de 4/4
    length_ticks = length_bars * beats_per_bar * ticks_per_beat
    
    # Construir escala
    root_note = 60  # C4 como tónica fija
    scale_notes = _build_scale_notes(mode, pitch_low, pitch_high, root_note)
    
    if len(scale_notes) == 0:
        raise ValueError(f"No hay notas válidas en el rango [{pitch_low}, {pitch_high}]")
    
    # Seleccionar patrón rítmico
    rhythm_pattern = _select_rhythm_pattern(rhythm_complexity)
    
    # Generar melodía
    melody = _generate_melody(
        scale_notes=scale_notes,
        density=density,
        rhythm_pattern=rhythm_pattern,
        velocity_mean=velocity_mean,
        velocity_spread=velocity_spread,
        length_ticks=length_ticks,
        rng=rng
    )
    
    # Crear archivo MIDI
    midi_file = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi_file.tracks.append(track)
    
    # Añadir metadatos
    track.append(MetaMessage('track_name', name='Emotional Melody', time=0))
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm), time=0))
    
    # Añadir notas MIDI
    current_time = 0
    for note, velocity, duration in melody:
        if note is not None:
            # Note On
            track.append(Message('note_on', note=note, velocity=velocity, time=current_time))
            # Note Off
            track.append(Message('note_off', note=note, velocity=0, time=duration))
            current_time = 0  # El tiempo se acumula en el note_off
        else:
            # Silencio: solo acumular tiempo
            current_time += duration
    
    # Finalizar track
    track.append(MetaMessage('end_of_track', time=0))
    
    # Guardar archivo
    output_path = Path(out_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_file.save(str(output_path))
    
    return str(output_path)


def get_default_params() -> Dict[str, any]:
    """
    Retorna parámetros por defecto para pruebas.
    
    Returns:
        Dict[str, any]: Parámetros musicales neutros/equilibrados
    """
    return {
        'tempo_bpm': 120,
        'mode': 'major',
        'density': 0.7,
        'pitch_low': 60,
        'pitch_high': 72,
        'rhythm_complexity': 0.5,
        'velocity_mean': 80,
        'velocity_spread': 15
    }
