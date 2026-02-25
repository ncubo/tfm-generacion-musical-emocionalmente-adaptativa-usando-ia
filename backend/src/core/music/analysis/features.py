"""
Extracción de características musicales de archivos MIDI.

Este módulo proporciona funciones para extraer features objetivas y cuantificables
de archivos MIDI generados, útiles para comparar diferentes motores de generación.

Se usa la librería mido para mantener el código ligero y evitar dependencias pesadas.
"""

import mido
from pathlib import Path
from typing import Dict, List, Optional


def extract_midi_features(midi_path: str) -> Dict[str, float]:
    """
    Extrae características musicales objetivas de un archivo MIDI.
    
    Analiza el archivo MIDI y calcula features cuantificables que permiten
    comparar diferentes generadores de música de forma objetiva.
    
    Args:
        midi_path: Ruta al archivo MIDI a analizar
    
    Returns:
        Diccionario con las siguientes 4 métricas:
        
        Métricas musicales (proxy de arousal):
            - note_density: Notas por segundo (float)
            - pitch_range: Rango tonal en semitonos (int)
            - mean_velocity: Intensidad dinámica MIDI 0-127 (float)
        
        Métricas de rendimiento (validación):
            - total_duration_seconds: Duración total en segundos (float)
        
        Nota: generation_time_ms se mide en los scripts de benchmark,
        no se calcula en esta función porque depende del contexto de ejecución.
    
    Raises:
        FileNotFoundError: Si el archivo MIDI no existe
        ValueError: Si el archivo MIDI es inválido o está vacío
    
    Example:
        >>> features = extract_midi_features('output/happy.mid')
        >>> print(f"Densidad: {features['note_density']:.2f} notas/seg")
        >>> print(f"Rango tonal: {features['pitch_range']} semitonos")
    """
    # Validar que el archivo existe
    midi_file = Path(midi_path)
    if not midi_file.exists():
        raise FileNotFoundError(f"Archivo MIDI no encontrado: {midi_path}")
    
    # Cargar archivo MIDI
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise ValueError(f"Error al cargar archivo MIDI: {e}")
    
    # Extraer todas las notas (note_on con velocity > 0)
    notes = []  # Lista de tuplas: (pitch, velocity, start_time, duration)
    tempo = 500000  # Tempo por defecto (120 BPM)
    ticks_per_beat = mid.ticks_per_beat
    
    # Procesar todos los tracks
    for track in mid.tracks:
        current_time = 0  # Tiempo actual en ticks
        active_notes = {}  # Dict[pitch] -> (start_time, velocity)
        
        for msg in track:
            current_time += msg.time
            
            # Capturar cambios de tempo
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            
            # Note on con velocity > 0 inicia una nota
            elif msg.type == 'note_on' and msg.velocity > 0:
                if msg.note in active_notes:
                    # Nota ya activa, cerrar la anterior primero
                    start_time, start_vel = active_notes[msg.note]
                    duration = current_time - start_time
                    notes.append((msg.note, start_vel, start_time, duration))
                
                # Registrar nueva nota activa
                active_notes[msg.note] = (current_time, msg.velocity)
            
            # Note off o note_on con velocity 0 termina una nota
            elif (msg.type == 'note_off' or 
                  (msg.type == 'note_on' and msg.velocity == 0)):
                if msg.note in active_notes:
                    start_time, start_vel = active_notes[msg.note]
                    duration = current_time - start_time
                    notes.append((msg.note, start_vel, start_time, duration))
                    del active_notes[msg.note]
        
        # Cerrar cualquier nota que siga activa al final del track
        for pitch, (start_time, velocity) in active_notes.items():
            duration = current_time - start_time
            notes.append((pitch, velocity, start_time, duration))
    
    # Validar que hay notas
    if not notes:
        raise ValueError(f"El archivo MIDI no contiene notas: {midi_path}")
    
    # Calcular duración total en segundos
    # Convertir ticks a segundos: segundos = (ticks / ticks_per_beat) * (tempo / 1_000_000)
    total_ticks = max(note[2] + note[3] for note in notes)  # max(start + duration)
    total_duration_seconds = (total_ticks / ticks_per_beat) * (tempo / 1_000_000)
    
    # Extraer features de las notas
    pitches = [note[0] for note in notes]
    velocities = [note[1] for note in notes]
    durations_ticks = [note[3] for note in notes]
    
    # Convertir duraciones de ticks a segundos
    durations_seconds = [
        (dur / ticks_per_beat) * (tempo / 1_000_000) 
        for dur in durations_ticks
    ]
    
    # Calcular métricas principales
    pitch_range = max(pitches) - min(pitches)
    mean_velocity = sum(velocities) / len(velocities)
    
    # Note density: notas por segundo
    note_density = len(notes) / total_duration_seconds if total_duration_seconds > 0 else 0
    
    # Retornar 4 métricas (3 musicales + 1 de rendimiento)
    # generation_time_ms se mide en los scripts, no aquí
    return {
        # Métricas musicales (proxies de arousal)
        'note_density': round(note_density, 3),
        'pitch_range': pitch_range,
        'mean_velocity': round(mean_velocity, 2),
        
        # Métrica de rendimiento (validación)
        'total_duration_seconds': round(total_duration_seconds, 2)
    }


def extract_features_batch(midi_paths: List[str]) -> List[Dict[str, any]]:
    """
    Extrae features de múltiples archivos MIDI.
    
    Args:
        midi_paths: Lista de rutas a archivos MIDI
    
    Returns:
        Lista de diccionarios con features, uno por archivo.
        Si un archivo falla, se incluye un dict con 'error' en lugar de features.
    
    Example:
        >>> paths = ['output/happy.mid', 'output/sad.mid']
        >>> results = extract_features_batch(paths)
        >>> for path, features in zip(paths, results):
        ...     if 'error' not in features:
        ...         print(f"{path}: {features['note_density']:.2f} notas/s")
    """
    results = []
    
    for path in midi_paths:
        try:
            features = extract_midi_features(path)
            features['midi_path'] = path
            results.append(features)
        except Exception as e:
            results.append({
                'midi_path': path,
                'error': str(e)
            })
    
    return results
