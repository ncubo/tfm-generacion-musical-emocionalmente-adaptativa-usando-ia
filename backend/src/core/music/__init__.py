"""
Módulo de traducción de emociones a parámetros musicales.

Este módulo implementa el mapeo desde el espacio Valence-Arousal
a parámetros musicales interpretables para generación MIDI.
"""

from .mapping import va_to_music_params, clamp, lerp, to_unit

__all__ = [
    'va_to_music_params',
    'clamp',
    'lerp',
    'to_unit'
]
