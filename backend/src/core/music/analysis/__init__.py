"""
Módulo de análisis de características musicales.

Proporciona herramientas para extraer features objetivas de archivos MIDI
para análisis y benchmark de modelos generativos.
"""

from .features import extract_midi_features

__all__ = [
    'extract_midi_features',
]
