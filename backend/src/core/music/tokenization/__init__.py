"""
Módulo de tokenización para archivos MIDI.

Proporciona utilidades para convertir archivos MIDI a tokens usando diferentes esquemas
de tokenización (REMI, etc.) compatibles con modelos de lenguaje.
"""

from .remi import midi_to_remi_tokens, load_remi_tokenizer

__all__ = [
    'midi_to_remi_tokens',
    'load_remi_tokenizer',
]
