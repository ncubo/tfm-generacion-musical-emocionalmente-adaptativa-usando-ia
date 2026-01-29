"""
Módulo de mapeo Valence-Arousal (VA).

Este módulo implementa la conversión de emociones discretas a coordenadas
continuas en el espacio bidimensional Valence-Arousal basado en el modelo
Circumplex de Russell (1980).

Componentes:
    - va_table: Tabla de mapeo de emociones a coordenadas VA
    - mapper: Funciones de conversión y utilidades
"""

from .va_table import VA_TABLE
from .mapper import emotion_to_va, clamp_va, EmotionVAMapper

__all__ = [
    'VA_TABLE',
    'emotion_to_va',
    'clamp_va',
    'EmotionVAMapper'
]
