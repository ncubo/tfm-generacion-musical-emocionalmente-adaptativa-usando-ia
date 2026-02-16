"""
Módulo de mapeo Valencia-Activación (VA).

Este módulo implementa la conversión de emociones discretas a coordenadas
continuas en el espacio bidimensional Valencia-Activación basado en el modelo
Circumplex de Russell (1980).

Componentes:
    - va_table: Tabla de mapeo de emociones a coordenadas VA
    - mapper: Funciones de conversión y utilidades
"""

from .va_table import VA_TABLE, get_va_coordinates, get_all_emotions
from .mapper import emotion_to_va, EmotionVAMapper

__all__ = [
    'VA_TABLE',
    'get_va_coordinates',
    'get_all_emotions',
    'emotion_to_va',
    'EmotionVAMapper'
]
