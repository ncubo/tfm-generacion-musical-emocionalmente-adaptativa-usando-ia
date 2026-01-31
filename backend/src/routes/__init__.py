"""
Módulo de rutas de la API Flask.

Este paquete contiene los blueprints que definen los endpoints
de la API REST para el sistema de generación musical emocional.
"""

from .health import health_bp
from .emotion import emotion_bp
from .music import music_bp

__all__ = ['health_bp', 'emotion_bp', 'music_bp']
