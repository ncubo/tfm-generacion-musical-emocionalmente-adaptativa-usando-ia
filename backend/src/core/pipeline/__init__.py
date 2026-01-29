"""
Módulo de pipeline para procesamiento afectivo en tiempo real.

Este módulo integra los componentes del sistema para crear un flujo
completo desde captura de video hasta coordenadas emocionales VA.
"""

from .emotion_pipeline import EmotionPipeline

__all__ = ['EmotionPipeline']
