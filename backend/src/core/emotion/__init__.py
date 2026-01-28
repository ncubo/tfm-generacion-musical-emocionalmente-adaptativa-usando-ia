"""
Módulo de reconocimiento emocional facial.

Este paquete contiene las implementaciones de detectores de emociones
utilizando diferentes frameworks de deep learning.
"""

from .deepface_detector import DeepFaceEmotionDetector

__all__ = ['DeepFaceEmotionDetector']
