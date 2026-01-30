"""
Módulo de reconocimiento emocional facial.

Este paquete contiene las implementaciones de detectores de emociones
utilizando diferentes frameworks de deep learning.
"""

from .deepface_detector import DeepFaceEmotionDetector
from .schema import normalize_emotion, is_valid_emotion, get_all_emotions, STANDARD_EMOTIONS

__all__ = [
    'DeepFaceEmotionDetector',
    'normalize_emotion',
    'is_valid_emotion',
    'get_all_emotions',
    'STANDARD_EMOTIONS'
]
