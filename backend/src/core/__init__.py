"""
Core - Módulo principal del sistema de reconocimiento emocional y mapeo musical.

Este paquete contiene todos los componentes fundamentales del sistema:
- camera: Captura de video desde webcam
- emotion: Detección y normalización emocional
- va: Mapeo de emociones a espacio Valencia-Activación
- music: Conversión de coordenadas VA a parámetros musicales y generación MIDI
- pipeline: Orquestación del flujo completo
- utils: Utilidades matemáticas comunes
"""

from . import camera
from . import emotion
from . import va
from . import music
from . import pipeline
from . import utils

# Exponer componentes principales para facilitar imports
from .camera import WebcamCapture
from .emotion import DeepFaceEmotionDetector, normalize_emotion
from .va import emotion_to_va
from .music import va_to_music_params, generate_midi_baseline
from .pipeline import EmotionPipeline

__all__ = [
    'camera',
    'emotion',
    'va',
    'music',
    'pipeline',
    'utils',
    'WebcamCapture',
    'DeepFaceEmotionDetector',
    'normalize_emotion',
    'emotion_to_va',
    'va_to_music_params',
    'generate_midi_baseline',
    'EmotionPipeline',
]
