"""
Módulo de detección emocional usando DeepFace.

Este módulo proporciona una clase para detectar emociones faciales en tiempo real
utilizando la librería DeepFace con modelos preentrenados.
"""

import numpy as np
from typing import Dict, Optional
from deepface import DeepFace
from .schema import normalize_emotion


class DeepFaceEmotionDetector:
    """
    Detector de emociones faciales usando DeepFace.
    
    Esta clase encapsula la funcionalidad de DeepFace para analizar frames
    y extraer la emoción dominante junto con sus probabilidades.
    
    DeepFace soporta las siguientes emociones:
    - angry (enfadado)
    - disgust (disgusto)
    - fear (miedo)
    - happy (feliz)
    - sad (triste)
    - surprise (sorpresa)
    - neutral (neutral)
    
    Attributes:
        enforce_detection (bool): Si es True, lanza excepción cuando no detecta rostro
    """
    
    def __init__(self, enforce_detection: bool = False):
        """
        Inicializa el detector de emociones.
        
        Args:
            enforce_detection (bool): Si es True, lanza excepción cuando no detecta rostro.
                                     Si es False (default), devuelve neutral sin error.
        """
        self.enforce_detection = enforce_detection
        print("[OK] DeepFaceEmotionDetector inicializado")
        print("  Nota: La primera prediccion puede tardar mas (carga de modelos)")
    
    def predict(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Predice la emoción dominante en un frame.
        
        Este método analiza el frame usando DeepFace y retorna la emoción
        detectada junto con las probabilidades de cada categoría emocional.
        
        Args:
            frame (np.ndarray): Frame de imagen en formato BGR (OpenCV)
        
        Returns:
            Dict[str, any]: Diccionario con las claves:
                - 'emotion' (str): Emoción dominante detectada
                - 'probabilities' (dict): Diccionario con probabilidades por emoción
                - 'face_detected' (bool): True si se detectó un rostro
        
        Example:
            >>> detector = DeepFaceEmotionDetector()
            >>> result = detector.predict(frame)
            >>> print(result)
            {
                'emotion': 'happy',
                'probabilities': {
                    'angry': 0.5, 'disgust': 0.1, 'fear': 0.2,
                    'happy': 89.3, 'sad': 1.2, 'surprise': 3.5, 'neutral': 5.2
                },
                'face_detected': True
            }
        """
        try:
            # Análisis de emociones con DeepFace
            # - actions=['emotion']: Solo queremos detectar emociones
            # - enforce_detection=False: No lanzar error si no hay rostro
            # - silent=True: Suprimir logs verbosos de DeepFace
            result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
                silent=True
            )
            
            # DeepFace puede devolver una lista si detecta múltiples rostros
            # Tomamos el primer rostro detectado
            if isinstance(result, list):
                result = result[0]
            
            # Verificar confianza de detección facial
            # face_confidence es 0.0 cuando no hay rostro real
            # Usamos un umbral de 0.9 (90%) para considerar un rostro válido
            face_confidence = result.get('face_confidence', 0.0)
            
            if face_confidence < 0.9:
                # Confianza muy baja = no hay rostro real
                return {
                    'emotion': 'neutral',
                    'probabilities': {},
                    'face_detected': False
                }
            
            # Extraer emoción dominante y probabilidades
            dominant_emotion = result['dominant_emotion']
            emotion_probabilities = result['emotion']
            
            # Normalizar la emoción al conjunto estándar del sistema
            normalized_emotion = normalize_emotion(dominant_emotion)
            
            return {
                'emotion': normalized_emotion,
                'probabilities': emotion_probabilities,
                'face_detected': True
            }
        
        except ValueError as e:
            # ValueError se lanza cuando no se detecta un rostro
            # Esto puede ocurrir si enforce_detection=True o si hay problemas con el frame
            return {
                'emotion': 'neutral',
                'probabilities': {},
                'face_detected': False
            }
        
        except Exception as e:
            # Cualquier otro error (formato de imagen incorrecto, etc.)
            print(f"[WARN] Error en deteccion emocional: {str(e)}")
            return {
                'emotion': 'neutral',
                'probabilities': {},
                'face_detected': False
            }
    
    def get_emotion_label_spanish(self, emotion: str) -> str:
        """
        Convierte una etiqueta de emoción del inglés al español.
        
        Args:
            emotion (str): Emoción en inglés
        
        Returns:
            str: Emoción traducida al español
        """
        translations = {
            'angry': 'Enfadado',
            'disgust': 'Disgusto',
            'fear': 'Miedo',
            'happy': 'Feliz',
            'sad': 'Triste',
            'surprise': 'Sorpresa',
            'neutral': 'Neutral'
        }
        return translations.get(emotion, emotion.capitalize())
