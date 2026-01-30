"""
Pipeline de procesamiento emocional en tiempo real.

Este módulo implementa un pipeline integrado que conecta la captura de video,
detección emocional facial, normalización de emociones y mapeo a coordenadas
Valence-Arousal, con suavizado temporal para estabilizar las predicciones.
"""

from typing import Dict
from collections import deque
from ..emotion.schema import normalize_emotion
from ..va.mapper import emotion_to_va


class EmotionPipeline:
    """
    Pipeline completo para procesamiento afectivo en tiempo real.
    
    Integra los siguientes componentes:
    1. Captura de video (WebcamCapture)
    2. Detección emocional (DeepFaceEmotionDetector)
    3. Normalización de emoción (normalize_emotion)
    4. Mapeo a coordenadas VA (emotion_to_va)
    5. Suavizado temporal (media móvil simple)
    
    El pipeline permite procesamiento frame a frame con un método step()
    que devuelve la emoción detectada y sus coordenadas VA suavizadas.
    
    Attributes:
        camera: Instancia de WebcamCapture para captura de frames
        detector: Instancia de DeepFaceEmotionDetector
        window_size (int): Tamaño de ventana para suavizado temporal
        valence_buffer (deque): Buffer de valores de valencia
        arousal_buffer (deque): Buffer de valores de arousal
    
    Example:
        >>> from core.camera import WebcamCapture
        >>> from core.emotion import DeepFaceEmotionDetector
        >>> 
        >>> camera = WebcamCapture(camera_index=0)
        >>> detector = DeepFaceEmotionDetector()
        >>> 
        >>> pipeline = EmotionPipeline(camera, detector, window_size=5)
        >>> pipeline.start()
        >>> 
        >>> result = pipeline.step()
        >>> print(result)
        >>> # {'emotion': 'happy', 'valence': 0.70, 'arousal': 0.60, 'scores': {...}}
        >>> 
        >>> pipeline.stop()
    """
    
    def __init__(
        self,
        camera,
        detector,
        window_size: int = 5
    ):
        """
        Inicializa el pipeline de procesamiento emocional.
        
        Args:
            camera: Instancia de WebcamCapture para captura de video
            detector: Instancia de DeepFaceEmotionDetector
            window_size (int): Tamaño de ventana para media móvil.
                              Si es <= 1, no se aplica suavizado.
                              Valores típicos: 3-10. Mayor = más suavizado.
                              Default: 5
        """
        self.camera = camera
        self.detector = detector
        
        # Configurar suavizado temporal
        self.window_size = max(1, window_size)
        
        # Solo crear buffers si window_size > 1
        if self.window_size > 1:
            self.valence_buffer: deque = deque(maxlen=self.window_size)
            self.arousal_buffer: deque = deque(maxlen=self.window_size)
        else:
            self.valence_buffer = None
            self.arousal_buffer = None
    
    def start(self):
        """
        Inicia el pipeline (activa la cámara).
        
        Raises:
            RuntimeError: Si la cámara no se puede iniciar
        
        Example:
            >>> pipeline.start()
        """
        self.camera.start()
    
    def step(self) -> Dict[str, any]:
        """
        Ejecuta un paso del pipeline: captura → detección → normalización → mapeo → suavizado.
        
        Proceso:
        1. Captura un frame de la cámara
        2. Detecta emoción facial en el frame
        3. Normaliza la emoción usando normalize_emotion
        4. Mapea la emoción a coordenadas VA usando emotion_to_va
        5. Aplica suavizado temporal (media móvil) si window_size > 1
        6. Retorna resultado
        
        Si no se detecta rostro, hay error o falla la captura, retorna neutral (0.0, 0.0).
        Este método NUNCA lanza excepciones, siempre devuelve un resultado válido.
        
        Returns:
            Dict[str, any]: Diccionario con las claves:
                - 'emotion' (str): Emoción normalizada detectada
                - 'valence' (float): Coordenada V suavizada en [-1, 1]
                - 'arousal' (float): Coordenada A suavizada en [-1, 1]
                - 'scores' (dict): Scores/probabilidades por emoción (vacío si no hay rostro)
        
        Example:
            >>> result = pipeline.step()
            >>> print(result['emotion'])
            'happy'
            >>> print(f"V: {result['valence']:.2f}, A: {result['arousal']:.2f}")
            V: 0.68, A: 0.58
        """
        try:
            # 1. Capturar frame
            success, frame = self.camera.read()
            
            if not success or frame is None:
                # Si falla la captura, retornar neutral
                return self._get_neutral_result()
            
            # 2. Detectar emoción en el frame
            emotion_result = self.detector.predict(frame)
            raw_emotion = emotion_result.get('emotion', 'neutral')
            # El detector devuelve 'probabilities', lo mapeamos a 'scores'
            scores = emotion_result.get('probabilities', {})
            
            # 3. Normalizar emoción
            normalized_emotion = normalize_emotion(raw_emotion)
            
            # 4. Mapear emoción a coordenadas VA
            valence, arousal = emotion_to_va(normalized_emotion)
            
            # 5. Aplicar suavizado temporal SOLO si window_size > 1
            if self.window_size > 1 and self.valence_buffer is not None:
                self.valence_buffer.append(valence)
                self.arousal_buffer.append(arousal)
                
                smoothed_valence = sum(self.valence_buffer) / len(self.valence_buffer)
                smoothed_arousal = sum(self.arousal_buffer) / len(self.arousal_buffer)
            else:
                # Sin suavizado
                smoothed_valence = valence
                smoothed_arousal = arousal
            
            # 6. Construir y retornar resultado
            return {
                'emotion': normalized_emotion,
                'valence': smoothed_valence,
                'arousal': smoothed_arousal,
                'scores': scores
            }
            
        except Exception as e:
            # En caso de cualquier error, no crashear: retornar neutral
            # Nota: En producción se podría loggear el error
            return self._get_neutral_result()
    
    def stop(self):
        """
        Detiene el pipeline y libera recursos (libera la cámara).
        
        Example:
            >>> pipeline.stop()
        """
        self.camera.release()
    
    def _get_neutral_result(self) -> Dict[str, any]:
        """
        Retorna un resultado neutral por defecto.
        
        Usado cuando:
        - No se puede capturar un frame
        - No se detecta rostro
        - Ocurre cualquier error durante el procesamiento
        
        Returns:
            Dict[str, any]: Resultado con emoción neutral y coordenadas (0, 0)
        """
        return {
            'emotion': 'neutral',
            'valence': 0.0,
            'arousal': 0.0,
            'scores': {}
        }

