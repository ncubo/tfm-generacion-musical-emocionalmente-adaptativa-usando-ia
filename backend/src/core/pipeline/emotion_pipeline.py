"""
Pipeline de procesamiento emocional en tiempo real.

Este módulo implementa un pipeline integrado que conecta la captura de video,
detección emocional facial, normalización de emociones y mapeo a coordenadas
Valence-Arousal, con suavizado temporal para estabilizar las predicciones.
"""

from typing import Dict
from collections import deque


class EmotionPipeline:
    """
    Pipeline completo para procesamiento afectivo en tiempo real.
    
    Integra los siguientes componentes:
    1. Captura de video (WebcamCapture)
    2. Detección emocional (DeepFaceEmotionDetector)
    3. Mapeo a coordenadas VA (emotion_to_va)
    4. Suavizado temporal (media móvil)
    
    El pipeline permite procesamiento frame a frame con un método step()
    que devuelve la emoción detectada y sus coordenadas VA suavizadas.
    
    Attributes:
        camera: Instancia de WebcamCapture para captura de frames
        emotion_detector: Instancia de DeepFaceEmotionDetector
        va_mapper: Función de mapeo emotion_to_va
        window_size (int): Tamaño de ventana para suavizado temporal
        valence_history (deque): Historial de valores de valencia
        arousal_history (deque): Historial de valores de arousal
        current_emotion (str): Última emoción detectada
        face_detected (bool): Estado de detección facial
    
    Example:
        >>> from core.camera import WebcamCapture
        >>> from core.emotion import DeepFaceEmotionDetector
        >>> from core.va import emotion_to_va
        >>> 
        >>> camera = WebcamCapture(camera_index=0)
        >>> detector = DeepFaceEmotionDetector()
        >>> 
        >>> pipeline = EmotionPipeline(camera, detector, emotion_to_va)
        >>> pipeline.start()
        >>> 
        >>> result = pipeline.step()
        >>> print(result)
        >>> # {'emotion': 'happy', 'valence': 0.70, 'arousal': 0.60}
        >>> 
        >>> pipeline.stop()
    """
    
    def __init__(
        self,
        camera,
        emotion_detector,
        va_mapper,
        window_size: int = 5
    ):
        """
        Inicializa el pipeline de procesamiento emocional.
        
        Args:
            camera: Instancia de WebcamCapture para captura de video
            emotion_detector: Instancia de DeepFaceEmotionDetector
            va_mapper: Función que mapea emotion -> (valence, arousal)
                      Típicamente emotion_to_va del módulo core.va
            window_size (int): Tamaño de ventana para media móvil.
                              Valores típicos: 3-10. Mayor = más suavizado.
                              Default: 5
        """
        self.camera = camera
        self.emotion_detector = emotion_detector
        self.va_mapper = va_mapper
        
        # Configurar suavizado temporal
        self.window_size = max(1, window_size)
        self.valence_history: deque = deque(maxlen=self.window_size)
        self.arousal_history: deque = deque(maxlen=self.window_size)
        
        # Estado actual
        self.current_emotion = 'neutral'
        self.face_detected = False
        self._is_running = False
    
    def start(self):
        """
        Inicia el pipeline (activa la cámara).
        
        Raises:
            RuntimeError: Si la cámara no se puede iniciar
        
        Example:
            >>> pipeline.start()
        """
        if not self._is_running:
            self.camera.start()
            self._is_running = True
    
    def step(self) -> Dict[str, any]:
        """
        Ejecuta un paso del pipeline: captura → detección → mapeo → suavizado.
        
        Proceso:
        1. Captura un frame de la cámara
        2. Detecta emoción facial en el frame
        3. Mapea la emoción a coordenadas VA
        4. Aplica suavizado temporal (media móvil)
        5. Retorna resultado
        
        Si no se detecta rostro o hay error, retorna neutral (0.0, 0.0).
        
        Returns:
            Dict[str, any]: Diccionario con las claves:
                - 'emotion' (str): Emoción normalizada detectada
                - 'valence' (float): Coordenada V suavizada en [-1, 1]
                - 'arousal' (float): Coordenada A suavizada en [-1, 1]
                - 'face_detected' (bool): True si se detectó rostro
                - 'probabilities' (dict): Probabilidades por emoción (si hay rostro)
        
        Example:
            >>> result = pipeline.step()
            >>> print(result['emotion'])
            'happy'
            >>> print(f"V: {result['valence']:.2f}, A: {result['arousal']:.2f}")
            V: 0.68, A: 0.58
        """
        # Capturar frame
        success, frame = self.camera.read()
        
        if not success:
            # Si falla la captura, retornar neutral
            return self._get_neutral_result()
        
        # Detectar emoción en el frame
        try:
            emotion_result = self.emotion_detector.predict(frame)
            self.current_emotion = emotion_result['emotion']
            self.face_detected = emotion_result['face_detected']
            probabilities = emotion_result['probabilities']
            
        except Exception as e:
            # En caso de error, retornar neutral
            print(f"⚠ Error en detección: {e}")
            return self._get_neutral_result()
        
        # Mapear emoción a coordenadas VA
        valence, arousal = self.va_mapper(self.current_emotion)
        
        # Aplicar suavizado temporal
        self.valence_history.append(valence)
        self.arousal_history.append(arousal)
        
        smoothed_valence = self._smooth_values(self.valence_history)
        smoothed_arousal = self._smooth_values(self.arousal_history)
        
        # Construir resultado
        return {
            'emotion': self.current_emotion,
            'valence': smoothed_valence,
            'arousal': smoothed_arousal,
            'face_detected': self.face_detected,
            'probabilities': probabilities
        }
    
    def stop(self):
        """
        Detiene el pipeline (libera la cámara).
        
        Example:
            >>> pipeline.stop()
        """
        if self._is_running:
            self.camera.release()
            self._is_running = False
    
    def reset_smoothing(self):
        """
        Limpia el historial de suavizado temporal.
        
        Útil cuando se quiere reiniciar el suavizado sin recrear el pipeline,
        por ejemplo al cambiar de sesión o después de una pausa larga.
        
        Example:
            >>> pipeline.reset_smoothing()
        """
        self.valence_history.clear()
        self.arousal_history.clear()
    
    def get_current_state(self) -> Dict[str, any]:
        """
        Obtiene el estado actual sin procesar nuevo frame.
        
        Returns:
            Dict[str, any]: Estado actual con emoción y coordenadas VA
        
        Example:
            >>> state = pipeline.get_current_state()
            >>> print(state['emotion'])
            'happy'
        """
        if len(self.valence_history) == 0:
            return self._get_neutral_result()
        
        smoothed_valence = self._smooth_values(self.valence_history)
        smoothed_arousal = self._smooth_values(self.arousal_history)
        
        return {
            'emotion': self.current_emotion,
            'valence': smoothed_valence,
            'arousal': smoothed_arousal,
            'face_detected': self.face_detected,
            'probabilities': {}
        }
    
    def is_running(self) -> bool:
        """
        Verifica si el pipeline está activo.
        
        Returns:
            bool: True si el pipeline está ejecutándose
        
        Example:
            >>> pipeline.is_running()
            True
        """
        return self._is_running
    
    def _smooth_values(self, history: deque) -> float:
        """
        Calcula la media móvil de un historial de valores.
        
        Args:
            history (deque): Historial de valores
        
        Returns:
            float: Promedio de los valores
        """
        if len(history) == 0:
            return 0.0
        return sum(history) / len(history)
    
    def _get_neutral_result(self) -> Dict[str, any]:
        """
        Retorna un resultado neutral por defecto.
        
        Returns:
            Dict[str, any]: Resultado con emoción neutral y coordenadas (0, 0)
        """
        return {
            'emotion': 'neutral',
            'valence': 0.0,
            'arousal': 0.0,
            'face_detected': False,
            'probabilities': {}
        }
