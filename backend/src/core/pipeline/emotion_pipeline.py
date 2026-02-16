"""
Pipeline de procesamiento emocional en tiempo real.

Este módulo implementa un pipeline integrado que conecta la captura de video,
detección emocional facial, normalización de emociones y mapeo a coordenadas
Valencia-Activación, con suavizado temporal para estabilizar las predicciones.

Estrategias de estabilización implementadas:
1. Media Móvil Exponencial (EMA) para valores V/A - más responsiva que SMA
2. Ventana de mayoría para emoción discreta - evita cambios abruptos
3. Umbral de confianza mínimo para cambios de emoción
"""

from typing import Dict, Optional
from collections import deque, Counter
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
    5. Suavizado temporal avanzado:
       - EMA (Exponential Moving Average) para valores V/A
       - Ventana de mayoría para emoción discreta
       - Umbral de confianza para cambios de emoción
    
    El pipeline permite procesamiento frame a frame con un método step()
    que devuelve la emoción detectada y sus coordenadas VA estabilizadas.
    
    Attributes:
        camera: Instancia de WebcamCapture para captura de frames
        detector: Instancia de DeepFaceEmotionDetector
        window_size (int): Tamaño de ventana para ventana de mayoría (emoción discreta)
        alpha (float): Factor de suavizado EMA para V/A (0 < alpha <= 1)
        min_confidence (float): Confianza mínima para aceptar cambio de emoción
        emotion_buffer (deque): Buffer de emociones para ventana de mayoría
        smoothed_valence (float): Valor suavizado actual de valencia
        smoothed_arousal (float): Valor suavizado actual de arousal
        current_emotion (str): Emoción estable actual
    
    Example:
        >>> from core.camera import WebcamCapture
        >>> from core.emotion import DeepFaceEmotionDetector
        >>> 
        >>> camera = WebcamCapture(camera_index=0)
        >>> detector = DeepFaceEmotionDetector()
        >>> 
        >>> # Pipeline con estabilización moderada
        >>> pipeline = EmotionPipeline(
        ...     camera, 
        ...     detector, 
        ...     window_size=7,
        ...     alpha=0.3,
        ...     min_confidence=60.0
        ... )
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
        window_size: int = 7,
        alpha: float = 0.3,
        min_confidence: float = 60.0
    ):
        """
        Inicializa el pipeline de procesamiento emocional con estabilización temporal.
        
        Args:
            camera: Instancia de WebcamCapture para captura de video
            detector: Instancia de DeepFaceEmotionDetector
            window_size (int): Tamaño de ventana para mayoría de emoción discreta.
                              Valores típicos: 5-10. Mayor = más estable pero menos responsivo.
                              Default: 7 (equilibrio entre estabilidad y respuesta)
            alpha (float): Factor de suavizado EMA para V/A.
                          0 < alpha <= 1. Valores bajos (0.1-0.3) = más suavizado.
                          Valores altos (0.5-1.0) = más responsivo.
                          Default: 0.3 (suavizado moderado)
            min_confidence (float): Confianza mínima (%) para aceptar cambio de emoción.
                                   Evita cambios por detecciones poco confiables.
                                   Default: 60.0%
        
        Notas técnicas:
            - EMA es más responsive que SMA: da más peso a valores recientes
            - La ventana de mayoría reduce "parpadeos" en la emoción discreta
            - min_confidence añade robustez ante detecciones ruidosas
        """
        self.camera = camera
        self.detector = detector
        
        # Parámetros de estabilización
        self.window_size = max(1, window_size)
        self.alpha = max(0.01, min(1.0, alpha))  # Clamped entre 0.01 y 1.0
        self.min_confidence = max(0.0, min(100.0, min_confidence))  # Entre 0 y 100
        
        # Buffer para ventana de mayoría (emoción discreta)
        self.emotion_buffer: deque = deque(maxlen=self.window_size)
        
        # Estado suavizado de V/A (EMA)
        self.smoothed_valence: Optional[float] = None
        self.smoothed_arousal: Optional[float] = None
        
        # Emoción estable actual
        self.current_emotion: str = 'neutral'
    
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
        Ejecuta un paso del pipeline: captura -> deteccion -> normalizacion -> estabilizacion.
        
        Proceso mejorado:
        1. Captura un frame de la cámara
        2. Detecta emoción facial en el frame
        3. Normaliza la emoción usando normalize_emotion
        4. Aplica ventana de mayoría para estabilizar emoción discreta
        5. Mapea la emoción estable a coordenadas VA
        6. Aplica EMA (Exponential Moving Average) para suavizar V/A
        7. Retorna resultado estabilizado
        
        Si no se detecta rostro, hay error o falla la captura, retorna neutral (0.0, 0.0).
        Este método NUNCA lanza excepciones, siempre devuelve un resultado válido.
        
        Returns:
            Dict[str, any]: Diccionario con las claves:
                - 'emotion' (str): Emoción estabilizada (ventana de mayoría)
                - 'valence' (float): Coordenada V suavizada en [-1, 1] (EMA)
                - 'arousal' (float): Coordenada A suavizada en [-1, 1] (EMA)
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
            scores = emotion_result.get('probabilities', {})
            
            # 3. Normalizar emoción
            normalized_emotion = normalize_emotion(raw_emotion)
            
            # 4. Estabilizar emoción discreta con ventana de mayoría
            stable_emotion = self._stabilize_emotion(normalized_emotion, scores)
            
            # 5. Mapear emoción estable a coordenadas VA
            valence, arousal = emotion_to_va(stable_emotion)
            
            # 6. Aplicar EMA para suavizar V/A
            smoothed_valence, smoothed_arousal = self._apply_ema(valence, arousal)
            
            # 7. Construir y retornar resultado
            return {
                'emotion': stable_emotion,
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
    
    def _stabilize_emotion(self, emotion: str, scores: Dict[str, float]) -> str:
        """
        Estabiliza la emoción discreta usando ventana de mayoría.
        
        Esta función implementa un mecanismo de estabilización temporal que:
        1. Mantiene un buffer de las últimas N emociones detectadas
        2. Calcula la emoción más frecuente (mayoría) en la ventana
        3. Solo acepta cambios si la nueva emoción tiene confianza suficiente
        
        Esto evita "parpadeos" cuando el detector oscila entre emociones cercanas.
        
        Args:
            emotion (str): Emoción recién detectada
            scores (Dict[str, float]): Scores de confianza por emoción
        
        Returns:
            str: Emoción estabilizada (la más frecuente en la ventana)
        
        Notas técnicas:
            - La ventana se llena gradualmente al inicio
            - Si hay empate, se mantiene la emoción actual
            - El umbral de confianza evita cambios por ruido
            - En el arranque inicial (buffer vacío), acepta la primera detección
        """
        # Verificar confianza de la nueva emoción
        confidence = scores.get(emotion, 0.0) if scores else 0.0
        
        # Aplicar filtro de confianza solo si ya tenemos historial
        # En el arranque (buffer vacío), aceptamos la primera detección
        if confidence < self.min_confidence and len(self.emotion_buffer) > 0:
            # Confianza baja: mantener emoción actual si ya tenemos historial
            emotion = self.current_emotion
        
        # Añadir a buffer
        self.emotion_buffer.append(emotion)
        
        # Calcular emoción más frecuente (mayoría)
        if len(self.emotion_buffer) > 0:
            emotion_counts = Counter(self.emotion_buffer)
            # most_common(1) retorna lista de tuplas [(emocion, count)]
            stable_emotion = emotion_counts.most_common(1)[0][0]
        else:
            stable_emotion = emotion
        
        # Actualizar emoción estable actual
        self.current_emotion = stable_emotion
        
        return stable_emotion
    
    def _apply_ema(self, valence: float, arousal: float) -> tuple:
        """
        Aplica Media Móvil Exponencial (EMA) a valores V/A.
        
        EMA da más peso a valores recientes que SMA (Simple Moving Average),
        resultando en mejor balance entre suavizado y responsividad:
        
        EMA(t) = alpha * valor_nuevo + (1 - alpha) * EMA(t-1)
        
        Donde:
        - alpha cercano a 1: más peso a valor nuevo (más responsive)
        - alpha cercano a 0: más peso a histórico (más suavizado)
        
        Args:
            valence (float): Nuevo valor de valencia
            arousal (float): Nuevo valor de arousal
        
        Returns:
            tuple: (valence_suavizado, arousal_suavizado)
        
        Notas técnicas:
            - En el primer frame, inicializa con el valor actual
            - EMA converge más rápido que SMA al valor real
            - No requiere mantener buffer histórico (eficiente en memoria)
        """
        # Primera iteración: inicializar con valores actuales
        if self.smoothed_valence is None or self.smoothed_arousal is None:
            self.smoothed_valence = valence
            self.smoothed_arousal = arousal
        else:
            # Aplicar fórmula EMA
            self.smoothed_valence = (
                self.alpha * valence + (1 - self.alpha) * self.smoothed_valence
            )
            self.smoothed_arousal = (
                self.alpha * arousal + (1 - self.alpha) * self.smoothed_arousal
            )
        
        return (self.smoothed_valence, self.smoothed_arousal)
    
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

