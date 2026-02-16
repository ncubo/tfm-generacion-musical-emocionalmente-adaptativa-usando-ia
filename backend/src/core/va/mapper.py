"""
Mapper de emociones discretas a coordenadas Valencia-Activación.

Este módulo proporciona funciones y clases para convertir emociones discretas
detectadas por el sistema a coordenadas continuas en el espacio VA, con
capacidades de validación, clamping y suavizado temporal.
"""

from typing import Tuple, Optional
from collections import deque
from .va_table import get_va_coordinates
from ..utils import clamp_va

# Tipo para coordenadas VA
VACoordinates = Tuple[float, float]


def emotion_to_va(emotion: str) -> VACoordinates:
    """
    Convierte una emoción discreta a coordenadas Valencia-Activación.
    
    Esta función toma una etiqueta de emoción normalizada y retorna sus
    coordenadas (V, A) correspondientes según la tabla de mapeo definida.
    Si la emoción no es reconocida, retorna el punto neutral (0.0, 0.0).
    
    Args:
        emotion (str): Etiqueta de emoción normalizada
                      (debe ser una de: happy, sad, angry, fear, 
                       surprise, disgust, neutral)
    
    Returns:
        VACoordinates: Tupla (valence, arousal) en el rango [-1, 1]
                       Retorna (0.0, 0.0) si la emoción no es reconocida
    
    Examples:
        >>> emotion_to_va("happy")
        (0.70, 0.60)
        
        >>> emotion_to_va("sad")
        (-0.70, -0.40)
        
        >>> emotion_to_va("neutral")
        (0.0, 0.0)
        
        >>> emotion_to_va("unknown_emotion")
        (0.0, 0.0)
    """
    if not emotion or not isinstance(emotion, str):
        return (0.0, 0.0)
    
    # Obtener coordenadas de la tabla
    # Si no existe, get_va_coordinates devuelve (0.0, 0.0) por defecto
    return get_va_coordinates(emotion.lower().strip())


class EmotionVAMapper:
    """
    Mapper con suavizado temporal para convertir emociones a coordenadas VA.
    
    Esta clase proporciona una conversión suavizada de emociones a coordenadas
    VA mediante un promedio móvil de los últimos N valores. Esto es útil para
    reducir el ruido y las transiciones bruscas en detección en tiempo real.
    
    Attributes:
        window_size (int): Tamaño de la ventana del promedio móvil
        valence_history (deque): Historial de valores de valencia
        arousal_history (deque): Historial de valores de arousal
    
    Example:
        >>> mapper = EmotionVAMapper(window_size=5)
        >>> 
        >>> # Primeras detecciones
        >>> v, a = mapper.map("happy")  # (0.70, 0.60)
        >>> v, a = mapper.map("happy")  # Promedio de 2 valores
        >>> v, a = mapper.map("sad")    # Promedio de 3 valores
        >>> 
        >>> # Resetear historial
        >>> mapper.reset()
    """
    
    def __init__(self, window_size: int = 5):
        """
        Inicializa el mapper con suavizado temporal.
        
        Args:
            window_size (int): Número de valores históricos a promediar.
                              Valores típicos: 3-10 para video en tiempo real.
                              Mayor tamaño = más suavizado pero más lag.
        """
        self.window_size = max(1, window_size)  # Mínimo 1
        self.valence_history: deque = deque(maxlen=self.window_size)
        self.arousal_history: deque = deque(maxlen=self.window_size)
    
    def map(self, emotion: str) -> VACoordinates:
        """
        Convierte una emoción a coordenadas VA con suavizado temporal.
        
        Args:
            emotion (str): Etiqueta de emoción normalizada
        
        Returns:
            VACoordinates: Tupla (valence, arousal) suavizada
        
        Example:
            >>> mapper = EmotionVAMapper(window_size=3)
            >>> mapper.map("happy")
            (0.70, 0.60)
            >>> mapper.map("happy")
            (0.70, 0.60)
            >>> mapper.map("sad")
            (-0.1, 0.27)  # Promedio de happy, happy, sad
        """
        # Obtener coordenadas base de la emoción
        valence, arousal = emotion_to_va(emotion)
        
        # Agregar a historial
        self.valence_history.append(valence)
        self.arousal_history.append(arousal)
        
        # Calcular promedio móvil
        avg_valence = sum(self.valence_history) / len(self.valence_history)
        avg_arousal = sum(self.arousal_history) / len(self.arousal_history)
        
        # Asegurar rango [-1, 1] (aunque no debería ser necesario con la tabla)
        return clamp_va(avg_valence, avg_arousal)
    
    def reset(self):
        """
        Limpia el historial de valores, reiniciando el suavizado.
        
        Útil cuando se cambia de sesión o se quiere evitar que valores
        antiguos afecten las nuevas detecciones.
        
        Example:
            >>> mapper = EmotionVAMapper()
            >>> mapper.map("happy")
            >>> mapper.map("sad")
            >>> mapper.reset()  # Limpia historial
            >>> mapper.map("angry")  # Comienza historial nuevo
        """
        self.valence_history.clear()
        self.arousal_history.clear()
    
    def get_history_size(self) -> int:
        """
        Retorna el número actual de valores en el historial.
        
        Returns:
            int: Cantidad de valores almacenados (máximo window_size)
        
        Example:
            >>> mapper = EmotionVAMapper(window_size=5)
            >>> mapper.map("happy")
            >>> mapper.map("sad")
            >>> mapper.get_history_size()
            2
        """
        return len(self.valence_history)
    
    def get_current_va(self) -> Optional[VACoordinates]:
        """
        Retorna las coordenadas VA actuales sin procesar nueva emoción.
        
        Returns:
            Optional[VACoordinates]: Coordenadas VA actuales o None si no hay historial
        
        Example:
            >>> mapper = EmotionVAMapper()
            >>> mapper.get_current_va()
            None
            >>> mapper.map("happy")
            (0.70, 0.60)
            >>> mapper.get_current_va()
            (0.70, 0.60)
        """
        if len(self.valence_history) == 0:
            return None
        
        avg_valence = sum(self.valence_history) / len(self.valence_history)
        avg_arousal = sum(self.arousal_history) / len(self.arousal_history)
        
        return clamp_va(avg_valence, avg_arousal)
