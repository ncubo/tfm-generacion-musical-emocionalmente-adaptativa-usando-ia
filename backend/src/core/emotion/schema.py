"""
Módulo de normalización de emociones.

Este módulo define el esquema de emociones estándar utilizado en el sistema
y proporciona funciones para normalizar etiquetas de emociones provenientes
de diferentes detectores (como DeepFace) a un conjunto fijo y controlado.

El conjunto de emociones estándar está basado en las emociones básicas de Ekman
más el estado neutral, asegurando consistencia en todo el sistema.
"""

from typing import Dict, List

# Conjunto fijo de emociones estándar del sistema
STANDARD_EMOTIONS: List[str] = [
    "happy",      # Felicidad, alegría
    "sad",        # Tristeza
    "angry",      # Enfado, ira
    "fear",       # Miedo
    "surprise",   # Sorpresa
    "disgust",    # Disgusto, asco
    "neutral"     # Estado neutral, sin emoción dominante
]

# Mapeo de etiquetas de DeepFace a emociones estándar
# Este diccionario permite convertir las salidas de DeepFace al formato interno
DEEPFACE_TO_STANDARD: Dict[str, str] = {
    # Mapeo directo (DeepFace usa las mismas etiquetas que nuestro estándar)
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "neutral": "neutral",
    
    # Variaciones o sinónimos posibles (por si DeepFace cambia en el futuro)
    "happiness": "happy",
    "sadness": "sad",
    "anger": "angry",
    "scared": "fear",
    "surprised": "surprise",
    "disgusted": "disgust",
}


def normalize_emotion(emotion: str) -> str:
    """
    Normaliza una etiqueta de emoción a una del conjunto estándar.
    
    Esta función convierte etiquetas de emoción provenientes de detectores
    externos (como DeepFace) al conjunto fijo de emociones estándar del sistema.
    Si la emoción no se reconoce, devuelve "neutral" como valor por defecto.
    
    Args:
        emotion (str): Etiqueta de emoción a normalizar (puede venir de DeepFace)
    
    Returns:
        str: Emoción normalizada del conjunto STANDARD_EMOTIONS
    
    Examples:
        >>> normalize_emotion("happy")
        'happy'
        
        >>> normalize_emotion("anger")
        'angry'
        
        >>> normalize_emotion("unknown_emotion")
        'neutral'
        
        >>> normalize_emotion("")
        'neutral'
    """
    if not emotion:
        return "neutral"
    
    # Convertir a minúsculas para búsqueda case-insensitive
    emotion_lower = emotion.lower().strip()
    
    # Buscar en el diccionario de mapeo
    normalized = DEEPFACE_TO_STANDARD.get(emotion_lower)
    
    if normalized:
        return normalized
    
    # Si no se encuentra en el mapeo, intentar búsqueda directa en estándar
    if emotion_lower in STANDARD_EMOTIONS:
        return emotion_lower
    
    # Valor por defecto si no se reconoce la emoción
    return "neutral"


def is_valid_emotion(emotion: str) -> bool:
    """
    Verifica si una emoción pertenece al conjunto estándar.
    
    Args:
        emotion (str): Etiqueta de emoción a verificar
    
    Returns:
        bool: True si la emoción es válida, False en caso contrario
    
    Examples:
        >>> is_valid_emotion("happy")
        True
        
        >>> is_valid_emotion("angry")
        True
        
        >>> is_valid_emotion("confused")
        False
    """
    return emotion in STANDARD_EMOTIONS


def get_all_emotions() -> List[str]:
    """
    Obtiene la lista completa de emociones estándar del sistema.
    
    Returns:
        List[str]: Lista de emociones estándar
    
    Example:
        >>> emotions = get_all_emotions()
        >>> print(emotions)
        ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
    """
    return STANDARD_EMOTIONS.copy()
