"""
Tabla de mapeo de emociones discretas a coordenadas Valencia-Activación.

Este módulo define la tabla de correspondencia entre emociones básicas
y sus coordenadas en el espacio bidimensional Valencia-Activación según el
modelo Circumplex de Russell (1980).

Referencias:
    - Russell, J. A. (1980). A circumplex model of affect. Journal of 
      Personality and Social Psychology, 39(6), 1161–1178.
    - Posner, J., Russell, J. A., & Peterson, B. S. (2005). The circumplex 
      model of affect: An integrative approach to affective neuroscience, 
      cognitive development, and psychopathology.

Coordenadas:
    - Valence (V): Dimensión de placer/displacer. Rango: [-1, 1]
        * Valores positivos: emociones placenteras (ej. felicidad)
        * Valores negativos: emociones displacenteras (ej. tristeza)
        * Valor cero: estado neutral
    
    - Arousal (A): Dimensión de activación/energía. Rango: [-1, 1]
        * Valores positivos: alta activación (ej. sorpresa, miedo)
        * Valores negativos: baja activación (ej. tristeza, calma)
        * Valor cero: activación neutral

Nota:
    Los valores definidos son aproximaciones iniciales basadas en la 
    literatura y pueden ser ajustados mediante calibración experimental
    o aprendizaje del modelo en fases posteriores del proyecto.
"""

from typing import Dict, Tuple

# Tipo para coordenadas VA
VACoordinates = Tuple[float, float]

# Tabla de mapeo de emociones a coordenadas (Valence, Arousal)
# Valores iniciales aproximados basados en el modelo Circumplex
VA_TABLE: Dict[str, VACoordinates] = {
    # HAPPY (Felicidad)
    # Alta valencia positiva (placentero), arousal medio-alto (activación moderada)
    # Ubicación: Cuadrante superior derecho del circumplex
    "happy": (0.70, 0.60),
    
    # SAD (Tristeza)
    # Alta valencia negativa (displacentero), arousal negativo (baja energía)
    # Ubicación: Cuadrante inferior izquierdo del circumplex
    "sad": (-0.70, -0.40),
    
    # ANGRY (Enfado/Ira)
    # Valencia negativa (displacentero), arousal muy alto (alta activación)
    # Ubicación: Cuadrante superior izquierdo del circumplex
    "angry": (-0.60, 0.70),
    
    # FEAR (Miedo)
    # Alta valencia negativa (displacentero), arousal alto (alta activación)
    # Ubicación: Cuadrante superior izquierdo del circumplex
    "fear": (-0.70, 0.60),
    
    # SURPRISE (Sorpresa)
    # Valencia ligeramente positiva o neutra, arousal muy alto (máxima activación)
    # Ubicación: Parte superior central del circumplex
    # Nota: La sorpresa puede ser positiva o negativa según el contexto
    "surprise": (0.20, 0.80),
    
    # DISGUST (Disgusto/Asco)
    # Valencia negativa (displacentero), arousal medio (activación moderada)
    # Ubicación: Cuadrante izquierdo del circumplex
    "disgust": (-0.60, 0.30),
    
    # NEUTRAL (Estado neutral)
    # Sin valencia ni arousal, punto de origen del espacio VA
    # Ubicación: Centro del circumplex (0, 0)
    "neutral": (0.00, 0.00)
}


def get_va_coordinates(emotion: str) -> VACoordinates:
    """
    Obtiene las coordenadas VA de una emoción.
    
    Args:
        emotion (str): Etiqueta de emoción normalizada
    
    Returns:
        VACoordinates: Tupla (valence, arousal) o (0.0, 0.0) si no existe
    
    Example:
        >>> get_va_coordinates("happy")
        (0.70, 0.60)
        
        >>> get_va_coordinates("unknown")
        (0.0, 0.0)
    """
    return VA_TABLE.get(emotion, (0.0, 0.0))


def get_all_emotions() -> list[str]:
    """
    Retorna la lista de todas las emociones con mapeo VA definido.
    
    Returns:
        list[str]: Lista de etiquetas de emociones
    
    Example:
        >>> emotions = get_all_emotions()
        >>> print(emotions)
        ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
    """
    return list(VA_TABLE.keys())
