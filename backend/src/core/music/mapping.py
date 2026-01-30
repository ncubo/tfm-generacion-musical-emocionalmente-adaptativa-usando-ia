"""
Mapeo de coordenadas Valence-Arousal a parámetros musicales.

Este módulo implementa funciones matemáticas explícitas para convertir
coordenadas emocionales en el espacio VA a parámetros musicales controlables
que pueden usarse para condicionar generación MIDI.

La intuición del mapeo:
    - Arousal (activación) controla la energía musical: tempo, densidad,
      complejidad rítmica y velocidad de las notas
    - Valence (valencia) controla la cualidad afectiva: modo (major/minor)
      y registro tonal (grave/agudo)

Referencias:
    - Juslin, P. N., & Laukka, P. (2004). Expression, perception, and
      induction of musical emotions: A review and a questionnaire study of
      everyday listening. Journal of New Music Research, 33(3), 217-238.
    - Eerola, T., & Vuoskoski, J. K. (2011). A comparison of the discrete
      and dimensional models of emotion in music. Psychology of Music, 39(1).
"""

from typing import Dict
from ..utils import clamp, lerp, to_unit


def va_to_music_params(v: float, a: float) -> Dict[str, any]:
    """
    Convierte coordenadas Valence-Arousal a parámetros musicales.
    
    Este es el mapeo central del sistema que traduce estados emocionales
    continuos a parámetros musicales discretos y continuos que pueden
    usarse para condicionar la generación de música MIDI.
    
    Intuición del mapeo:
        - Arousal controla la energía musical (tempo, densidad, ritmo, velocidad)
        - Valence controla la cualidad afectiva (modo y registro tonal)
        - Alta activacion -> musica mas rapida, densa, compleja y fuerte
        - Valencia positiva -> modos mayores y registros mas agudos
    
    Args:
        v (float): Valence (valencia) en rango [-1, 1]
                  -1 = muy negativo, 0 = neutral, +1 = muy positivo
        a (float): Arousal (activación) en rango [-1, 1]
                  -1 = muy calmado, 0 = neutral, +1 = muy activado
    
    Returns:
        Dict[str, any]: Diccionario con los parámetros musicales:
            - tempo_bpm (int): Tempo en beats por minuto [60, 180]
            - mode (str): Modo musical "major" o "minor"
            - density (float): Densidad de notas [0.2, 1.0]
            - pitch_low (int): Nota MIDI más grave del rango [48, 60]
            - pitch_high (int): Nota MIDI más aguda del rango [72, 84]
            - rhythm_complexity (float): Complejidad rítmica [0.0, 1.0]
            - velocity_mean (int): Velocidad media de notas [40, 120]
            - velocity_spread (int): Variación de velocidad [0, 30]
    
    Examples:
        >>> params = va_to_music_params(0.7, 0.6)  # Happy
        >>> params['mode']
        'major'
        >>> params['tempo_bpm']
        132
        
        >>> params = va_to_music_params(-0.7, -0.4)  # Sad
        >>> params['mode']
        'minor'
        >>> params['tempo_bpm']
        78
        
        >>> params = va_to_music_params(0.0, 0.0)  # Neutral
        >>> params['mode']
        'major'
        >>> params['tempo_bpm']
        120
    """
    # Normalizar V y A al rango [0, 1] para facilitar interpolaciones
    t_v = to_unit(v)
    t_a = to_unit(a)
    
    # ===== TEMPO =====
    # El arousal controla directamente la velocidad del tempo
    # Bajo arousal (calma) -> tempos lentos (~60 BPM)
    # Alto arousal (energia) -> tempos rapidos (~180 BPM)
    tempo_bpm = int(lerp(60, 180, t_a))
    
    # ===== MODO =====
    # La valencia determina el modo musical
    # Valence positiva -> modo mayor (sonido alegre/brillante)
    # Valence negativa -> modo menor (sonido triste/oscuro)
    mode = "major" if v >= 0 else "minor"
    
    # ===== DENSIDAD =====
    # El arousal controla la densidad de notas
    # Bajo arousal -> pocas notas (musica espaciada)
    # Alto arousal -> muchas notas (musica densa)
    density = lerp(0.2, 1.0, t_a)
    
    # ===== RANGO TONAL =====
    # Registro (centro tonal): sube con valencia
    # Valencia baja -> registro grave (C4 = MIDI 60)
    # Valencia alta -> registro agudo (C5 = MIDI 72)
    pitch_center = lerp(60, 72, t_v)
    
    # Amplitud (span): sube con arousal
    # Bajo arousal -> rango estrecho (1 octava = 12 semitonos)
    # Alto arousal -> rango amplio (2 octavas = 24 semitonos)
    span = lerp(12, 24, t_a)
    
    # Calcular límites del rango tonal
    pitch_low = int(clamp(pitch_center - span / 2, 48, 60))
    pitch_high = int(clamp(pitch_center + span / 2, 72, 84))
    
    # Asegurar mínimo de 1 octava de rango
    if pitch_high - pitch_low < 12:
        # Expandir simétricamente manteniendo el centro
        mid = (pitch_low + pitch_high) / 2
        pitch_low = int(clamp(mid - 6, 48, 60))
        pitch_high = int(clamp(mid + 6, 72, 84))
        
        # Si aún no cumple, forzar mínimo
        if pitch_high - pitch_low < 12:
            pitch_high = pitch_low + 12
    
    # ===== COMPLEJIDAD RÍTMICA =====
    # El arousal controla la complejidad del ritmo
    # Bajo arousal -> ritmos simples (negras, blancas)
    # Alto arousal -> ritmos complejos (corcheas, tresillos, sincopas)
    rhythm_complexity = lerp(0.0, 1.0, t_a)
    
    # ===== VELOCITY (Intensidad de las notas) =====
    # La velocity aumenta principalmente con arousal, con un bonus de valencia
    # Arousal alto -> notas mas fuertes (mas impacto)
    # Valencia positiva -> ligero incremento en brillo/intensidad
    vel = 40 + 60 * t_a + 20 * t_v
    velocity_mean = int(clamp(vel, 40, 120))
    
    # La variacion de velocity tambien sube con arousal
    # Arousal alto -> mayor variacion dinamica
    velocity_spread = int(clamp(lerp(0, 30, t_a), 0, 30))
    
    # Construir diccionario de parámetros
    return {
        'tempo_bpm': tempo_bpm,
        'mode': mode,
        'density': density,
        'pitch_low': pitch_low,
        'pitch_high': pitch_high,
        'rhythm_complexity': rhythm_complexity,
        'velocity_mean': velocity_mean,
        'velocity_spread': velocity_spread
    }
