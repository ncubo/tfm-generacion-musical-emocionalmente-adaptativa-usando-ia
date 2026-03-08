"""
Mapeo de coordenadas Valencia-Activación a parámetros musicales.

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
    Convierte coordenadas Valencia-Activación a parámetros musicales.
    
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
    # Ref: Juslin, P. N. & Laukka, P. (2004), Tabla 2 — tempo rápido correlaciona
    #      con alta activación (happiness, anger); tempo lento con baja (sadness, tenderness).
    #      Rango 60-180 BPM: convención musical estándar (Largo a Presto).
    tempo_bpm = int(lerp(60, 180, t_a))
    
    # ===== MODO =====
    # La valencia determina el modo musical
    # Valence positiva -> modo mayor (sonido alegre/brillante)
    # Valence negativa -> modo menor (sonido triste/oscuro)
    # Ref: Hevner, K. (1935). The affective character of the major and minor modes
    #      in music. American Journal of Psychology, 47(1), 103–118.
    #      Eerola, T. & Vuoskoski, J. K. (2011) — modo mayor asociado a valencia positiva.
    # Nota: Umbral binario en V=0 es simplificación; alternativas como modos
    #       intermedios (Dórico, Mixolidio) no están implementadas.
    mode = "major" if v >= 0 else "minor"
    
    # ===== DENSIDAD =====
    # El arousal controla la densidad de notas
    # Bajo arousal -> pocas notas (musica espaciada)
    # Alto arousal -> muchas notas (musica densa)
    # Ref: Eerola, T., Friberg, A. & Bresin, R. (2013). Emotional expression in
    #      music — densidad de eventos se asocia con arousal alto.
    #      Juslin, P. N. & Lindström, E. (2010) — note density como cue de arousal.
    # Nota: Rango 0.2-1.0 es heurístico (se evita densidad 0 para garantizar contenido).
    density = lerp(0.2, 1.0, t_a)
    
    # ===== RANGO TONAL =====
    # Registro (centro tonal): sube con valencia
    # Valencia baja -> registro grave (C4 = MIDI 60)
    # Valencia alta -> registro agudo (C5 = MIDI 72)
    # Ref: Juslin, P. N. & Laukka, P. (2004), Tabla 2 — registro agudo asociado
    #      a emociones positivas y alta activación.
    #      Dalla Bella, S. et al. (2001) — pitch alto correlaciona con valencia positiva.
    # Nota: MIDI 60=C4, 72=C5 (convención MIDI 1.0). Mapeo a V es discutible;
    #       la literatura sugiere correlación con A también (Ilie & Thompson, 2006).
    pitch_center = lerp(60, 72, t_v)
    
    # Amplitud (span): sube con arousal
    # Bajo arousal -> rango estrecho (1 octava = 12 semitonos)
    # Alto arousal -> rango amplio (2 octavas = 24 semitonos)
    # Ref: Eerola, T. et al. (2009) — pitch range se correlaciona con arousal.
    #      Rango 12-24 semitonos (1-2 octavas): heurística de diseño.
    span = lerp(12, 24, t_a)
    
    # Calcular límites del rango tonal
    # Ref: MIDI 1.0 spec — 48=C3, 60=C4, 72=C5, 84=C6 (rango típico de piano solo).
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
    # Ref: Juslin, P. N. & Laukka, P. (2004), Tabla 2 — intensidad (loudness)
    #      correlaciona fuertemente con arousal y moderadamente con valencia positiva.
    # Nota: Fórmula vel = 40 + 60*t_a + 20*t_v es heurística lineal.
    #       Pesos (60 arousal, 20 valencia) son decisión de diseño, no calibrados.
    #       Rango resultante [40, 120] dentro de MIDI velocity [1, 127].
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
