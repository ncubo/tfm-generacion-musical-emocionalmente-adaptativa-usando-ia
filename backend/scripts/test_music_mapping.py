"""
Script de prueba para el mapeo VA → Parámetros Musicales.

Este script valida que la función va_to_music_params genera parámetros
musicales coherentes y dentro de los rangos esperados para diferentes
estados emocionales.

Uso:
    python backend/scripts/test_music_mapping.py
"""

import sys
import os

# Añadir el directorio src al path para poder importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.music import va_to_music_params


def print_params(emotion_label: str, v: float, a: float):
    """
    Imprime los parámetros musicales para una emoción dada.
    
    Args:
        emotion_label (str): Nombre de la emoción (para display)
        v (float): Valence
        a (float): Arousal
    """
    params = va_to_music_params(v, a)
    
    print(f"\n{'='*70}")
    print(f"Emoción: {emotion_label}")
    print(f"Valence: {v:+.2f}  |  Arousal: {a:+.2f}")
    print(f"{'='*70}")
    print(f"  Tempo:              {params['tempo_bpm']} BPM")
    print(f"  Modo:               {params['mode']}")
    print(f"  Densidad:           {params['density']:.2f}")
    print(f"  Rango tonal:        MIDI {params['pitch_low']} - {params['pitch_high']}")
    print(f"  Complejidad rítmica: {params['rhythm_complexity']:.2f}")
    print(f"  Velocity media:     {params['velocity_mean']}")
    print(f"  Velocity spread:    {params['velocity_spread']}")
    
    return params


def validate_params(params: dict, emotion_label: str):
    """
    Valida que los parámetros estén en los rangos correctos.
    
    Args:
        params (dict): Parámetros musicales generados
        emotion_label (str): Nombre de la emoción (para mensajes de error)
    
    Raises:
        AssertionError: Si algún parámetro está fuera de rango
    """
    # Validar tempo
    assert 60 <= params['tempo_bpm'] <= 180, \
        f"[{emotion_label}] Tempo fuera de rango: {params['tempo_bpm']}"
    
    # Validar modo
    assert params['mode'] in ['major', 'minor'], \
        f"[{emotion_label}] Modo inválido: {params['mode']}"
    
    # Validar densidad
    assert 0.2 <= params['density'] <= 1.0, \
        f"[{emotion_label}] Densidad fuera de rango: {params['density']}"
    
    # Validar rango tonal
    assert 48 <= params['pitch_low'] <= 60, \
        f"[{emotion_label}] pitch_low fuera de rango: {params['pitch_low']}"
    assert 72 <= params['pitch_high'] <= 84, \
        f"[{emotion_label}] pitch_high fuera de rango: {params['pitch_high']}"
    assert params['pitch_high'] >= params['pitch_low'] + 12, \
        f"[{emotion_label}] Rango tonal menor a 1 octava: {params['pitch_high'] - params['pitch_low']}"
    
    # Validar complejidad rítmica
    assert 0.0 <= params['rhythm_complexity'] <= 1.0, \
        f"[{emotion_label}] Complejidad rítmica fuera de rango: {params['rhythm_complexity']}"
    
    # Validar velocity
    assert 40 <= params['velocity_mean'] <= 120, \
        f"[{emotion_label}] velocity_mean fuera de rango: {params['velocity_mean']}"
    assert 0 <= params['velocity_spread'] <= 30, \
        f"[{emotion_label}] velocity_spread fuera de rango: {params['velocity_spread']}"


def main():
    """
    Función principal de pruebas.
    """
    print("=" * 70)
    print("TEST: Mapeo Valence-Arousal → Parámetros Musicales")
    print("=" * 70)
    
    # Casos de prueba: (label, valence, arousal)
    test_cases = [
        ("HAPPY", 0.7, 0.6),      # Valencia positiva, arousal alto
        ("SAD", -0.7, -0.4),      # Valencia negativa, arousal bajo
        ("ANGRY", -0.6, 0.7),     # Valencia negativa, arousal alto
        ("NEUTRAL", 0.0, 0.0)     # Estado neutro
    ]
    
    results = []
    
    # Generar y validar parámetros para cada caso
    for emotion_label, v, a in test_cases:
        params = print_params(emotion_label, v, a)
        validate_params(params, emotion_label)
        results.append((emotion_label, params))
    
    # Resumen de validaciones
    print(f"\n{'='*70}")
    print("RESUMEN DE VALIDACIONES")
    print(f"{'='*70}")
    
    # Verificar coherencia entre emociones
    happy_params = results[0][1]
    sad_params = results[1][1]
    angry_params = results[2][1]
    neutral_params = results[3][1]
    
    # Happy debería tener tempo más rápido que sad
    assert happy_params['tempo_bpm'] > sad_params['tempo_bpm'], \
        "Happy debería tener tempo más rápido que sad"
    print("✓ Happy tiene tempo más rápido que sad")
    
    # Happy debería ser major, sad debería ser minor
    assert happy_params['mode'] == 'major', "Happy debería ser modo major"
    assert sad_params['mode'] == 'minor', "Sad debería ser modo minor"
    print("✓ Happy es major, sad es minor")
    
    # Angry debería tener alta densidad (arousal alto)
    assert angry_params['density'] > neutral_params['density'], \
        "Angry debería tener mayor densidad que neutral"
    print("✓ Angry tiene mayor densidad que neutral")
    
    # Happy debería tener velocity más alta que sad
    assert happy_params['velocity_mean'] > sad_params['velocity_mean'], \
        "Happy debería tener velocity más alta que sad"
    print("✓ Happy tiene velocity más alta que sad")
    
    # Angry debería tener alta complejidad rítmica (arousal alto)
    assert angry_params['rhythm_complexity'] > sad_params['rhythm_complexity'], \
        "Angry debería tener mayor complejidad rítmica que sad"
    print("✓ Angry tiene mayor complejidad rítmica que sad")
    
    print(f"\n{'='*70}")
    print("✅ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
    print(f"{'='*70}")
    print(f"\nTotal de casos probados: {len(test_cases)}")
    print("Total de validaciones: 12")
    print("\nEl módulo de mapeo está listo para ser usado por el generador MIDI.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n❌ ERROR DE VALIDACIÓN: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
