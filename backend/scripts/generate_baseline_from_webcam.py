"""
Script de generación MIDI baseline desde detección emocional por webcam.

Este script demuestra el flujo completo del sistema:
1. Captura video de webcam
2. Detecta emoción facial en tiempo real
3. Promedia valores de Valence y Arousal
4. Convierte a parámetros musicales
5. Genera un archivo MIDI baseline

Uso:
    python backend/scripts/generate_baseline_from_webcam.py [opciones]

Opciones:
    --duration SECONDS    Duración de captura en segundos (default: 10)
    --bars BARS          Número de compases a generar (default: 8)
    --output PATH        Ruta del archivo MIDI de salida (default: output/emotion.mid)
    --seed SEED          Semilla para reproducibilidad (default: None)
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.camera import WebcamCapture
from core.emotion import DeepFaceEmotionDetector
from core.pipeline import EmotionPipeline
from core.music import va_to_music_params, generate_midi_baseline


def main():
    """
    Función principal del script de generación MIDI desde webcam.
    """
    # Parsear argumentos
    parser = argparse.ArgumentParser(
        description='Genera un archivo MIDI baseline desde detección emocional por webcam'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='Duración de captura emocional en segundos (default: 10)'
    )
    parser.add_argument(
        '--bars',
        type=int,
        default=8,
        help='Número de compases a generar (default: 8)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/emotion.mid',
        help='Ruta del archivo MIDI de salida (default: output/emotion.mid)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Semilla aleatoria para reproducibilidad (default: None)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Generador MIDI Baseline desde Webcam - TFM Generación Musical")
    print("=" * 70)
    print()
    print(f"Configuración:")
    print(f"  - Duración de captura: {args.duration} segundos")
    print(f"  - Compases a generar: {args.bars}")
    print(f"  - Archivo de salida: {args.output}")
    print(f"  - Semilla aleatoria: {args.seed if args.seed else 'ninguna (aleatorio)'}")
    print()
    print("NOTA: Muestra tu emoción frente a la cámara durante la captura")
    print()
    
    # Crear componentes del pipeline
    webcam = WebcamCapture(camera_index=0)
    detector = DeepFaceEmotionDetector(enforce_detection=False)
    
    # Pipeline con suavizado moderado para captura estable
    pipeline = EmotionPipeline(
        camera=webcam,
        detector=detector,
        window_size=10
    )
    
    try:
        # Iniciar pipeline
        print("Iniciando captura de webcam...")
        pipeline.start()
        print("[OK] Camara iniciada correctamente")
        print()
        
        # Capturar emociones durante el tiempo especificado
        print(f"Capturando emoción durante {args.duration} segundos...")
        print("(Mantén una expresión emocional frente a la cámara)")
        print()
        
        valence_values = []
        arousal_values = []
        emotions_detected = []
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < args.duration:
            result = pipeline.step()
            
            # Acumular valores
            valence_values.append(result['valence'])
            arousal_values.append(result['arousal'])
            emotions_detected.append(result['emotion'])
            sample_count += 1
            
            # Mostrar progreso cada segundo
            elapsed = int(time.time() - start_time)
            if sample_count % 30 == 0:  # Aproximadamente cada segundo a 30 FPS
                print(f"  [{elapsed}s] Emoción: {result['emotion']}, V: {result['valence']:+.2f}, A: {result['arousal']:+.2f}")
            
            time.sleep(0.033)  # ~30 FPS
        
        print()
        print(f"[OK] Captura completada: {sample_count} muestras procesadas")
        print()
        
        # Calcular promedios
        avg_valence = sum(valence_values) / len(valence_values)
        avg_arousal = sum(arousal_values) / len(arousal_values)
        
        # Emoción más frecuente
        most_common_emotion = max(set(emotions_detected), key=emotions_detected.count)
        
        print("Resumen emocional:")
        print(f"  - Emoción predominante: {most_common_emotion}")
        print(f"  - Valence promedio: {avg_valence:+.3f}")
        print(f"  - Arousal promedio: {avg_arousal:+.3f}")
        print()
        
        # Convertir a parámetros musicales
        print("Convirtiendo emoción a parámetros musicales...")
        music_params = va_to_music_params(avg_valence, avg_arousal)
        
        print("Parámetros musicales generados:")
        print(f"  - Tempo: {music_params['tempo_bpm']} BPM")
        print(f"  - Modo: {music_params['mode']}")
        print(f"  - Densidad: {music_params['density']:.2f}")
        print(f"  - Rango tonal: MIDI {music_params['pitch_low']} - {music_params['pitch_high']}")
        print(f"  - Complejidad rítmica: {music_params['rhythm_complexity']:.2f}")
        print(f"  - Velocity: {music_params['velocity_mean']} ± {music_params['velocity_spread']}")
        print()
        
        # Generar archivo MIDI
        print(f"Generando archivo MIDI ({args.bars} compases)...")
        output_path = generate_midi_baseline(
            params=music_params,
            out_path=args.output,
            length_bars=args.bars,
            seed=args.seed
        )
        
        print()
        print("=" * 70)
        print("[OK] GENERACION COMPLETADA")
        print("=" * 70)
        print(f"Archivo MIDI guardado en: {output_path}")
        print()
        print("Puedes reproducir el archivo con cualquier reproductor MIDI o DAW")
        print()
        
    except RuntimeError as e:
        print(f"[ERROR] Error: {e}")
        print()
        print("Soluciones posibles:")
        print("  1. Verifica que la webcam esté conectada")
        print("  2. Asegúrate de que ninguna otra aplicación esté usando la cámara")
        print("  3. Verifica los permisos de acceso a la cámara")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print()
        print()
        print("[OK] Interrumpido por el usuario")
        
    except Exception as e:
        print(f"[ERROR] Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Liberar recursos
        pipeline.stop()
        print("[OK] Recursos liberados correctamente")
        print()


if __name__ == "__main__":
    main()
