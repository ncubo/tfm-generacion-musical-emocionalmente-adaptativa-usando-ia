"""
Script de demostración de captura de webcam con reconocimiento emocional.

Este script utiliza el EmotionPipeline integrado que conecta captura de video,
detección emocional facial y mapeo a coordenadas Valence-Arousal en tiempo real.

Uso:
    python backend/scripts/run_webcam_demo.py

Controles:
    - Presiona 'q' para salir
"""

import sys
import os

# Añadir el directorio src al path para poder importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
from core.camera import WebcamCapture
from core.emotion import DeepFaceEmotionDetector
from core.va import emotion_to_va
from core.pipeline import EmotionPipeline


def main():
    """
    Función principal del script de demostración.
    
    Captura video de la webcam, detecta emociones faciales usando DeepFace,
    mapea a coordenadas VA y dibuja información sobre los frames hasta que
    el usuario presione 'q'.
    """
    print("=" * 70)
    print("Demo Webcam + Reconocimiento Emocional - TFM Generación Musical")
    print("=" * 70)
    print("\nPresiona 'q' para salir")
    print("\nNOTA: La primera detección puede tardar unos segundos (carga de modelos)\n")
    
    # Crear componentes del pipeline
    webcam = WebcamCapture(camera_index=0)
    emotion_detector = DeepFaceEmotionDetector(enforce_detection=False)
    
    # Crear pipeline integrado con suavizado temporal (ventana de 5 frames)
    pipeline = EmotionPipeline(
        camera=webcam,
        emotion_detector=emotion_detector,
        va_mapper=emotion_to_va,
        window_size=5
    )
    
    try:
        # Iniciar el pipeline
        pipeline.start()
        
        # Obtener y mostrar propiedades de la cámara
        props = webcam.get_properties()
        print(f"Propiedades de la cámara:")
        print(f"  - Resolución: {props.get('width')}x{props.get('height')}")
        print(f"  - FPS: {props.get('fps')}")
        print(f"  - Suavizado temporal: ventana de 5 frames")
        print()
        
        frame_count = 0
        
        # Bucle principal de captura y procesamiento
        while True:
            # Leer frame de la cámara
            success, frame = webcam.read()
            
            if not success:
                print("Error: No se pudo leer el frame")
                break
            
            frame_count += 1
            
            # Procesar frame con el pipeline (cada N frames para mejorar performance)
            # Para mejorar FPS, solo analizamos cada 10 frames
            if frame_count % 10 == 0 or frame_count == 1:
                result = pipeline.step()
                current_emotion = result['emotion']
                face_detected = result['face_detected']
                probabilities = result['probabilities']
                valence = result['valence']
                arousal = result['arousal']
            else:
                # Mantener último estado conocido
                state = pipeline.get_current_state()
                current_emotion = state['emotion']
                face_detected = state['face_detected']
                probabilities = state.get('probabilities', {})
                valence = state['valence']
                arousal = state['arousal']
            
            # Dibujar información sobre el frame
            # Fondo semi-transparente para el texto
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Texto de emoción o estado
            if face_detected:
                # Obtener etiqueta en español para visualización
                emotion_spanish = emotion_detector.get_emotion_label_spanish(current_emotion)
                display_text = f"Emocion: {emotion_spanish}"
                emotion_color = (0, 255, 0)  # Verde
            else:
                display_text = "Sin rostro detectado"
                emotion_color = (0, 165, 255)  # Naranja
            
            cv2.putText(
                frame,
                display_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                emotion_color,
                2,
                cv2.LINE_AA
            )
            
            # Estado de detección de rostro
            face_status = "Rostro detectado" if face_detected else "Sin rostro"
            cv2.putText(
                frame,
                face_status,
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            # Probabilidad de la emoción dominante (si hay rostro)
            if face_detected and probabilities:
                prob_value = probabilities.get(current_emotion, 0)
                cv2.putText(
                    frame,
                    f"Confianza: {prob_value:.1f}%",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
            
            # Mostrar coordenadas Valence-Arousal
            va_text = f"V: {valence:+.2f}  A: {arousal:+.2f}"
            cv2.putText(
                frame,
                va_text,
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 200, 255),  # Color amarillo-naranja
                2,
                cv2.LINE_AA
            )
            
            # Información de control
            cv2.putText(
                frame,
                "Presiona 'q' para salir",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )
            
            # Mostrar el frame en una ventana
            cv2.imshow('TFM - Reconocimiento Emocional', frame)
            
            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n✓ Saliendo del demo...")
                break
        
        print(f"✓ Total de frames procesados: {frame_count}")
        
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("\nSoluciones posibles:")
        print("  1. Verifica que la webcam esté conectada")
        print("  2. Asegúrate de que ninguna otra aplicación esté usando la cámara")
        print("  3. Verifica los permisos de acceso a la cámara")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n✓ Interrumpido por el usuario")
        
    except Exception as e:
        print(f"\n✗ Error inesperado: {e}")
        sys.exit(1)
        
    finally:
        # Liberar recursos
        pipeline.stop()
        cv2.destroyAllWindows()
        print("✓ Recursos liberados correctamente")


if __name__ == "__main__":
    main()
