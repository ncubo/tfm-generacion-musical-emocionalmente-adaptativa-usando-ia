"""
Script de demostración de captura de webcam con reconocimiento emocional.

Este script muestra cómo utilizar la clase WebcamCapture junto con
DeepFaceEmotionDetector para capturar video en tiempo real desde la webcam
y detectar emociones faciales.

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


def main():
    """
    Función principal del script de demostración.
    
    Captura video de la webcam, detecta emociones faciales usando DeepFace,
    dibuja información sobre los frames y muestra la ventana hasta que
    el usuario presione 'q'.
    """
    print("=" * 70)
    print("Demo Webcam + Reconocimiento Emocional - TFM Generación Musical")
    print("=" * 70)
    print("\nPresiona 'q' para salir")
    print("\nNOTA: La primera detección puede tardar unos segundos (carga de modelos)\n")
    
    # Crear instancias de captura de webcam y detector emocional
    webcam = WebcamCapture(camera_index=0)
    emotion_detector = DeepFaceEmotionDetector(enforce_detection=False)
    
    try:
        # Iniciar la cámara
        webcam.start()
        
        # Obtener y mostrar propiedades de la cámara
        props = webcam.get_properties()
        print(f"Propiedades de la cámara:")
        print(f"  - Resolución: {props.get('width')}x{props.get('height')}")
        print(f"  - FPS: {props.get('fps')}")
        print()
        
        frame_count = 0
        
        # Inicializar variables de detección emocional
        current_emotion = 'neutral'
        face_detected = False
        probabilities = {}
        
        # Bucle principal de captura y detección
        while True:
            # Leer frame de la cámara
            success, frame = webcam.read()
            
            if not success:
                print("Error: No se pudo leer el frame")
                break
            
            frame_count += 1
            
            # Detectar emoción en el frame (cada N frames para mejorar performance)
            # Para mejorar FPS, solo analizamos cada 10 frames
            if frame_count % 10 == 0 or frame_count == 1:
                emotion_result = emotion_detector.predict(frame)
                current_emotion = emotion_result['emotion']
                face_detected = emotion_result['face_detected']
                probabilities = emotion_result['probabilities']
            
            # Dibujar información sobre el frame
            # Fondo semi-transparente para el texto
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 140), (0, 0, 0), -1)
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
            
            # Información de control
            cv2.putText(
                frame,
                "Presiona 'q' para salir",
                (20, 135),
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
        webcam.release()
        cv2.destroyAllWindows()
        print("✓ Recursos liberados correctamente")


if __name__ == "__main__":
    main()
