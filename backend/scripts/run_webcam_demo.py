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
from core.pipeline import EmotionPipeline


# Configuración del demo
WINDOW_SIZE = 15  # Tamaño de ventana para suavizado temporal
ANALYSIS_INTERVAL = 5  # Analizar cada N frames para mejor rendimiento


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
    detector = DeepFaceEmotionDetector(enforce_detection=False)
    
    # Crear pipeline integrado con suavizado temporal
    # Mayor window_size = cambios más graduales y suaves
    pipeline = EmotionPipeline(
        camera=webcam,
        detector=detector,
        window_size=WINDOW_SIZE
    )
    
    try:
        # Iniciar el pipeline
        pipeline.start()
        
        # Obtener y mostrar propiedades de la cámara
        props = webcam.get_properties()
        print(f"Propiedades de la cámara:")
        print(f"  - Resolución: {props.get('width')}x{props.get('height')}")
        print(f"  - FPS: {props.get('fps')}")
        print(f"  - Suavizado temporal: ventana de {WINDOW_SIZE} frames")
        print(f"  - Análisis cada {ANALYSIS_INTERVAL} frames (para mejor rendimiento)")
        print()
        
        frame_count = 0
        current_emotion = 'neutral'
        current_valence = 0.0
        current_arousal = 0.0
        current_scores = {}
        
        # Bucle principal de captura y procesamiento
        while True:
            # Leer frame de la cámara
            success, frame = webcam.read()
            
            if not success or frame is None:
                print("Error: No se pudo leer el frame")
                break
            
            frame_count += 1
            
            # Procesar frame con el pipeline periódicamente para mejor rendimiento
            # Esto también ayuda a estabilizar las detecciones
            if frame_count % ANALYSIS_INTERVAL == 0 or frame_count == 1:
                result = pipeline.step()
                
                # Extraer datos del resultado
                current_emotion = result['emotion']
                current_valence = result['valence']
                current_arousal = result['arousal']
                current_scores = result['scores']
            
            # Usar los últimos valores conocidos
            emotion = current_emotion
            valence = current_valence
            arousal = current_arousal
            scores = current_scores
            
            # Dibujar información sobre el frame
            # Fondo semi-transparente para el texto
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Texto de emoción en español
            emotion_spanish = detector.get_emotion_label_spanish(emotion)
            emotion_text = f"Emocion: {emotion_spanish}"
            cv2.putText(
                frame,
                emotion_text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Mostrar coordenadas Valence-Arousal
            va_text = f"V: {valence:+.2f}  A: {arousal:+.2f}"
            cv2.putText(
                frame,
                va_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 200, 255),
                2,
                cv2.LINE_AA
            )
            
            # Mostrar top-1 score si existe
            if scores:
                top_score = max(scores.values()) if scores else 0.0
                score_text = f"Confianza: {top_score:.1f}%"
                cv2.putText(
                    frame,
                    score_text,
                    (20, 110),
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
                (20, 130),
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
                print("\n[OK] Saliendo del demo...")
                break
        
        print(f"[OK] Total de frames procesados: {frame_count}")
        
    except RuntimeError as e:
        print(f"[ERROR] Error: {e}")
        print("\nSoluciones posibles:")
        print("  1. Verifica que la webcam esté conectada")
        print("  2. Asegúrate de que ninguna otra aplicación esté usando la cámara")
        print("  3. Verifica los permisos de acceso a la cámara")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n[OK] Interrumpido por el usuario")
        
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Liberar recursos
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[OK] Recursos liberados correctamente")


if __name__ == "__main__":
    main()
