"""
Script de comparación de estabilidad temporal del sistema emocional.

Este script permite comparar visualmente el comportamiento del sistema
con diferentes configuraciones de estabilización temporal.

Uso:
    python backend/scripts/compare_stability.py
    
Funcionalidad:
    - Muestra dos ventanas lado a lado (o secuencialmente)
    - Una con configuración minimal (sin estabilización)
    - Otra con configuración optimizada (estabilización completa)
    - Permite observar diferencias en estabilidad y responsividad
"""

import sys
import os

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
from core.camera import WebcamCapture
from core.emotion import DeepFaceEmotionDetector
from core.pipeline import EmotionPipeline


# Configuraciones a comparar
CONFIGS = {
    'minimal': {
        'window_size': 1,
        'alpha': 1.0,
        'min_confidence': 0.0,
        'label': 'Sin Estabilización'
    },
    'optimized': {
        'window_size': 7,
        'alpha': 0.3,
        'min_confidence': 60.0,
        'label': 'Con Estabilización'
    }
}

ANALYSIS_INTERVAL = 5  # Analizar cada N frames


def draw_info(frame, emotion, valence, arousal, scores, config_label):
    """
    Dibuja información sobre el frame.
    
    Args:
        frame: Frame de video
        emotion: Emoción detectada
        valence: Valor de valencia
        arousal: Valor de arousal
        scores: Scores de detección
        config_label: Etiqueta de la configuración
    """
    # Fondo semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Título de configuración
    cv2.putText(
        frame,
        config_label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 200, 0),
        2,
        cv2.LINE_AA
    )
    
    # Emoción
    emotion_text = f"Emocion: {emotion}"
    cv2.putText(
        frame,
        emotion_text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    
    # V/A
    va_text = f"V: {valence:+.3f}  A: {arousal:+.3f}"
    cv2.putText(
        frame,
        va_text,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 200, 255),
        2,
        cv2.LINE_AA
    )
    
    # Confianza
    if scores:
        top_score = max(scores.values())
        score_text = f"Confianza: {top_score:.1f}%"
        cv2.putText(
            frame,
            score_text,
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    # Control
    cv2.putText(
        frame,
        "Presiona 'q' para salir, 's' para cambiar modo",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )
    
    return frame


def main():
    """
    Función principal de comparación.
    """
    print("=" * 80)
    print("Comparación de Estabilidad Temporal - TFM Generación Musical")
    print("=" * 80)
    print()
    print("Este script compara dos configuraciones:")
    print()
    print("1. SIN estabilización:")
    print("   - window_size=1 (sin ventana de mayoría)")
    print("   - alpha=1.0 (sin suavizado EMA)")
    print("   - min_confidence=0.0 (sin filtro de confianza)")
    print()
    print("2. CON estabilización optimizada:")
    print("   - window_size=7 (ventana de mayoría)")
    print("   - alpha=0.3 (suavizado EMA moderado)")
    print("   - min_confidence=60.0 (filtro de confianza)")
    print()
    print("Presiona 's' para cambiar entre modos")
    print("Presiona 'q' para salir")
    print()
    
    # Crear componentes compartidos
    webcam = WebcamCapture(camera_index=0)
    detector = DeepFaceEmotionDetector(enforce_detection=False)
    
    # Crear ambos pipelines
    pipelines = {}
    for key, config in CONFIGS.items():
        pipelines[key] = EmotionPipeline(
            camera=webcam,
            detector=detector,
            window_size=config['window_size'],
            alpha=config['alpha'],
            min_confidence=config['min_confidence']
        )
    
    # Modo inicial
    current_mode = 'optimized'
    
    try:
        # Iniciar cámara
        webcam.start()
        print("[OK] Cámara iniciada")
        print(f"\nModo actual: {CONFIGS[current_mode]['label']}")
        print()
        
        frame_count = 0
        
        # Estados para cada pipeline
        states = {
            'minimal': {
                'emotion': 'neutral',
                'valence': 0.0,
                'arousal': 0.0,
                'scores': {}
            },
            'optimized': {
                'emotion': 'neutral',
                'valence': 0.0,
                'arousal': 0.0,
                'scores': {}
            }
        }
        
        # Bucle principal
        while True:
            # Leer frame
            success, frame = webcam.read()
            
            if not success or frame is None:
                print("Error: No se pudo leer el frame")
                break
            
            frame_count += 1
            
            # Procesar con ambos pipelines periódicamente
            if frame_count % ANALYSIS_INTERVAL == 0 or frame_count == 1:
                for key, pipeline in pipelines.items():
                    result = pipeline.step()
                    states[key]['emotion'] = result['emotion']
                    states[key]['valence'] = result['valence']
                    states[key]['arousal'] = result['arousal']
                    states[key]['scores'] = result['scores']
            
            # Obtener estado del modo actual
            state = states[current_mode]
            
            # Dibujar información
            frame_display = draw_info(
                frame.copy(),
                state['emotion'],
                state['valence'],
                state['arousal'],
                state['scores'],
                CONFIGS[current_mode]['label']
            )
            
            # Mostrar frame
            cv2.imshow('Comparación Estabilidad Temporal', frame_display)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[OK] Saliendo...")
                break
            elif key == ord('s'):
                # Cambiar modo
                current_mode = 'minimal' if current_mode == 'optimized' else 'optimized'
                print(f"\nCambiado a: {CONFIGS[current_mode]['label']}")
        
        print(f"[OK] Total de frames procesados: {frame_count}")
        
    except RuntimeError as e:
        print(f"[ERROR] Error: {e}")
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
        webcam.release()
        cv2.destroyAllWindows()
        print("[OK] Recursos liberados correctamente")


if __name__ == "__main__":
    main()
