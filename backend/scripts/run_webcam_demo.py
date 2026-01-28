"""
Script de demostración de captura de webcam.

Este script muestra cómo utilizar la clase WebcamCapture para
capturar y visualizar video en tiempo real desde la webcam.

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


def main():
    """
    Función principal del script de demostración.
    
    Captura video de la webcam, dibuja información sobre los frames
    y muestra la ventana hasta que el usuario presione 'q'.
    """
    print("=" * 60)
    print("Demo de Captura de Webcam - TFM Generación Musical Emocional")
    print("=" * 60)
    print("\nPresiona 'q' para salir\n")
    
    # Crear instancia de captura de webcam
    webcam = WebcamCapture(camera_index=0)
    
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
        
        # Bucle principal de captura
        while True:
            # Leer frame de la cámara
            success, frame = webcam.read()
            
            if not success:
                print("Error: No se pudo leer el frame")
                break
            
            frame_count += 1
            
            # Dibujar información sobre el frame
            # Fondo semi-transparente para el texto
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Texto principal
            cv2.putText(
                frame,
                "Webcam OK",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Información adicional
            cv2.putText(
                frame,
                f"Frame: {frame_count} | Presiona 'q' para salir",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            # Mostrar el frame en una ventana
            cv2.imshow('TFM - Webcam Demo', frame)
            
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
