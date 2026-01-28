"""
Módulo de captura de webcam usando OpenCV.

Este módulo proporciona una clase para gestionar la captura de video
desde una webcam, diseñada para ser utilizada en el sistema de reconocimiento
emocional del TFM.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class WebcamCapture:
    """
    Clase para gestionar la captura de video desde webcam.
    
    Esta clase encapsula la funcionalidad de OpenCV para abrir, leer
    y liberar la cámara de manera controlada y extensible.
    
    Attributes:
        camera_index (int): Índice de la cámara a utilizar (default: 0)
        cap (cv2.VideoCapture): Objeto de captura de OpenCV
        is_opened (bool): Estado de la cámara
    """
    
    def __init__(self, camera_index: int = 0):
        """
        Inicializa la clase WebcamCapture.
        
        Args:
            camera_index (int): Índice de la cámara a utilizar (default: 0)
        """
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
    
    def start(self) -> bool:
        """
        Abre la conexión con la webcam.
        
        Returns:
            bool: True si la cámara se abrió correctamente, False en caso contrario
            
        Raises:
            RuntimeError: Si no se puede abrir la cámara
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"No se pudo abrir la cámara con índice {self.camera_index}. "
                    "Verifica que la cámara esté conectada y no esté siendo utilizada por otra aplicación."
                )
            
            # Configurar resolución (opcional, puede ajustarse según necesidades)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_opened = True
            print(f"✓ Cámara {self.camera_index} abierta correctamente")
            return True
            
        except Exception as e:
            self.is_opened = False
            raise RuntimeError(f"Error al iniciar la cámara: {str(e)}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee un frame de la webcam.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: 
                - success (bool): True si se leyó correctamente el frame
                - frame (np.ndarray | None): Frame capturado o None si hubo error
        
        Raises:
            RuntimeError: Si se intenta leer sin haber abierto la cámara
        """
        if not self.is_opened or self.cap is None:
            raise RuntimeError(
                "La cámara no está abierta. Llama a start() antes de leer frames."
            )
        
        success, frame = self.cap.read()
        
        if not success:
            print("⚠ Advertencia: No se pudo leer el frame de la cámara")
            return False, None
        
        return success, frame
    
    def release(self) -> None:
        """
        Libera los recursos de la cámara y cierra la conexión.
        
        Este método debe llamarse siempre al finalizar el uso de la cámara
        para evitar que quede bloqueada.
        """
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("✓ Recursos de cámara liberados")
    
    def get_properties(self) -> dict:
        """
        Obtiene las propiedades actuales de la cámara.
        
        Returns:
            dict: Diccionario con propiedades de la cámara (ancho, alto, fps)
        """
        if not self.is_opened or self.cap is None:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS))
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor para asegurar liberación de recursos."""
        self.release()
