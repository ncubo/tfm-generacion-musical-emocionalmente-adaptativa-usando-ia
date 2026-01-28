"""
Módulo core del backend.
Contiene los componentes principales del sistema:
- camera: Captura de video desde webcam
- emotion: Detección emocional facial
"""

from . import camera
from . import emotion

__all__ = ['camera', 'emotion']
