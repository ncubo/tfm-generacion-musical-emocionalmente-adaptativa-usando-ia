"""
Interfaz base para engines de generación musical.

Define el contrato que deben cumplir todos los engines de generación MIDI.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class MusicGenerationEngine(ABC):
    """
    Interfaz base para engines de generación musical.
    
    Todos los engines deben implementar el método generate() que toma
    coordenadas emocionales (valence, arousal) y genera un archivo MIDI.
    """
    
    @abstractmethod
    def generate(
        self,
        valence: float,
        arousal: float,
        out_path: str,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Genera un archivo MIDI basado en coordenadas emocionales.
        
        Args:
            valence: Valencia emocional en [-1, 1] o [0, 1]
            arousal: Activación emocional en [-1, 1] o [0, 1]
            out_path: Ruta donde guardar el archivo MIDI generado
            seed: Semilla para reproducibilidad (opcional)
            **kwargs: Parámetros adicionales específicos del engine
        
        Returns:
            Dict con metadata de la generación:
            {
                'midi_path': str,           # Ruta del archivo generado
                'generation_params': dict,  # Parámetros utilizados
                'engine': str,              # Nombre del engine
                'valence': float,           # Valencia usada
                'arousal': float            # Arousal usado
            }
        
        Raises:
            ValueError: Si los parámetros son inválidos
            RuntimeError: Si el engine no está correctamente inicializado
            FileNotFoundError: Si faltan recursos necesarios (checkpoints, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Retorna el nombre identificador del engine."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Retorna una descripción breve del engine."""
        pass
