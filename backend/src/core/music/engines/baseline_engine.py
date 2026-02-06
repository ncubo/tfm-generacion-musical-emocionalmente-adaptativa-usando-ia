"""
Engine de generación musical basado en reglas (Baseline).

Este engine envuelve el generador baseline existente (baseline_rules.py)
para cumplir con la interfaz MusicGenerationEngine.

Es un engine determinista que mapea coordenadas emocionales V/A a parámetros
musicales explícitos (tempo, modo, densidad, etc.) y genera MIDI mediante reglas.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from .base import MusicGenerationEngine
from ..mapping import va_to_music_params
from ..baseline_rules import generate_midi_baseline

logger = logging.getLogger(__name__)


class BaselineEngine(MusicGenerationEngine):
    """
    Engine de generación baseline basado en reglas deterministas.
    
    Este engine:
    1. Mapea coordenadas V/A a parámetros musicales (tempo, modo, densidad, etc.)
    2. Genera MIDI usando reglas composicionales explícitas
    3. No usa modelos de ML, solo lógica programada
    
    Ventajas:
    - Rápido y predecible
    - No requiere recursos computacionales intensivos
    - Útil como baseline para comparación
    
    Limitaciones:
    - Música simple y repetitiva
    - Poca diversidad
    - No aprende de datos
    """
    
    def __init__(self, length_bars: int = 8):
        """
        Inicializa el engine baseline.
        
        Args:
            length_bars: Número de compases a generar (default: 8)
        """
        self.length_bars = length_bars
    
    @property
    def name(self) -> str:
        return "baseline"
    
    @property
    def description(self) -> str:
        return "Generación basada en reglas deterministas (mapeo V/A -> parámetros musicales)"
    
    def generate(
        self,
        valence: float,
        arousal: float,
        out_path: str,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Genera música MIDI usando reglas deterministas.
        
        Args:
            valence: Valencia emocional en [0, 1] o [-1, 1]
            arousal: Activación emocional en [0, 1] o [-1, 1]
            out_path: Ruta donde guardar el MIDI
            seed: Semilla para reproducibilidad
            **kwargs: Parámetros adicionales
                - length_bars: Número de compases (override del default)
        
        Returns:
            Dict con metadata de la generación
        """
        # Normalizar V/A a [0, 1] si es necesario
        # (va_to_music_params espera valores en [0, 1])
        v = (valence + 1) / 2 if valence < 0 else valence
        a = (arousal + 1) / 2 if arousal < 0 else arousal
        
        # Validar rangos
        if not (0 <= v <= 1):
            raise ValueError(f"Valence debe estar en [0, 1] o [-1, 1], recibido: {valence}")
        if not (0 <= a <= 1):
            raise ValueError(f"Arousal debe estar en [0, 1] o [-1, 1], recibido: {arousal}")
        
        # Mapear V/A a parámetros musicales
        logger.info(f"Mapeando V={v:.2f}, A={a:.2f} a parámetros musicales")
        music_params = va_to_music_params(v, a)
        
        # Parámetros de generación
        length_bars = kwargs.get('length_bars', self.length_bars)
        
        # Generar MIDI
        logger.info(f"Generando MIDI baseline con {length_bars} compases")
        logger.debug(f"Parámetros musicales: {music_params}")
        
        generated_path = generate_midi_baseline(
            params=music_params,
            out_path=out_path,
            length_bars=length_bars,
            seed=seed
        )
        
        # Preparar respuesta
        return {
            'midi_path': generated_path,
            'engine': self.name,
            'valence': valence,
            'arousal': arousal,
            'generation_params': {
                **music_params,
                'length_bars': length_bars,
                'seed': seed
            }
        }
