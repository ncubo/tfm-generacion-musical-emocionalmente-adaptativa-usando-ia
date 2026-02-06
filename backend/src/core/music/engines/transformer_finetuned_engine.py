"""
Engine de generación musical con Transformer fine-tuned (PLACEHOLDER).

Este engine es un placeholder para futuras implementaciones de modelos
Transformer fine-tuned en datasets específicos.

Por ahora, retorna error 501 (Not Implemented) cuando se intenta usar.
"""

import logging
from typing import Dict, Optional

from .base import MusicGenerationEngine

logger = logging.getLogger(__name__)


class TransformerFinetunedEngine(MusicGenerationEngine):
    """
    Engine placeholder para Transformer fine-tuned.
    
    Este engine aún no está implementado. Cuando se complete el fine-tuning
    del modelo en datos emocionales específicos, se implementará aquí.
    
    Características futuras:
    - Condicionamiento directo por emoción (V/A embeddings)
    - Fine-tuned en dataset con anotaciones emocionales
    - Mayor control sobre la generación que transformer_pretrained
    """
    
    @property
    def name(self) -> str:
        return "transformer_finetuned"
    
    @property
    def description(self) -> str:
        return "Transformer fine-tuned con condicionamiento emocional (no disponible aún)"
    
    def generate(
        self,
        valence: float,
        arousal: float,
        out_path: str,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Placeholder: lanza NotImplementedError.
        
        Este engine aún no está implementado y no debería ser llamado.
        El endpoint /generate-midi debe retornar 501 antes de llegar aquí.
        
        Raises:
            NotImplementedError: Siempre, ya que este engine no está listo
        """
        logger.warning(f"Intento de usar engine {self.name} que aún no está disponible")
        
        raise NotImplementedError(
            f"El engine '{self.name}' aún no está disponible. "
            f"Usa 'baseline' o 'transformer_pretrained' por ahora."
        )
