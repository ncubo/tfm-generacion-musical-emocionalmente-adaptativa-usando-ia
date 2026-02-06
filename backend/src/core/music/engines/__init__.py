"""
Engines para generación musical.

Este paquete contiene diferentes motores (engines) para generar música MIDI
a partir de coordenadas emocionales (Valence-Arousal).

Engines disponibles:
- baseline: Generación basada en reglas deterministas
- transformer_pretrained: Generación con modelo Transformer preentrenado (SkyTNT)
- transformer_finetuned: Generación con modelo Transformer fine-tuned (placeholder)
"""

from .registry import get_engine, list_engines

__all__ = ['get_engine', 'list_engines']
