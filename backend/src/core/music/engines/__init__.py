"""
Music generation engines.

Este módulo contiene diferentes motores de generación musical:
- baseline: Generador basado en reglas deterministas
- hf_maestro_remi: Generador basado en Hugging Face Transformers (Maestro-REMI)
- [futuro] finetuned: Generador con modelo fine-tuned
"""

from .baseline import generate_midi_baseline
from .hf_maestro_remi import generate_midi_hf_maestro_remi

__all__ = [
    'generate_midi_baseline',
    'generate_midi_hf_maestro_remi',
]
