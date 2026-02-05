"""
Módulo de generación musical con Music Transformer preentrenado (PyTorch).

Este módulo proporciona generación de música simbólica usando un modelo
Transformer preentrenado, sin necesidad de fine-tuning. El condicionamiento
emocional se realiza indirectamente mediante control de parámetros de sampling
en tiempo de inferencia.

Modelo seleccionado:
    - Music Transformer con tokenización REMI (REpresentation for Music with Instruments)
    - Framework: PyTorch
    - Tokenización: miditok (REMI encoding)
    - Checkpoint: Preentrenado en datasets de música simbólica

Enfoque de condicionamiento:
    El modelo NO se entrena ni fine-tunea. El condicionamiento emocional es
    INDIRECTO mediante control de:
    - Temperature: Más arousal → mayor temperature → más creatividad
    - Top-p (nucleus sampling): Más arousal → mayor top-p → más diversidad
    - Top-k: Menos restrictivo con alto arousal

Referencias:
    - Huang et al. (2018). "Music Transformer"
    - REMI tokenization: https://github.com/YatingMusic/remi
    - miditok: https://github.com/Natooz/MidiTok
"""

from .transformer_infer import TransformerMusicGenerator

__all__ = ['TransformerMusicGenerator']
