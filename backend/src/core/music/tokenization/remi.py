"""
Módulo de tokenización REMI para archivos MIDI.

Proporciona funciones para convertir archivos MIDI a tokens REMI usando miditok,
compatible con el modelo Maestro-REMI-bpe20k.

Este módulo reutiliza la configuración y tokenizador del engine existente para
garantizar compatibilidad perfecta entre inferencia y fine-tuning.
"""

import logging
from pathlib import Path
from typing import List, Any, Optional

logger = logging.getLogger(__name__)

# Singleton para tokenizador (evitar recarga múltiple)
_tokenizer = None


def load_remi_tokenizer(model_id: str = "Natooz/Maestro-REMI-bpe20k") -> Any:
    """
    Carga el tokenizador REMI desde HuggingFace Hub.
    
    Usa singleton pattern para evitar recargas múltiples.
    
    Args:
        model_id: ID del modelo en HuggingFace Hub
        
    Returns:
        Tokenizador REMI (miditok.REMI instance)
        
    Raises:
        RuntimeError: Si falla la carga del tokenizador
    """
    global _tokenizer
    
    if _tokenizer is None:
        try:
            import miditok
            logger.info(f"Cargando tokenizador REMI desde: {model_id}")
            _tokenizer = miditok.REMI.from_pretrained(model_id)
            logger.info(f"Tokenizador REMI cargado: vocab_size={len(_tokenizer)}")
        except ImportError as e:
            raise RuntimeError(
                "miditok no está instalado. "
                "Ejecuta: pip install miditok>=3.0"
            ) from e
        except Exception as e:
            logger.error(f"Error al cargar tokenizador: {e}", exc_info=True)
            raise RuntimeError(
                f"No se pudo cargar tokenizador {model_id}. "
                f"Verifica conexión a internet."
            ) from e
    
    return _tokenizer


def midi_to_remi_tokens(
    midi_path: str,
    tokenizer: Optional[Any] = None,
    model_id: str = "Natooz/Maestro-REMI-bpe20k"
) -> List[int]:
    """
    Convierte un archivo MIDI a secuencia de tokens REMI.
    
    Pipeline:
    1. Carga MIDI como symusic.Score
    2. Tokeniza con miditok REMI
    3. Retorna lista de IDs de tokens
    
    Args:
        midi_path: Path al archivo MIDI
        tokenizer: Tokenizador REMI (opcional, se carga automáticamente si None)
        model_id: ID del modelo HF (usado si tokenizer es None)
        
    Returns:
        Lista de IDs de tokens (enteros)
        
    Raises:
        FileNotFoundError: Si el archivo MIDI no existe
        RuntimeError: Si falla la tokenización
    """
    midi_path_obj = Path(midi_path)
    
    if not midi_path_obj.exists():
        raise FileNotFoundError(f"Archivo MIDI no encontrado: {midi_path}")
    
    # Cargar tokenizador si no se proporciona
    if tokenizer is None:
        tokenizer = load_remi_tokenizer(model_id)
    
    try:
        # Lazy import de symusic
        import symusic
        
        # Cargar MIDI como Score
        score = symusic.Score(str(midi_path_obj))
        
        # Tokenizar (retorna lista de TokSequence, uno por pista)
        tok_sequences = tokenizer(score)
        
        # Si es una lista, tomar la primera pista (asumiendo piano solo)
        if isinstance(tok_sequences, list):
            if len(tok_sequences) == 0:
                logger.warning(f"MIDI sin pistas tokenizables: {midi_path}")
                return []
            tok_sequence = tok_sequences[0]
        else:
            tok_sequence = tok_sequences
        
        # Extraer IDs
        token_ids = tok_sequence.ids
        
        logger.debug(f"MIDI tokenizado: {midi_path_obj.name} -> {len(token_ids)} tokens")
        
        return token_ids
        
    except ImportError as e:
        raise RuntimeError(
            "symusic no está instalado (requerido por miditok 3.0+). "
            "Ejecuta: pip install miditok>=3.0"
        ) from e
    except Exception as e:
        logger.error(f"Error tokenizando {midi_path}: {e}", exc_info=True)
        raise RuntimeError(f"Fallo al tokenizar MIDI: {e}") from e
