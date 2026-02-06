"""
Registry (registro) de engines de generación musical.

Proporciona funciones para obtener y listar engines disponibles.
Patrón de diseño: Registry/Factory para gestión de engines.
"""

import logging
from typing import Dict, List

from .base import MusicGenerationEngine
from .baseline_engine import BaselineEngine
from .transformer_pretrained_engine import TransformerPretrainedEngine
from .transformer_finetuned_engine import TransformerFinetunedEngine

logger = logging.getLogger(__name__)


# Registry global de engines
# Patrón Singleton: las instancias se crean una sola vez
_ENGINE_REGISTRY: Dict[str, MusicGenerationEngine] = {}


def _initialize_registry():
    """
    Inicializa el registry con todos los engines disponibles.
    
    Se ejecuta lazy (primera vez que se necesita).
    """
    if _ENGINE_REGISTRY:
        return  # Ya inicializado
    
    logger.info("Inicializando registry de engines de generación musical")
    
    # Registrar engine baseline
    try:
        baseline = BaselineEngine()
        _ENGINE_REGISTRY['baseline'] = baseline
        logger.info(f"✓ Engine registrado: {baseline.name} - {baseline.description}")
    except Exception as e:
        logger.error(f"Error al registrar engine baseline: {str(e)}")
    
    # Registrar engine transformer_pretrained
    try:
        transformer_pretrained = TransformerPretrainedEngine()
        _ENGINE_REGISTRY['transformer_pretrained'] = transformer_pretrained
        logger.info(f"✓ Engine registrado: {transformer_pretrained.name} - {transformer_pretrained.description}")
    except Exception as e:
        logger.error(f"Error al registrar engine transformer_pretrained: {str(e)}")
    
    # Registrar engine transformer_finetuned (placeholder)
    try:
        transformer_finetuned = TransformerFinetunedEngine()
        _ENGINE_REGISTRY['transformer_finetuned'] = transformer_finetuned
        logger.info(f"✓ Engine registrado: {transformer_finetuned.name} - {transformer_finetuned.description}")
    except Exception as e:
        logger.error(f"Error al registrar engine transformer_finetuned: {str(e)}")
    
    logger.info(f"Registry inicializado con {len(_ENGINE_REGISTRY)} engine(s)")


def get_engine(name: str) -> MusicGenerationEngine:
    """
    Obtiene un engine por su nombre.
    
    Args:
        name: Nombre del engine ('baseline', 'transformer_pretrained', 'transformer_finetuned')
    
    Returns:
        Instancia del engine solicitado
    
    Raises:
        ValueError: Si el engine no existe
    
    Example:
        >>> engine = get_engine('baseline')
        >>> result = engine.generate(valence=0.7, arousal=0.6, out_path='output.mid')
    """
    # Lazy initialization
    _initialize_registry()
    
    if name not in _ENGINE_REGISTRY:
        available = list(_ENGINE_REGISTRY.keys())
        raise ValueError(
            f"Engine '{name}' no existe. "
            f"Engines disponibles: {available}"
        )
    
    return _ENGINE_REGISTRY[name]


def list_engines() -> List[Dict[str, str]]:
    """
    Lista todos los engines disponibles con sus descripciones.
    
    Returns:
        Lista de dicts con información de cada engine:
        [
            {
                'name': 'baseline',
                'description': '...',
                'available': True/False
            },
            ...
        ]
    
    Example:
        >>> engines = list_engines()
        >>> for engine in engines:
        ...     print(f"{engine['name']}: {engine['description']}")
    """
    # Lazy initialization
    _initialize_registry()
    
    engines_info = []
    
    for name, engine in _ENGINE_REGISTRY.items():
        # Verificar disponibilidad
        available = True
        
        # transformer_finetuned no está disponible (placeholder)
        if name == 'transformer_finetuned':
            available = False
        
        # transformer_pretrained requiere checkpoint
        if name == 'transformer_pretrained':
            try:
                # Verificar si el checkpoint existe
                checkpoint_path = engine.checkpoint_path
                available = checkpoint_path.exists()
            except Exception:
                available = False
        
        engines_info.append({
            'name': name,
            'description': engine.description,
            'available': available
        })
    
    return engines_info


def is_engine_available(name: str) -> bool:
    """
    Verifica si un engine está disponible para usar.
    
    Args:
        name: Nombre del engine
    
    Returns:
        True si el engine existe y está disponible, False en caso contrario
    
    Example:
        >>> if is_engine_available('transformer_pretrained'):
        ...     engine = get_engine('transformer_pretrained')
        ... else:
        ...     print("Descarga el checkpoint primero")
    """
    try:
        engines = list_engines()
        for engine_info in engines:
            if engine_info['name'] == name:
                return engine_info['available']
        return False
    except Exception:
        return False
