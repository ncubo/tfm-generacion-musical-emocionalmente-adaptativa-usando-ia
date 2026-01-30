"""
Módulo de utilidades comunes del sistema.

Este paquete contiene funciones reutilizables que se usan en diferentes
partes del sistema (mapeos VA, parámetros musicales, etc.).
"""

# Importar funciones matemáticas desde el módulo math
from .math import clamp, lerp, to_unit, clamp_va

__all__ = ['clamp', 'lerp', 'to_unit', 'clamp_va']
