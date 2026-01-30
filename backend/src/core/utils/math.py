"""
Utilidades matemáticas comunes del sistema.

Este módulo centraliza funciones matemáticas reutilizables que se usan
en diferentes partes del sistema (mapeos VA, parámetros musicales, etc.).
"""

from typing import Tuple


def clamp(x: float, lo: float, hi: float) -> float:
    """
    Restringe un valor al rango [lo, hi].
    
    Args:
        x (float): Valor a restringir
        lo (float): Límite inferior
        hi (float): Límite superior
    
    Returns:
        float: Valor restringido al rango [lo, hi]
    
    Examples:
        >>> clamp(5.0, 0.0, 10.0)
        5.0
        >>> clamp(15.0, 0.0, 10.0)
        10.0
        >>> clamp(-5.0, 0.0, 10.0)
        0.0
    """
    return max(lo, min(hi, x))


def lerp(lo: float, hi: float, t: float) -> float:
    """
    Interpolación lineal entre dos valores.
    
    Calcula: lo + (hi - lo) * t
    
    Args:
        lo (float): Valor inicial (cuando t=0)
        hi (float): Valor final (cuando t=1)
        t (float): Factor de interpolación [0, 1]
    
    Returns:
        float: Valor interpolado
    
    Examples:
        >>> lerp(0.0, 100.0, 0.5)
        50.0
        >>> lerp(60, 180, 0.0)
        60.0
        >>> lerp(60, 180, 1.0)
        180.0
    """
    return lo + (hi - lo) * t


def to_unit(x: float) -> float:
    """
    Convierte de rango [-1, 1] a rango [0, 1] con clamping.
    
    Fórmula: (x + 1) / 2, restringido a [0, 1]
    
    Args:
        x (float): Valor en rango [-1, 1] (se acepta fuera de rango)
    
    Returns:
        float: Valor normalizado en [0, 1]
    
    Examples:
        >>> to_unit(-1.0)
        0.0
        >>> to_unit(0.0)
        0.5
        >>> to_unit(1.0)
        1.0
        >>> to_unit(2.0)  # Fuera de rango
        1.0
    """
    return clamp((x + 1.0) / 2.0, 0.0, 1.0)


def clamp_va(valence: float, arousal: float) -> Tuple[float, float]:
    """
    Asegura que los valores de valence y arousal estén en el rango [-1, 1].
    
    Esta función es útil para validar o corregir coordenadas VA que puedan
    haber sido modificadas por cálculos externos (ej. interpolación, 
    transformaciones) y garantizar que permanezcan dentro del rango válido.
    
    Args:
        valence (float): Valor de valencia a restringir
        arousal (float): Valor de arousal a restringir
    
    Returns:
        Tuple[float, float]: Tupla (valence, arousal) con valores en [-1, 1]
    
    Examples:
        >>> clamp_va(0.5, 0.8)
        (0.5, 0.8)
        >>> clamp_va(1.5, -0.3)
        (1.0, -0.3)
        >>> clamp_va(-2.0, 1.2)
        (-1.0, 1.0)
    """
    clamped_v = clamp(valence, -1.0, 1.0)
    clamped_a = clamp(arousal, -1.0, 1.0)
    return (clamped_v, clamped_a)
