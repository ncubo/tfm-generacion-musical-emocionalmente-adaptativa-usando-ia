"""
Funciones de sampling (muestreo) para generación con Transformer.

Implementa estrategias de muestreo estocástico para controlar la generación
de secuencias musicales.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def get_temperature_from_arousal(arousal: float) -> float:
    """
    Mapea arousal [-1, 1] a temperature para sampling.
    
    Alta arousal (activación) → alta temperature → mayor diversidad
    Baja arousal (calma) → baja temperature → más predecibilidad
    
    Args:
        arousal: Arousal en [-1, 1]
    
    Returns:
        Temperature en [0.6, 1.4]
    """
    # Normalizar arousal de [-1, 1] a [0, 1]
    t = (arousal + 1.0) / 2.0
    # Rango de temperature: más conservador que texto (música requiere coherencia)
    temp_min, temp_max = 0.6, 1.4
    return temp_min + t * (temp_max - temp_min)


def get_top_p_from_arousal(arousal: float) -> float:
    """
    Mapea arousal [-1, 1] a top-p (nucleus sampling).
    
    Alta arousal → top-p alto → más diversidad de tokens
    Baja arousal → top-p bajo → más conservador
    
    Args:
        arousal: Arousal en [-1, 1]
    
    Returns:
        Top-p en [0.85, 0.98]
    """
    t = (arousal + 1.0) / 2.0
    p_min, p_max = 0.85, 0.98
    return p_min + t * (p_max - p_min)


def get_top_k_from_arousal(arousal: float) -> int:
    """
    Mapea arousal [-1, 1] a top-k.
    
    Alta arousal → top-k alto → más opciones consideradas
    Baja arousal → top-k bajo → más restrictivo
    
    Args:
        arousal: Arousal en [-1, 1]
    
    Returns:
        Top-k en [20, 120]
    """
    t = (arousal + 1.0) / 2.0
    k_min, k_max = 20, 120
    return int(k_min + t * (k_max - k_min))


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """
    Muestrea un token aplicando temperature, top-k y/o top-p.
    
    Args:
        logits: Logits de salida del modelo [batch_size, vocab_size]
        temperature: Factor de temperatura (>1 = más aleatorio, <1 = más determinista)
        top_k: Si se especifica, solo considera los top-k tokens más probables
        top_p: Si se especifica, usa nucleus sampling (suma acumulada hasta p)
    
    Returns:
        Índices de tokens muestreados [batch_size]
    """
    # Aplicar temperature
    logits = logits / temperature
    
    # Aplicar top-k filtering si se especifica
    if top_k is not None:
        # Obtener los k valores más grandes
        top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
        # Crear máscara: poner -inf a valores fuera del top-k
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, top_k_indices, top_k_logits)
        logits = mask
    
    # Aplicar top-p (nucleus) filtering si se especifica
    if top_p is not None and top_p < 1.0:
        # Ordenar logits de mayor a menor
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # Calcular probabilidades acumuladas
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Crear máscara para tokens con probabilidad acumulada > top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Mantener al menos el primer token (más probable)
        sorted_indices_to_remove[..., 0] = False
        
        # Mapear máscara de vuelta a los índices originales
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    # Convertir logits a probabilidades y muestrear
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token.squeeze(-1)


def sample_sequence(
    model: torch.nn.Module,
    start_tokens: torch.Tensor,
    max_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token: Optional[int] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Genera una secuencia completa usando muestreo autoregresivo.
    
    Args:
        model: Modelo transformer
        start_tokens: Tokens iniciales [batch_size, seq_len]
        max_length: Longitud máxima de la secuencia generada
        temperature: Temperature para sampling
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        eos_token: Token de fin de secuencia (opcional)
        device: Device ('cpu' o 'cuda')
    
    Returns:
        Secuencia generada [batch_size, max_length]
    """
    model.eval()
    generated = start_tokens.to(device)
    
    with torch.no_grad():
        for _ in range(max_length - start_tokens.size(1)):
            # Forward pass
            logits = model(generated)
            
            # Obtener logits del último token
            next_token_logits = logits[:, -1, :]
            
            # Muestrear siguiente token
            next_token = sample_with_temperature(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Añadir a la secuencia
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            
            # Detener si se genera EOS (si está definido)
            if eos_token is not None and (next_token == eos_token).all():
                break
    
    return generated
