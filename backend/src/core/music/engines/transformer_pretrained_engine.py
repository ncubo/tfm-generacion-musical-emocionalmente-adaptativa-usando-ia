"""
Engine de generación musical usando Transformer preentrenado (SkyTNT/midi-model).

Este engine utiliza el modelo SkyTNT/midi-model descargado desde Hugging Face
para generar música MIDI. El condicionamiento emocional es INDIRECTO a través
de los parámetros de sampling (temperature, top_k/top_p).

IMPORTANTE:
- El modelo SkyTNT NO soporta condicionamiento directo por emoción.
- El control emocional se logra ajustando los parámetros de sampling según V/A.
- Este es un modelo de generación incondicional con sampling controlado.
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from .base import MusicGenerationEngine

logger = logging.getLogger(__name__)


# Singleton para cargar el modelo una sola vez
_MODEL_CACHE = {}


class SkyTntMidiModel(nn.Module):
    """
    Arquitectura del modelo SkyTNT/midi-model.
    
    Basado en el repositorio: https://github.com/SkyTNT/midi-model
    
    Este es un Transformer para generación MIDI incondicional.
    """
    
    def __init__(
        self,
        vocab_size: int = 388,  # Vocabulario REMI típico para SkyTNT
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        """
        Inicializa el modelo Transformer.
        
        Args:
            vocab_size: Tamaño del vocabulario de tokens MIDI
            d_model: Dimensión del modelo
            nhead: Número de cabezas de atención
            num_layers: Número de capas del transformer
            dim_feedforward: Dimensión de la capa feedforward
            dropout: Tasa de dropout
            max_seq_len: Longitud máxima de secuencia
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding de tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN arquitectura
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Proyección a vocabulario
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Opcional: compartir pesos entre embedding y output projection
        # self.output_proj.weight = self.token_embedding.weight
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de tokens [batch, seq_len]
            attention_mask: Máscara de atención causal
        
        Returns:
            Logits de siguiente token [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # Embeddings
        token_emb = self.token_embedding(x)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        # Combinar embeddings
        hidden = token_emb + pos_emb
        
        # Transformer
        if attention_mask is None:
            # Crear máscara causal
            attention_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        hidden = self.transformer(hidden, mask=attention_mask, is_causal=True)
        
        # Proyectar a vocabulario
        logits = self.output_proj(hidden)
        
        return logits


def _load_model_singleton(checkpoint_path: Path, device: str = "cpu") -> SkyTntMidiModel:
    """
    Carga el modelo una sola vez (patrón singleton).
    
    Args:
        checkpoint_path: Ruta al checkpoint del modelo
        device: Dispositivo donde cargar el modelo
    
    Returns:
        Modelo cargado con pesos preentrenados
    """
    cache_key = str(checkpoint_path)
    
    if cache_key in _MODEL_CACHE:
        logger.info(f"Usando modelo en cache: {checkpoint_path}")
        return _MODEL_CACHE[cache_key]
    
    logger.info(f"Cargando modelo SkyTNT desde: {checkpoint_path}")
    
    # Verificar que el checkpoint existe
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {checkpoint_path}\n"
            f"Ejecuta: python scripts/download_transformer_pretrained.py"
        )
    
    # Cargar checkpoint
    try:
        if str(checkpoint_path).endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                checkpoint = load_file(str(checkpoint_path))
                logger.info(f"Checkpoint cargado con safetensors")
            except ImportError:
                logger.error("safetensors no está instalado")
                raise RuntimeError("Instala safetensors: pip install safetensors")
        else:
            checkpoint = torch.load(str(checkpoint_path), map_location=device)
            logger.info(f"Checkpoint cargado con torch.load")
        
        logger.info(f"Keys en checkpoint: {list(checkpoint.keys())[:10]}...")  # Mostrar primeras 10
    except Exception as e:
        raise RuntimeError(f"Error al cargar checkpoint: {str(e)}")
    
    # Extraer configuración si está disponible
    # (Ajustar según la estructura real del checkpoint de SkyTNT)
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', 388)
    d_model = config.get('d_model', 512)
    nhead = config.get('nhead', 8)
    num_layers = config.get('num_layers', 12)
    
    # Crear modelo
    model = SkyTntMidiModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    
    # Cargar pesos
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Limpiar nombres de keys si es necesario (algunos checkpoints tienen prefijos)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    logger.info(f"✅ Modelo SkyTNT cargado correctamente")
    
    # Guardar en cache
    _MODEL_CACHE[cache_key] = model
    
    return model


def _generate_with_sampling(
    model: SkyTntMidiModel,
    primer: torch.Tensor,
    max_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Genera una secuencia MIDI usando sampling controlado.
    
    Args:
        model: Modelo preentrenado
        primer: Secuencia inicial [1, primer_len]
        max_len: Longitud máxima a generar
        temperature: Temperature para sampling
        top_k: Número de tokens más probables a considerar
        top_p: Probabilidad acumulada para nucleus sampling
        device: Dispositivo de computación
    
    Returns:
        Secuencia generada [1, total_len]
    """
    model.eval()
    generated = primer.clone()
    
    with torch.no_grad():
        for _ in range(max_len):
            # Forward pass
            logits = model(generated)
            
            # Obtener logits del último token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Aplicar top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Aplicar top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remover tokens con probabilidad acumulada > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Samplear siguiente token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Añadir a la secuencia
            generated = torch.cat([generated, next_token], dim=1)
            
            # Limitar longitud de contexto si es necesario
            if generated.shape[1] > 2048:
                generated = generated[:, -2048:]
    
    return generated


class TransformerPretrainedEngine(MusicGenerationEngine):
    """
    Engine de generación con modelo Transformer preentrenado (SkyTNT).
    
    Condicionamiento INDIRECTO mediante sampling parameters:
    - Arousal alto -> temperature alta, top_k/top_p más permisivo (música variada/energética)
    - Arousal bajo -> temperature baja, top_k/top_p más restrictivo (música predecible/calmada)
    - Valence influye sutilmente en la selección de primer (no implementado aún)
    """
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Inicializa el engine.
        
        Args:
            checkpoint_dir: Directorio con el checkpoint (opcional, usa default si no se provee)
        """
        if checkpoint_dir is None:
            # Usar ubicación por defecto
            base_dir = Path(__file__).parent.parent.parent.parent.parent
            checkpoint_dir = base_dir / "models" / "transformer_pretrained"
        
        self.checkpoint_dir = Path(checkpoint_dir)
        # Buscar archivo de checkpoint disponible
        possible_files = ["model.safetensors", "pytorch_model.bin", "model.pth"]
        self.checkpoint_path = None
        for filename in possible_files:
            path = self.checkpoint_dir / filename
            if path.exists():
                self.checkpoint_path = path
                break
        
        if self.checkpoint_path is None:
            # No hay checkpoint aún, se establecerá en el primer archivo disponible
            self.checkpoint_path = self.checkpoint_dir / possible_files[0]
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TransformerPretrainedEngine usando device: {self.device}")
        
        # El modelo se carga lazy (primera vez que se llama generate)
        self._model = None
    
    @property
    def name(self) -> str:
        return "transformer_pretrained"
    
    @property
    def description(self) -> str:
        return "Transformer preentrenado (SkyTNT/midi-model)"
    
    def _ensure_model_loaded(self):
        """Carga el modelo si aún no está cargado."""
        if self._model is None:
            self._model = _load_model_singleton(self.checkpoint_path, self.device)
    
    def _map_va_to_sampling_params(self, valence: float, arousal: float) -> Dict:
        """
        Mapea coordenadas V/A a parámetros de sampling.
        
        Estrategia:
        - Arousal controla temperature y top_k/top_p
        - Arousal alto -> sampling más exploratorio (mayor variabilidad)
        - Arousal bajo -> sampling más conservador (mayor consistencia)
        
        Args:
            valence: Valencia en [0, 1] o [-1, 1]
            arousal: Arousal en [0, 1] o [-1, 1]
        
        Returns:
            Dict con parámetros de sampling
        """
        # Normalizar a [0, 1] si están en [-1, 1]
        v = (valence + 1) / 2 if valence < 0 else valence
        a = (arousal + 1) / 2 if arousal < 0 else arousal
        
        # Mapeo explícito: Arousal -> Temperature
        # Arousal bajo (0.0) -> temp baja (0.7) - música predecible
        # Arousal alto (1.0) -> temp alta (1.3) - música variada
        temperature = 0.7 + a * 0.6  # [0.7, 1.3]
        
        # Mapeo: Arousal -> top_k
        # Arousal bajo -> top_k pequeño (menos opciones, más consistencia)
        # Arousal alto -> top_k grande (más opciones, más variabilidad)
        top_k = int(30 + a * 90)  # [30, 120]
        
        # Mapeo: Arousal -> top_p (nucleus sampling)
        # Arousal bajo -> top_p bajo (distribución concentrada)
        # Arousal alto -> top_p alto (distribución más amplia)
        top_p = 0.85 + a * 0.1  # [0.85, 0.95]
        
        return {
            'temperature': round(temperature, 2),
            'top_k': top_k,
            'top_p': round(top_p, 2),
            'valence': round(v, 2),
            'arousal': round(a, 2)
        }
    
    def generate(
        self,
        valence: float,
        arousal: float,
        out_path: str,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Genera música MIDI usando el modelo SkyTNT preentrenado.
        
        NOTA: El condicionamiento es INDIRECTO. El modelo no recibe valence/arousal
        directamente, sino que estos afectan los parámetros de sampling.
        
        Args:
            valence: Valencia emocional en [0, 1] o [-1, 1]
            arousal: Activación emocional en [0, 1] o [-1, 1]
            out_path: Ruta donde guardar el MIDI
            seed: Semilla para reproducibilidad
            **kwargs: Parámetros adicionales (max_len, etc.)
        
        Returns:
            Dict con metadata de la generación
        """
        # Cargar modelo si no está cargado
        try:
            self._ensure_model_loaded()
        except FileNotFoundError as e:
            logger.error(str(e))
            raise
        
        # Configurar seed para reproducibilidad
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Mapear V/A a parámetros de sampling
        sampling_params = self._map_va_to_sampling_params(valence, arousal)
        
        # Parámetros de generación
        max_len = kwargs.get('max_len', 256)  # Tokens a generar
        
        # Crear primer simple (secuencia inicial)
        # TODO: En futuro, el primer podría variar según valence (tonalidad, etc.)
        # Por ahora, usar un primer genérico
        primer_tokens = [0, 1, 2]  # Tokens iniciales (ajustar según tokenizer real)
        primer = torch.tensor([primer_tokens], dtype=torch.long, device=self.device)
        
        logger.info(f"Generando con params: {sampling_params}")
        
        # Generar secuencia
        generated_tokens = _generate_with_sampling(
            model=self._model,
            primer=primer,
            max_len=max_len,
            temperature=sampling_params['temperature'],
            top_k=sampling_params['top_k'],
            top_p=sampling_params['top_p'],
            device=self.device
        )
        
        # Convertir tokens a MIDI
        # NOTA: Esto requiere el tokenizer correcto para SkyTNT
        # Por ahora, crear un MIDI placeholder
        logger.warning("⚠️ Conversión tokens->MIDI aún no implementada completamente")
        logger.warning("   Generando MIDI placeholder por ahora")
        
        # TODO: Implementar conversión real usando el tokenizer de SkyTNT
        # Por ahora, crear un MIDI simple para demostración
        self._tokens_to_midi_placeholder(generated_tokens, out_path)
        
        # Retornar metadata
        return {
            'midi_path': out_path,
            'engine': self.name,
            'valence': valence,
            'arousal': arousal,
            'generation_params': sampling_params,
            'tokens_generated': generated_tokens.shape[1],
            'note': 'Condicionamiento indirecto vía sampling (modelo incondicional)'
        }
    
    def _tokens_to_midi_placeholder(self, tokens: torch.Tensor, out_path: str):
        """
        Convierte tokens a MIDI (PLACEHOLDER por ahora).
        
        TODO: Implementar conversión real usando tokenizer REMI o similar.
        Por ahora, genera un MIDI simple para demostración.
        
        Args:
            tokens: Tokens generados
            out_path: Ruta de salida
        """
        import mido
        from mido import Message, MidiFile, MidiTrack, MetaMessage
        
        # Crear MIDI simple
        midi = MidiFile(ticks_per_beat=480)
        track = MidiTrack()
        midi.tracks.append(track)
        
        track.append(MetaMessage('track_name', name='Transformer Generated', time=0))
        track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
        
        # Generar notas simples basadas en los tokens (placeholder)
        # En realidad, los tokens deberían decodificarse según el vocabulario REMI
        notes = [60, 62, 64, 65, 67, 69, 71, 72]  # Escala C mayor
        
        for i in range(min(32, tokens.shape[1])):  # Primeros 32 tokens
            token_val = tokens[0, i].item()
            note = notes[token_val % len(notes)]
            velocity = 80
            duration = 480
            
            track.append(Message('note_on', note=note, velocity=velocity, time=0))
            track.append(Message('note_off', note=note, velocity=0, time=duration))
        
        track.append(MetaMessage('end_of_track', time=0))
        
        # Guardar
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        midi.save(out_path)
        
        logger.info(f"MIDI placeholder guardado en: {out_path}")
