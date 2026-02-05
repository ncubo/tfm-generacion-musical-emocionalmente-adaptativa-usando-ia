"""
Generador de música con Music Transformer preentrenado (PyTorch).

Este módulo implementa la generación de música simbólica usando un modelo
Transformer, con condicionamiento indirecto mediante parámetros de sampling.

IMPORTANTE SOBRE EL MODELO PREENTRENADO:
    Por simplicidad y compatibilidad, este módulo implementa un transformer
    ligero que puede ser:
    1. Cargado desde checkpoint preentrenado (si disponible)
    2. Usado con pesos aleatorios iniciales (para demostración)
    
    Para producción, se recomienda usar un checkpoint real entrenado en
    datasets como MAESTRO, Lakh MIDI, o similar.

Fuentes recomendadas de checkpoints:
    - Hugging Face: https://huggingface.co/models?filter=music
    - Music Transformer (original): https://github.com/jason9693/MusicTransformer-pytorch
    - Pop Music Transformer: https://github.com/YatingMusic/remi
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import math

from .sampling import (
    get_temperature_from_arousal,
    get_top_p_from_arousal,
    get_top_k_from_arousal,
    sample_sequence
)
from .io import MIDITokenConverter, create_simple_primer

logger = logging.getLogger(__name__)


class SimpleMusicTransformer(nn.Module):
    """
    Modelo Transformer simplificado para generación musical.
    
    Basado en el arquitectura del Music Transformer original pero simplificada.
    En producción, este modelo debería cargarse desde un checkpoint preentrenado.
    """
    
    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        """
        Inicializa el modelo Transformer.
        
        Args:
            vocab_size: Tamaño del vocabulario de tokens
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
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer decoder (autoregresivo)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Proyección a vocabulario
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Inicializar pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa los pesos del modelo."""
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_proj.bias.data.zero_()
        self.output_proj.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            src: Secuencia de entrada [batch_size, seq_len]
            memory: Memoria del encoder (None para generación autoregresiva)
        
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Embedding + positional encoding
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Máscara causal (atención solo a tokens anteriores)
        seq_len = src.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # Si no hay memoria (generación desde cero), usar la misma secuencia
        if memory is None:
            memory = x
        
        # Transformer decoder
        output = self.transformer(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask
        )
        
        # Proyección a vocabulario
        logits = self.output_proj(output)
        
        return logits


class PositionalEncoding(nn.Module):
    """
    Positional encoding para Transformer.
    
    Añade información posicional a los embeddings usando funciones seno/coseno.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Crear matriz de positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Añade positional encoding a los embeddings.
        
        Args:
            x: Embeddings [batch_size, seq_len, d_model]
        
        Returns:
            Embeddings con positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerMusicGenerator:
    """
    Generador de MIDI usando Music Transformer preentrenado.
    
    Este generador carga un modelo Transformer una sola vez y genera secuencias
    MIDI condicionadas indirectamente por parámetros emocionales (V,A).
    
    Atributos:
        model: Modelo Transformer
        tokenizer: Conversor tokens ↔ MIDI
        device: Device de PyTorch ('cpu' o 'cuda')
    
    Ejemplo:
        >>> generator = TransformerMusicGenerator(checkpoint_path="models/transformer.pt")
        >>> result = generator.generate(v=0.7, a=0.6, out_path="happy.mid")
        >>> print(result["midi_path"])
        'happy.mid'
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        vocab_size: int = 512
    ):
        """
        Inicializa el generador Transformer.
        
        Args:
            checkpoint_path: Path al checkpoint del modelo (None = usar pesos aleatorios)
            device: Device ('cpu', 'cuda', o None = auto-detectar)
            vocab_size: Tamaño del vocabulario
        
        Raises:
            FileNotFoundError: Si checkpoint_path no existe (cuando se especifica)
        """
        # Detectar device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Inicializando TransformerMusicGenerator en {self.device}")
        
        # Inicializar tokenizer
        self.tokenizer = MIDITokenConverter(vocab_size=vocab_size)
        actual_vocab_size = self.tokenizer.vocab_size
        
        # Inicializar modelo
        self.model = SimpleMusicTransformer(
            vocab_size=actual_vocab_size,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            max_seq_len=2048
        ).to(self.device)
        
        # Cargar checkpoint si se especifica
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                logger.warning(
                    f"Checkpoint no encontrado: {checkpoint_path}\n"
                    f"Usando pesos aleatorios (solo para demostración)."
                )
            else:
                try:
                    state_dict = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    logger.info(f"Checkpoint cargado: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Error cargando checkpoint: {e}")
                    logger.warning("Usando pesos aleatorios")
        else:
            logger.warning(
                "No se especificó checkpoint. Usando pesos aleatorios.\n"
                "Para producción, entrena o descarga un checkpoint preentrenado."
            )
        
        self.model.eval()
        logger.info("TransformerMusicGenerator inicializado")
    
    def generate(
        self,
        v: float,
        a: float,
        out_path: str,
        max_length: int = 512,
        seed: Optional[int] = None,
        use_primer: bool = False,
        music_params: Optional[Dict] = None
    ) -> Dict:
        """
        Genera un archivo MIDI condicionado por Valence-Arousal.
        
        FLUJO DE CONDICIONAMIENTO INDIRECTO:
        1. Arousal → temperature (alta A → mayor diversidad)
        2. Arousal → top-p (alta A → nucleus sampling más permisivo)
        3. Arousal → top-k (alta A → más opciones)
        4. (Opcional) (V, A) → music_params → primer MIDI
        
        Args:
            v: Valence en [-1, 1]
            a: Arousal en [-1, 1]
            out_path: Path donde guardar el archivo MIDI
            max_length: Longitud máxima de la secuencia (en tokens)
            seed: Semilla aleatoria para reproducibilidad
            use_primer: Si True, genera un primer desde music_params
            music_params: Parámetros musicales del mapeo VA (opcional)
        
        Returns:
            Dict con:
                - midi_path: Path al archivo MIDI generado
                - generation_params: Parámetros usados en la generación
                - num_tokens: Número de tokens generados
        
        Raises:
            ValueError: Si v o a están fuera de rango [-1, 1]
        """
        if not (-1 <= v <= 1) or not (-1 <= a <= 1):
            raise ValueError(f"V y A deben estar en [-1, 1]. Got V={v}, A={a}")
        
        # Configurar seed para reproducibilidad
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # 1. Derivar parámetros de sampling desde (V, A)
        temperature = get_temperature_from_arousal(a)
        top_p = get_top_p_from_arousal(a)
        top_k = get_top_k_from_arousal(a)
        
        logger.info(f"Generando con V={v:.2f}, A={a:.2f}")
        logger.info(f"  Temperature: {temperature:.3f}")
        logger.info(f"  Top-p: {top_p:.3f}")
        logger.info(f"  Top-k: {top_k}")
        logger.info(f"  Max length: {max_length} tokens")
        
        # 2. Crear tokens iniciales (primer o BOS)
        if use_primer and music_params:
            # Crear MIDI primer simple
            primer_path = create_simple_primer(
                mode=music_params.get('mode', 'major'),
                pitch_low=music_params.get('pitch_low', 60),
                num_notes=4
            )
            # Convertir a tokens
            try:
                start_tokens = self.tokenizer.midi_to_tokens(primer_path)
                logger.info(f"Usando primer con {len(start_tokens)} tokens")
            except Exception as e:
                logger.warning(f"Error creando primer: {e}. Usando BOS token")
                start_tokens = [self.tokenizer.pad_token]
        else:
            # Comenzar con token de padding/BOS
            start_tokens = [self.tokenizer.pad_token]
        
        # Convertir a tensor
        start_tokens_tensor = torch.tensor([start_tokens], dtype=torch.long)
        
        # 3. Generar secuencia con sampling
        try:
            generated_tokens = sample_sequence(
                model=self.model,
                start_tokens=start_tokens_tensor,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token=self.tokenizer.eos_token,
                device=self.device
            )
            
            # Convertir a lista
            generated_tokens_list = generated_tokens[0].cpu().tolist()
            
        except Exception as e:
            logger.error(f"Error durante generación: {e}")
            raise
        
        # 4. Convertir tokens a MIDI
        try:
            tempo = music_params.get('tempo_bpm', 120) if music_params else 120
            midi_path = self.tokenizer.tokens_to_midi(
                tokens=generated_tokens_list,
                out_path=out_path,
                tempo=tempo
            )
        except Exception as e:
            logger.error(f"Error convirtiendo tokens a MIDI: {e}")
            raise
        
        # 5. Retornar metadata
        return {
            "midi_path": midi_path,
            "generation_params": {
                "model": "music_transformer",
                "valence": v,
                "arousal": a,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_length": max_length,
                "seed": seed,
                "used_primer": use_primer and music_params is not None
            },
            "num_tokens": len(generated_tokens_list)
        }
