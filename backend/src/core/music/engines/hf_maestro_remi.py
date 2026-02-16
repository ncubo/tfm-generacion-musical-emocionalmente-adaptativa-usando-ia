"""
Motor de generación musical basado en Hugging Face y MidiTok REMI.

Este módulo implementa un generador de música MIDI usando el modelo preentrenado
Maestro-REMI-bpe20k de Hugging Face con tokenización REMI de MidiTok.

Modelo usado: https://huggingface.co/NathanFradet/Maestro-REMI-bpe20k

Características:
- Lazy loading del modelo y tokenizador (solo se carga una vez por proceso)
- Condicionamiento indirecto vía parámetros musicales derivados de VA
- Estrategia de primer: genera baseline de 2 compases como entrada
- Postprocesado para ajustar el MIDI generado a los parámetros objetivo
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import tempfile

# Lazy imports (solo se cargan cuando se necesitan)
_transformers = None
_miditok = None
_torch = None
_symusic = None

# Singletons para el modelo y tokenizador
_model = None
_tokenizer = None
_device = None

logger = logging.getLogger(__name__)


def _lazy_import_deps():
    """Importa dependencias de manera lazy para evitar overhead al inicio."""
    global _transformers, _miditok, _torch, _symusic
    
    if _transformers is None:
        try:
            import transformers
            _transformers = transformers
        except ImportError as e:
            raise RuntimeError(
                "transformers no está instalado. "
                "Ejecuta: pip install transformers>=4.40"
            ) from e
    
    if _miditok is None:
        try:
            import miditok
            _miditok = miditok
        except ImportError as e:
            raise RuntimeError(
                "miditok no está instalado. "
                "Ejecuta: pip install miditok>=3.0"
            ) from e
    
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError as e:
            raise RuntimeError(
                "torch no está instalado. "
                "Ejecuta: pip install torch"
            ) from e
    
    if _symusic is None:
        try:
            import symusic
            _symusic = symusic
        except ImportError as e:
            raise RuntimeError(
                "symusic no está instalado (requerido por miditok 3.0+). "
                "Ejecuta: pip install miditok>=3.0 (incluye symusic)"
            ) from e
    
    return _transformers, _miditok, _torch, _symusic


def _get_device():
    """Obtiene el device (CUDA si está disponible, sino CPU)."""
    global _device
    
    if _device is None:
        _, _, torch, _ = _lazy_import_deps()
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"HF Maestro REMI engine usando device: {_device}")
    
    return _device


def _load_model_and_tokenizer(model_id: str = "Natooz/Maestro-REMI-bpe20k") -> Tuple[Any, Any]:
    """
    Carga el modelo y tokenizador de manera lazy (singleton).
    
    Args:
        model_id: ID del modelo en Hugging Face Hub
        
    Returns:
        Tupla (model, tokenizer)
    """
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        transformers, miditok, torch, _ = _lazy_import_deps()
        device = _get_device()
        
        logger.info(f"Cargando modelo HF: {model_id}...")
        
        try:
            # Cargar tokenizador REMI desde HF Hub
            _tokenizer = miditok.REMI.from_pretrained(model_id)
            logger.info(f"Tokenizador REMI cargado: vocab_size={len(_tokenizer)}")
            
            # Cargar modelo de lenguaje causal
            _model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16
            )
            _model.to(device)
            _model.eval()
            
            logger.info(f"Modelo cargado exitosamente en {device}")
            
        except Exception as e:
            logger.error(f"Error al cargar modelo/tokenizador: {e}", exc_info=True)
            raise RuntimeError(
                f"No se pudo cargar el modelo {model_id}. "
                "Verifica tu conexión a internet y que transformers/miditok estén instalados."
            ) from e
    
    return _model, _tokenizer


def _derive_sampling_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deriva configuración de sampling del modelo basada en parámetros musicales.
    
    Mapeo indirecto:
    - Mayor rhythm_complexity/density/tempo -> mayor temperature, top_p, más tokens
    - Menor valores -> menor temperature/top_p, menos tokens
    
    Args:
        params: Parámetros musicales (tempo_bpm, density, rhythm_complexity, etc.)
        
    Returns:
        Dict con parámetros de sampling (temperature, top_p, max_new_tokens)
    """
    # Normalizar indicadores de "energía musical" (0-1)
    density = params.get('density', 0.5)
    rhythm_complexity = params.get('rhythm_complexity', 0.5)
    tempo_bpm = params.get('tempo_bpm', 120)
    tempo_normalized = min(1.0, max(0.0, (tempo_bpm - 60) / 120))  # 60-180 bpm -> 0-1
    
    # Promedio ponderado de energía
    energy = 0.4 * density + 0.4 * rhythm_complexity + 0.2 * tempo_normalized
    
    # Mapear energía a parámetros de sampling
    # Temperature: 0.7 (baja energía) -> 1.1 (alta energía)
    temperature = 0.7 + energy * 0.4
    
    # Top-p: 0.85 (baja energía) -> 0.95 (alta energía)
    top_p = 0.85 + energy * 0.1
    
    # Max tokens: aproximadamente 40 tokens por compás (valor ajustado para REMI)
    # Más energía -> más notas -> más tokens por compás
    tokens_per_bar = int(40 * (0.8 + energy * 0.4))
    
    return {
        'temperature': temperature,
        'top_p': top_p,
        'tokens_per_bar': tokens_per_bar,
        'do_sample': True
    }


def _generate_primer(
    params: Dict[str, Any],
    length_bars: int = 2,
    seed: Optional[int] = None
) -> Path:
    """
    Genera un MIDI de 2 compases usando baseline como primer.
    
    Args:
        params: Parámetros musicales
        length_bars: Número de compases (típicamente 2)
        seed: Semilla aleatoria opcional
        
    Returns:
        Path al archivo MIDI temporal generado
    """
    from .baseline import generate_midi_baseline
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
        primer_path = Path(tmp.name)
    
    # Garantizar density mínima para el primer (evitar MIDIs vacíos)
    # Esto no afecta la generación del transformer, solo el prompt inicial
    primer_params = params.copy()
    original_density = primer_params.get('density', 0.5)
    
    # Si density es muy baja, aumentarla solo para el primer
    if original_density < 0.4:
        primer_params['density'] = 0.5  # Mínimo que garantiza ~2 notas por compás
        logger.debug(f"Ajustando density del primer: {original_density:.2f} → 0.50 (evitar MIDI vacío)")
    
    # Generar baseline corto como primer
    generate_midi_baseline(
        params=primer_params,
        out_path=str(primer_path),
        length_bars=length_bars,
        seed=seed
    )
    
    # Diagnóstico: verificar contenido del primer
    import mido
    primer_mid = mido.MidiFile(str(primer_path))
    note_count = sum(1 for track in primer_mid.tracks 
                     for msg in track 
                     if msg.type == 'note_on' and msg.velocity > 0)
    logger.debug(f"Primer generado: {primer_path} con {note_count} notas")
    
    if note_count == 0:
        logger.warning(f"⚠️  ADVERTENCIA: Primer MIDI vacío (0 notas). Params: {params}")
    
    return primer_path


def _postprocess_score(score, params: Dict[str, Any]):
    """
    Postprocesa el Score generado para ajustarse a parámetros objetivo.
    
    Operaciones:
    - Clamp pitches al rango [pitch_low, pitch_high]
    - Ajusta velocidades hacia velocity_mean
    - Mantiene tempo_bpm como metadata (difícil de forzar en generación)
    
    Args:
        score: symusic.Score object (retornado por tokens_to_midi en miditok 3.0+)
        params: Parámetros musicales objetivo
        
    Modifica score in-place.
    """
    pitch_low = params.get('pitch_low', 48)
    pitch_high = params.get('pitch_high', 84)
    velocity_mean = params.get('velocity_mean', 80)
    
    # Iterar por todas las pistas (symusic Score)
    for track in score.tracks:
        for note in track.notes:
            # Ajustar pitch (MIDI note number)
            note.pitch = max(pitch_low, min(pitch_high, note.pitch))
            
            # Ajustar velocity hacia la media objetivo (promedio ponderado)
            # 70% hacia objetivo, 30% original
            target = velocity_mean
            current = note.velocity
            new_vel = int(0.7 * target + 0.3 * current)
            note.velocity = max(1, min(127, new_vel))
    
    logger.debug(f"Score postprocesado: pitch=[{pitch_low}, {pitch_high}], vel_mean≈{velocity_mean}")


def generate_midi_hf_maestro_remi(
    params: Dict[str, Any],
    out_path: str,
    length_bars: int = 8,
    seed: Optional[int] = None,
    max_new_tokens: Optional[int] = None
) -> str:
    """
    Genera un archivo MIDI usando el modelo Maestro-REMI preentrenado.
    
    Estrategia:
    1. Deriva config de sampling desde params
    2. Genera primer de 2 compases con baseline
    3. Convierte primer a tokens REMI
    4. Genera continuación con modelo HF
    5. Decodifica tokens a Score
    6. Postprocesa Score (pitch/velocity)
    7. Guarda MIDI
    
    Args:
        params: Parámetros musicales (tempo_bpm, mode, density, etc.)
        out_path: Path donde guardar el MIDI generado
        length_bars: Número de compases objetivo (default: 8)
        seed: Semilla aleatoria opcional para reproducibilidad
        max_new_tokens: Número fijo de tokens a generar (si None, se calcula automático)
                       Útil para benchmarks donde se requiere longitud constante
        
    Returns:
        Path al archivo MIDI generado (out_path)
        
    Raises:
        RuntimeError: Si falla carga del modelo o generación
    """
    try:
        # Lazy imports
        _, _, torch, symusic = _lazy_import_deps()
        
        # Cargar modelo y tokenizador (lazy, singleton)
        model, tokenizer = _load_model_and_tokenizer()
        device = _get_device()
        
        # Configurar semilla si se proporciona
        if seed is not None:
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)
            logger.info(f"Semilla configurada: {seed}")
        
        # Diagnóstico: mostrar parámetros musicales recibidos
        logger.debug(f"Parámetros musicales recibidos:")
        logger.debug(f"  - tempo_bpm: {params.get('tempo_bpm', 'N/A')}")
        logger.debug(f"  - density: {params.get('density', 'N/A')}")
        logger.debug(f"  - rhythm_complexity: {params.get('rhythm_complexity', 'N/A')}")
        logger.debug(f"  - velocity_mean: {params.get('velocity_mean', 'N/A')}")
        logger.debug(f"  - pitch_low/high: {params.get('pitch_low', 'N/A')}/{params.get('pitch_high', 'N/A')}")
        
        # 1. Derivar configuración de sampling
        sampling_config = _derive_sampling_config(params)
        logger.info(f"Sampling config: temp={sampling_config['temperature']:.2f}, "
                   f"top_p={sampling_config['top_p']:.2f}")
        
        # 2. Generar primer (2 compases baseline)
        primer_path = _generate_primer(params, length_bars=2, seed=seed)
        
        try:
            # 3. Convertir primer a tokens usando la API oficial de HF
            # Cargar MIDI como Score de symusic
            primer_score = symusic.Score(str(primer_path))
            
            # Tokenizar (retorna lista de TokSequence, uno por pista)
            primer_tokens = tokenizer(primer_score)
            
            # Tomar primera pista si es una lista
            if isinstance(primer_tokens, list):
                primer_tokens = primer_tokens[0]
            
            # Extraer IDs
            input_token_ids = primer_tokens.ids
            logger.info(f"Primer convertido a {len(input_token_ids)} tokens")
            
            # Diagnóstico: verificar tokens vacíos
            if len(input_token_ids) == 0:
                logger.error(f"   ERROR: Tokenización resultó en 0 tokens!")
                logger.error(f"   Params musicales: {params}")
                logger.error(f"   Primer path: {primer_path}")
                raise RuntimeError(
                    f"El primer MIDI no pudo ser tokenizado (0 tokens). "
                    f"Probablemente el MIDI generado está vacío o es inválido para REMI."
                )
            
            # 4. Calcular tokens a generar
            # Si max_new_tokens se especifica explícitamente, usarlo (para benchmarks)
            # Si no, calcular dinámicamente basado en compases restantes
            if max_new_tokens is not None:
                tokens_to_generate = max_new_tokens
                logger.info(f"Usando max_new_tokens fijo: {tokens_to_generate} (benchmark mode)")
            else:
                # Restar los compases del primer del total deseado
                remaining_bars = max(1, length_bars - 2)
                tokens_to_generate = remaining_bars * sampling_config['tokens_per_bar']
                logger.info(f"Calculando tokens dinámicamente para {remaining_bars} compases: {tokens_to_generate}")
            
            logger.info(f"Generando {tokens_to_generate} nuevos tokens...")
            
            # 5. Generar continuación con el modelo (API oficial de HF)
            # Convertir lista de IDs a tensor (batch_size=1)
            input_tensor = torch.tensor([input_token_ids], dtype=torch.long, device=device)
            
            generated_token_ids = model.generate(
                input_tensor,
                max_new_tokens=tokens_to_generate,
                temperature=sampling_config['temperature'],
                top_p=sampling_config['top_p'],
                do_sample=sampling_config['do_sample'],
            )
            
            logger.info(f"Generación completa: {generated_token_ids.shape[1]} tokens totales")
            
            # 6. Decodificar tokens a Score
            # Pasar el tensor directamente al tokenizer (como en el ejemplo oficial)
            # miditok maneja internamente la decodificación BPE
            generated_score = tokenizer(generated_token_ids)
            
            # 7. Postprocesar Score
            _postprocess_score(generated_score, params)
            
            # 8. Guardar MIDI
            out_path_obj = Path(out_path)
            out_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # symusic Score tiene método dump_midi
            generated_score.dump_midi(str(out_path_obj))
            
            logger.info(f"MIDI generado exitosamente: {out_path}")
            
            return str(out_path_obj)
            
        finally:
            # Limpiar archivo temporal del primer
            if primer_path.exists():
                primer_path.unlink()
                logger.debug(f"Primer temporal eliminado: {primer_path}")
    
    except Exception as e:
        logger.error(f"Error en generate_midi_hf_maestro_remi: {e}", exc_info=True)
        raise RuntimeError(f"Error al generar MIDI con HF Maestro REMI: {e}") from e
