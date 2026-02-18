#!/usr/bin/env python3
"""
Script para construir dataset de fine-tuning para Maestro-REMI-bpe20k.

Procesa archivos MIDI del subset Lakh Piano con labels VA heurísticas,
tokeniza usando REMI, añade conditioning tokens de VA discretos, y construye
un dataset HuggingFace compatible con transformers.Trainer.

Pipeline:
1. Cargar labels VA desde CSV
2. Para cada MIDI: tokenizar con REMI
3. Ventaneo de secuencias largas (window_size + stride)
4. Añadir conditioning tokens de VA al inicio de cada ventana
5. Split train/val por archivo (evitar data leakage)
6. Guardar dataset en formato HuggingFace

Uso:
    python scripts/build_finetune_dataset.py
    python scripts/build_finetune_dataset.py --max_files 100 --window_size 512
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

# Añadir src al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / "src"))

from core.music.tokenization.remi import load_remi_tokenizer, midi_to_remi_tokens

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_labels(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Carga labels VA desde CSV.
    
    Args:
        csv_path: Path al CSV con columnas: filename, valence, arousal
        
    Returns:
        Dict[filename] = {"valence": float, "arousal": float}
    """
    logger.info(f"Cargando labels desde: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV de labels no encontrado: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Verificar columnas requeridas
    required_cols = ['filename', 'valence', 'arousal']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en CSV: {missing}")
    
    # Construir diccionario
    labels_dict = {}
    for _, row in df.iterrows():
        filename = row['filename']
        valence = float(row['valence'])
        arousal = float(row['arousal'])
        labels_dict[filename] = {'valence': valence, 'arousal': arousal}
    
    logger.info(f"Cargadas {len(labels_dict)} labels VA")
    return labels_dict


def discretize_va(valence: float, arousal: float, bins: int = 9) -> Tuple[int, int]:
    """
    Discretiza valores continuos de VA a bins discretos.
    
    Convierte valores en [-1, 1] a bins enteros en rango simétrico.
    - bins=9 -> rango [-4, +4]
    - bins=7 -> rango [-3, +3]
    
    Args:
        valence: Valor continuo en [-1, 1]
        arousal: Valor continuo en [-1, 1]
        bins: Número de bins (debe ser impar para tener bin central 0)
        
    Returns:
        Tupla (valence_bin, arousal_bin) como enteros
    """
    # Clamp valores a [-1, 1]
    valence = np.clip(valence, -1, 1)
    arousal = np.clip(arousal, -1, 1)
    
    # Calcular rango de bins: [-half, +half]
    half = (bins - 1) // 2
    
    # Normalizar de [-1, 1] a [0, bins-1] y luego centrar
    v_bin = int(np.round((valence + 1) / 2 * (bins - 1))) - half
    a_bin = int(np.round((arousal + 1) / 2 * (bins - 1))) - half
    
    # Asegurar que esté en rango
    v_bin = np.clip(v_bin, -half, half)
    a_bin = np.clip(a_bin, -half, half)
    
    return int(v_bin), int(a_bin)


def build_conditioning_tokens(v_bin: int, a_bin: int) -> List[str]:
    """
    Construye tokens de conditioning desde bins discretos.
    
    Formato:
        ["[VAL=-2]", "[ARO=+3]"]
    
    Args:
        v_bin: Bin discreto de valence (ej: -4 a +4)
        a_bin: Bin discreto de arousal (ej: -4 a +4)
        
    Returns:
        Lista de 2 strings representando conditioning tokens
    """
    # Formatear con signo explícito para positivos
    v_str = f"{v_bin:+d}"  # "+3" o "-2"
    a_str = f"{a_bin:+d}"
    
    return [f"[VAL={v_str}]", f"[ARO={a_str}]"]


def generate_all_conditioning_tokens(bins: int = 9) -> List[str]:
    """
    Genera lista completa de todos los conditioning tokens posibles.
    
    Args:
        bins: Número de bins para VA
        
    Returns:
        Lista de todos los tokens únicos (ej: ["[VAL=-4]", ..., "[ARO=+4]"])
    """
    half = (bins - 1) // 2
    tokens = []
    
    # Todos los bins de valence
    for v in range(-half, half + 1):
        tokens.append(f"[VAL={v:+d}]")
    
    # Todos los bins de arousal
    for a in range(-half, half + 1):
        tokens.append(f"[ARO={a:+d}]")
    
    return tokens


def create_windows(
    token_ids: List[int],
    window_size: int,
    stride: int,
    min_window_tokens: int
) -> List[List[int]]:
    """
    Divide secuencia de tokens en ventanas con overlap.
    
    Args:
        token_ids: Lista de IDs de tokens
        window_size: Tamaño de cada ventana
        stride: Desplazamiento entre ventanas
        min_window_tokens: Mínimo de tokens para considerar una ventana válida
        
    Returns:
        Lista de ventanas (cada ventana es una lista de IDs)
    """
    windows = []
    
    for i in range(0, len(token_ids), stride):
        window = token_ids[i:i + window_size]
        
        # Descartar ventanas muy cortas
        if len(window) >= min_window_tokens:
            windows.append(window)
        
        # Si esta ventana alcanza el final, parar
        if i + window_size >= len(token_ids):
            break
    
    return windows


def normalize_midi_pitches(midi_path: Path, min_pitch: int = 21, max_pitch: int = 108) -> Optional[Path]:
    """
    Normaliza pitches de un MIDI al rango válido del tokenizador.
    
    Crea una copia temporal del MIDI con pitches clampeados.
    
    Args:
        midi_path: Path al MIDI original
        min_pitch: Pitch mínimo válido (default: 21, A0)
        max_pitch: Pitch máximo válido (default: 108, C8)
        
    Returns:
        Path al MIDI normalizado (temporal) o None si falla
    """
    try:
        import symusic
        import tempfile
        
        # Cargar MIDI
        score = symusic.Score(str(midi_path))
        
        # Normalizar pitches en todas las pistas
        modified = False
        for track in score.tracks:
            for note in track.notes:
                if note.pitch < min_pitch or note.pitch > max_pitch:
                    note.pitch = max(min_pitch, min(max_pitch, note.pitch))
                    modified = True
        
        # Si no se modificó nada, devolver el original
        if not modified:
            return midi_path
        
        # Guardar versión normalizada en archivo temporal
        temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        
        score.dump_midi(str(temp_path))
        logger.debug(f"MIDI normalizado: {midi_path.name} (pitches clampeados a [{min_pitch}, {max_pitch}])")
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Error normalizando pitches en {midi_path.name}: {e}")
        return None


def process_midi_file(
    midi_path: Path,
    labels: Dict[str, float],
    tokenizer,
    conditioning_token_ids: Dict[str, int],
    window_size: int,
    stride: int,
    min_window_tokens: int,
    bins: int = 9
) -> Tuple[List[Dict], Optional[str]]:
    """
    Procesa un archivo MIDI completo.
    
    Pipeline:
    1. Normalizar pitches MIDI
    2. Tokenizar MIDI
    3. Crear ventanas
    4. Añadir conditioning tokens (IDs reales) a cada ventana
    5. Construir ejemplos de dataset
    
    Args:
        midi_path: Path al archivo MIDI
        labels: Dict con valence/arousal
        tokenizer: Tokenizador REMI (con conditioning tokens añadidos)
        conditioning_token_ids: Dict mapeando tokens a IDs
        window_size: Tamaño de ventana
        stride: Stride para ventaneo
        min_window_tokens: Mínimo de tokens por ventana
        bins: Número de bins para discretización
        
    Returns:
        Tupla (lista de ejemplos, razón de descarte o None)
    """
    # Normalizar pitches al rango válido
    normalized_path = normalize_midi_pitches(midi_path)
    if normalized_path is None:
        return [], "pitch_normalization_failed"
    
    # Tokenizar MIDI
    try:
        token_ids = midi_to_remi_tokens(str(normalized_path), tokenizer)
    except KeyError as e:
        # Error de token fuera de vocabulario (ej: Pitch_109)
        logger.warning(f"Token OOV en {midi_path.name}: {e}")
        if normalized_path != midi_path and normalized_path.exists():
            normalized_path.unlink()
        return [], "oov_token"
    except Exception as e:
        logger.warning(f"Error tokenizando {midi_path.name}: {e}")
        if normalized_path != midi_path and normalized_path.exists():
            normalized_path.unlink()
        return [], "tokenization_error"
    finally:
        # Limpiar archivo temporal si se creó
        if normalized_path != midi_path and normalized_path.exists():
            normalized_path.unlink()
    
    if len(token_ids) == 0:
        logger.warning(f"MIDI sin tokens: {midi_path.name}")
        return [], "no_tokens"
    
    # Discretizar VA
    v_bin, a_bin = discretize_va(labels['valence'], labels['arousal'], bins)
    
    # Construir conditioning tokens strings
    cond_tokens = build_conditioning_tokens(v_bin, a_bin)
    
    # Obtener IDs reales de conditioning tokens
    cond_ids = [
        conditioning_token_ids[cond_tokens[0]],  # [VAL=...]
        conditioning_token_ids[cond_tokens[1]]   # [ARO=...]
    ]
    
    # Crear ventanas
    windows = create_windows(token_ids, window_size, stride, min_window_tokens)
    
    # Construir ejemplos
    examples = []
    for window_idx, window in enumerate(windows):
        # Concatenar conditioning + window
        # Los conditioning tokens van AL INICIO
        final_ids = cond_ids + window
        
        # Truncar si excede window_size + len(cond_ids)
        max_length = window_size + len(cond_ids)
        final_ids = final_ids[:max_length]
        
        # Construir ejemplo
        example = {
            'input_ids': final_ids,
            'attention_mask': [1] * len(final_ids),
            'labels': final_ids.copy(),  # Para LM, labels = input_ids
            'valence': labels['valence'],
            'arousal': labels['arousal'],
            'valence_bin': v_bin,
            'arousal_bin': a_bin,
            'source_file': midi_path.name,
            'window_idx': window_idx,
            'conditioning_tokens': cond_tokens  # Guardar para referencia
        }
        
        examples.append(example)
    
    return examples, None


def split_by_files(
    examples: List[Dict],
    train_split: float,
    seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Divide ejemplos en train/val por archivo (evitar data leakage).
    
    Todos los segmentos de un mismo archivo van al mismo split.
    
    Args:
        examples: Lista de ejemplos
        train_split: Fracción para train (ej: 0.9)
        seed: Semilla aleatoria
        
    Returns:
        Tupla (train_examples, val_examples)
    """
    np.random.seed(seed)
    
    # Agrupar por archivo
    files_examples = {}
    for ex in examples:
        filename = ex['source_file']
        if filename not in files_examples:
            files_examples[filename] = []
        files_examples[filename].append(ex)
    
    # Obtener lista de archivos únicos
    files = list(files_examples.keys())
    np.random.shuffle(files)
    
    # Split
    n_train = int(len(files) * train_split)
    train_files = set(files[:n_train])
    val_files = set(files[n_train:])
    
    # Asignar ejemplos
    train_examples = []
    val_examples = []
    
    for filename in files_examples:
        if filename in train_files:
            train_examples.extend(files_examples[filename])
        else:
            val_examples.extend(files_examples[filename])
    
    logger.info(f"Split: {len(train_files)} archivos train, {len(val_files)} archivos val")
    logger.info(f"Ejemplos: {len(train_examples)} train, {len(val_examples)} val")
    
    return train_examples, val_examples


def save_dataset(
    train_examples: List[Dict],
    val_examples: List[Dict],
    output_dir: Path,
    config: Dict
):
    """
    Guarda dataset en formato HuggingFace.
    
    Args:
        train_examples: Ejemplos de entrenamiento
        val_examples: Ejemplos de validación
        output_dir: Directorio de salida
        config: Configuración del dataset (metadata)
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise RuntimeError(
            "datasets no está instalado. "
            "Ejecuta: pip install datasets"
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear datasets HF
    logger.info("Creando datasets HuggingFace...")
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    # Guardar
    train_path = output_dir / "train"
    val_path = output_dir / "val"
    
    logger.info(f"Guardando train dataset en: {train_path}")
    train_dataset.save_to_disk(str(train_path))
    
    logger.info(f"Guardando val dataset en: {val_path}")
    val_dataset.save_to_disk(str(val_path))
    
    # Guardar metadata
    metadata_path = output_dir / "dataset_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Metadata guardada en: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Construir dataset de fine-tuning para Maestro-REMI'
    )
    
    # Paths
    parser.add_argument(
        '--midi_dir',
        default='data/lakh_piano_clean',
        help='Directorio con archivos MIDI'
    )
    parser.add_argument(
        '--labels_csv',
        default='data/lakh_piano_metadata/lakh_piano_subset_labeled.csv',
        help='CSV con labels VA'
    )
    parser.add_argument(
        '--output_dir',
        default='data/finetune_dataset',
        help='Directorio de salida para dataset'
    )
    
    # Modelo
    parser.add_argument(
        '--model_name',
        default='Natooz/Maestro-REMI-bpe20k',
        help='Nombre del modelo HF'
    )
    
    # Parámetros de ventaneo
    parser.add_argument('--window_size', type=int, default=1024,
                       help='Tamaño de ventana en tokens')
    parser.add_argument('--stride', type=int, default=512,
                       help='Stride para ventanas overlap')
    parser.add_argument('--min_window_tokens', type=int, default=256,
                       help='Mínimo de tokens por ventana')
    
    # Discretización
    parser.add_argument('--bins', type=int, default=9,
                       help='Número de bins para VA (debe ser impar)')
    
    # Split y procesamiento
    parser.add_argument('--train_split', type=float, default=0.9,
                       help='Fracción para train')
    parser.add_argument('--seed', type=int, default=42,
                       help='Semilla aleatoria')
    parser.add_argument('--max_files', type=int, default=5000,
                       help='Máximo archivos MIDI (5000 archivos ≈ 20,000+ ejemplos con windowing)')
    
    args = parser.parse_args()
    
    # Paths absolutos
    midi_dir = backend_dir / args.midi_dir
    labels_csv = backend_dir / args.labels_csv
    output_dir = backend_dir / args.output_dir
    
    logger.info("=" * 80)
    logger.info("CONSTRUCCIÓN DE DATASET FINE-TUNING MAESTRO-REMI")
    logger.info("=" * 80)
    logger.info(f"MIDI dir: {midi_dir}")
    logger.info(f"Labels CSV: {labels_csv}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Window size: {args.window_size}, stride: {args.stride}")
    logger.info(f"Bins VA: {args.bins}")
    logger.info("")
    
    # 1. Cargar tokenizador
    logger.info("1. Cargando tokenizador REMI...")
    tokenizer = load_remi_tokenizer(args.model_name)
    original_vocab_size = len(tokenizer)
    
    # 1b. Añadir conditioning tokens al tokenizador
    logger.info("1b. Añadiendo conditioning tokens al tokenizador...")
    conditioning_tokens = generate_all_conditioning_tokens(args.bins)
    
    # miditok usa add_to_vocab (diferente API que transformers)
    # Los tokens se añaden al final del vocabulario con IDs consecutivos
    for token in conditioning_tokens:
        tokenizer.add_to_vocab([token])
    
    new_vocab_size = len(tokenizer)
    num_added = new_vocab_size - original_vocab_size
    logger.info(f"Añadidos {num_added} conditioning tokens al vocabulario")
    logger.info(f"Vocab size: {original_vocab_size} -> {new_vocab_size}")
    
    # Crear mapeo de tokens a IDs
    # Los tokens añadidos tienen IDs: original_vocab_size, original_vocab_size+1, ...
    conditioning_token_ids = {}
    for i, token in enumerate(conditioning_tokens):
        token_id = original_vocab_size + i
        conditioning_token_ids[token] = token_id
    
    logger.info(f"Token IDs asignados:")
    logger.info(f"  {conditioning_tokens[0]} -> {conditioning_token_ids[conditioning_tokens[0]]}")
    logger.info(f"  {conditioning_tokens[1]} -> {conditioning_token_ids[conditioning_tokens[1]]}")
    logger.info(f"  ...")
    logger.info(f"  {conditioning_tokens[-1]} -> {conditioning_token_ids[conditioning_tokens[-1]]}")
    
    # 2. Cargar labels
    logger.info("2. Cargando labels VA...")
    labels_dict = load_labels(labels_csv)
    
    # 3. Obtener lista de archivos MIDI
    logger.info("3. Escaneando archivos MIDI...")
    midi_files = sorted(midi_dir.glob("*.mid"))
    
    # Filtrar solo archivos con labels
    midi_files_with_labels = [
        f for f in midi_files 
        if f.name in labels_dict
    ]
    
    logger.info(f"Total MIDIs: {len(midi_files)}")
    logger.info(f"MIDIs con labels: {len(midi_files_with_labels)}")
    
    # Limitar si se especifica max_files
    if args.max_files is not None:
        midi_files_with_labels = midi_files_with_labels[:args.max_files]
        logger.info(f"Limitando a {args.max_files} archivos (testing)")
    
    # 4. Procesar cada MIDI
    logger.info(f"4. Procesando {len(midi_files_with_labels)} archivos MIDI...")
    all_examples = []
    discard_stats = {}
    
    for midi_file in tqdm(midi_files_with_labels, desc="Tokenizando MIDIs"):
        labels = labels_dict[midi_file.name]
        examples, discard_reason = process_midi_file(
            midi_file,
            labels,
            tokenizer,
            conditioning_token_ids,
            args.window_size,
            args.stride,
            args.min_window_tokens,
            args.bins
        )
        
        if discard_reason:
            discard_stats[discard_reason] = discard_stats.get(discard_reason, 0) + 1
        else:
            all_examples.extend(examples)
    
    logger.info(f"Total ejemplos generados: {len(all_examples)}")
    if discard_stats:
        logger.info(f"Archivos descartados por razón:")
        for reason, count in discard_stats.items():
            logger.info(f"  - {reason}: {count}")
    
    if len(all_examples) == 0:
        logger.error("No se generaron ejemplos. Verifica los datos de entrada.")
        return 1
    
    # 5. Split train/val
    logger.info("5. Dividiendo train/val por archivos...")
    train_examples, val_examples = split_by_files(
        all_examples,
        args.train_split,
        args.seed
    )
    
    logger.info(f"Conditioning tokens usados: {len(conditioning_tokens)}")
    logger.info(f"Ejemplos: {conditioning_tokens[:4]} ... {conditioning_tokens[-2:]}")
    
    # 7. Guardar dataset
    logger.info("6. Guardando dataset...")
    config = {
        'model_name': args.model_name,
        'tokenizer_vocab_size': len(tokenizer),
        'window_size': args.window_size,
        'stride': args.stride,
        'min_window_tokens': args.min_window_tokens,
        'bins': args.bins,
        'train_split': args.train_split,
        'seed': args.seed,
        'total_files_attempted': len(midi_files_with_labels),
        'total_files_processed': len(midi_files_with_labels) - sum(discard_stats.values()),
        'train_examples': len(train_examples),
        'val_examples': len(val_examples),
        'conditioning_tokens': conditioning_tokens,
        'conditioning_token_ids': conditioning_token_ids,
        'num_conditioning_tokens': len(conditioning_tokens),
        'discard_stats': discard_stats
    }
    
    save_dataset(train_examples, val_examples, output_dir, config)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("DATASET CONSTRUIDO EXITOSAMENTE")
    logger.info("=" * 80)
    logger.info(f"Train: {len(train_examples)} ejemplos")
    logger.info(f"Val: {len(val_examples)} ejemplos")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    logger.info("Próximos pasos:")
    logger.info("1. Verificar dataset: python scripts/verify_finetune_dataset.py")
    logger.info("2. Entrenar modelo (próximo ticket)")
    logger.info("")
    
    return 0


if __name__ == '__main__':
    exit(main())
