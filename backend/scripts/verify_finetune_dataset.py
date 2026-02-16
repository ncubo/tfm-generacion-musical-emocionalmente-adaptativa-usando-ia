#!/usr/bin/env python3
"""
Script de verificación para el dataset de fine-tuning.

Carga el dataset construido y verifica:
- Estructura correcta
- Presencia de conditioning tokens
- Longitudes de secuencias
- Distribución de VA bins
- Ejemplos decodificados

Uso:
    python scripts/verify_finetune_dataset.py
    python scripts/verify_finetune_dataset.py --dataset_dir data/finetune_dataset
    python scripts/verify_finetune_dataset.py --num_examples 5
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

# Añadir src al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / "src"))

from core.music.tokenization.remi import load_remi_tokenizer


def load_dataset_info(dataset_dir: Path) -> dict:
    """Carga metadata del dataset."""
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Metadata no encontrada: {info_path}")
    
    with open(info_path) as f:
        return json.load(f)


def print_section(title: str):
    """Imprime encabezado de sección."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def verify_dataset_structure(dataset_dir: Path):
    """Verifica que existan train y val."""
    train_path = dataset_dir / "train"
    val_path = dataset_dir / "val"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset no encontrado: {train_path}")
    
    if not val_path.exists():
        raise FileNotFoundError(f"Val dataset no encontrado: {val_path}")
    
    print(f"Train dataset: {train_path}")
    print(f"Val dataset: {val_path}")


def analyze_dataset_stats(dataset, split_name: str):
    """Analiza estadísticas del dataset."""
    print_section(f"ESTADÍSTICAS - {split_name.upper()}")
    
    print(f"Total ejemplos: {len(dataset)}")
    
    # Longitudes de secuencias
    lengths = [len(ex['input_ids']) for ex in dataset]
    print(f"\nLongitudes de secuencias:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    
    # Distribución de VA bins
    valence_bins = [ex['valence_bin'] for ex in dataset]
    arousal_bins = [ex['arousal_bin'] for ex in dataset]
    
    v_counter = Counter(valence_bins)
    a_counter = Counter(arousal_bins)
    
    print(f"\nDistribución Valence bins:")
    for bin_val in sorted(v_counter.keys()):
        count = v_counter[bin_val]
        pct = count / len(dataset) * 100
        print(f"  {bin_val:+2d}: {count:5d} ({pct:5.2f}%)")
    
    print(f"\nDistribución Arousal bins:")
    for bin_val in sorted(a_counter.keys()):
        count = a_counter[bin_val]
        pct = count / len(dataset) * 100
        print(f"  {bin_val:+2d}: {count:5d} ({pct:5.2f}%)")
    
    # Archivos únicos
    source_files = set(ex['source_file'] for ex in dataset)
    print(f"\nArchivos fuente únicos: {len(source_files)}")


def show_examples(dataset, tokenizer, info: dict, num_examples: int = 3):
    """Muestra ejemplos del dataset decodificados."""
    print_section(f"EJEMPLOS DEL DATASET (primeros {num_examples})")
    
    # Obtener mapeo de conditioning tokens si está disponible
    cond_token_ids = info.get('conditioning_token_ids', {})
    
    for i in range(min(num_examples, len(dataset))):
        ex = dataset[i]
        
        print(f"\n--- Ejemplo {i+1} ---")
        print(f"Archivo fuente: {ex['source_file']}")
        print(f"Window index: {ex['window_idx']}")
        print(f"Valence: {ex['valence']:.3f} (bin: {ex['valence_bin']:+d})")
        print(f"Arousal: {ex['arousal']:.3f} (bin: {ex['arousal_bin']:+d})")
        print(f"Conditioning tokens esperados: {ex['conditioning_tokens']}")
        print(f"Longitud secuencia: {len(ex['input_ids'])} tokens")
        
        # Mostrar primeros y últimos tokens
        input_ids = ex['input_ids']
        print(f"\nPrimeros 20 token IDs:")
        print(f"  {input_ids[:20]}")
        
        # Decodificar primeros tokens como strings
        try:
            # miditok decode espera un solo ID, no una lista
            # Decodificamos cada token y los unimos
            first_10_decoded = []
            for token_id in input_ids[:10]:
                try:
                    # Primero buscar en conditioning tokens
                    token_found = None
                    for token_str, tid in cond_token_ids.items():
                        if tid == token_id:
                            token_found = token_str
                            break
                    
                    if token_found:
                        first_10_decoded.append(token_found)
                    else:
                        # Es un token musical regular, usar tokenizador
                        # Para REMI, los tokens se pueden acceder desde el vocab
                        if hasattr(tokenizer, 'vocab'):
                            # Buscar en vocab inverso
                            inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
                            if token_id in inv_vocab:
                                first_10_decoded.append(inv_vocab[token_id])
                            else:
                                first_10_decoded.append(f"<ID:{token_id}>")
                        else:
                            first_10_decoded.append(f"<ID:{token_id}>")
                except Exception:
                    first_10_decoded.append(f"<ID:{token_id}>")
            
            print(f"\nPrimeros 10 tokens decodificados:")
            print(f"  {' '.join(first_10_decoded)}")
        except Exception as e:
            print(f"\nWARNING: No se pudo decodificar: {e}")
        
        print(f"\nÚltimos 10 token IDs:")
        print(f"  {input_ids[-10:]}")
        
        # VERIFICACIÓN CRÍTICA: No debe haber IDs negativos
        negative_ids = [idx for idx in input_ids if idx < 0]
        if negative_ids:
            print(f"\nERROR CRÍTICO: IDs negativos encontrados: {negative_ids}")
            print(f"   Esto indica que los conditioning tokens no fueron reemplazados correctamente.")
            raise ValueError(f"Dataset contiene IDs negativos (placeholders). Reconstruir dataset.")
        else:
            print(f"\nVALIDACION: Sin IDs negativos")
        
        # Decodificar primeros 2 tokens (deberían ser conditioning)
        try:
            # miditok usa decode para convertir IDs a tokens
            first_two_ids = input_ids[:2]
            
            # Buscar en el mapeo de conditioning tokens
            decoded_conds = []
            for token_str, token_id in cond_token_ids.items():
                if token_id in first_two_ids:
                    decoded_conds.append(token_str)
            
            if len(decoded_conds) == 2:
                print(f"Conditioning tokens decodificados: {decoded_conds}")
                if decoded_conds == ex['conditioning_tokens']:
                    print(f"Coinciden con los esperados")
                else:
                    print(f"WARNING: No coinciden con esperados: {ex['conditioning_tokens']}")
            else:
                print(f"WARNING: No se pudieron decodificar los primeros 2 tokens como conditioning")
                
        except Exception as e:
            print(f"WARNING: Error decodificando conditioning tokens: {e}")
        
        # Verificar attention mask
        if len(ex['attention_mask']) != len(ex['input_ids']):
            print(f"WARNING: attention_mask length mismatch!")
        
        # Verificar labels
        if ex['labels'] != ex['input_ids']:
            print(f"WARNING: labels != input_ids (para LM deberían ser iguales)")


def main():
    parser = argparse.ArgumentParser(
        description='Verificar dataset de fine-tuning'
    )
    parser.add_argument(
        '--dataset_dir',
        default='data/finetune_dataset',
        help='Directorio del dataset'
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=3,
        help='Número de ejemplos a mostrar'
    )
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'both'],
        default='both',
        help='Qué split verificar'
    )
    
    args = parser.parse_args()
    
    dataset_dir = backend_dir / args.dataset_dir
    
    print_section("VERIFICACIÓN DE DATASET FINE-TUNING")
    print(f"Dataset dir: {dataset_dir}")
    
    # 1. Verificar estructura
    print_section("1. VERIFICANDO ESTRUCTURA")
    verify_dataset_structure(dataset_dir)
    
    # 2. Cargar metadata
    print_section("2. METADATA DEL DATASET")
    info = load_dataset_info(dataset_dir)
    
    for key, value in info.items():
        if key == 'conditioning_tokens':
            print(f"{key}: {len(value)} tokens")
            print(f"  Ejemplos: {value[:4]} ... {value[-2:]}")
        else:
            print(f"{key}: {value}")
    
    # 3. Cargar datasets
    print_section("3. CARGANDO DATASETS")
    
    try:
        from datasets import load_from_disk
    except ImportError:
        print("ERROR: datasets no está instalado")
        print("Ejecuta: pip install datasets")
        return 1
    
    splits_to_check = []
    if args.split in ['train', 'both']:
        splits_to_check.append('train')
    if args.split in ['val', 'both']:
        splits_to_check.append('val')
    
    # Cargar tokenizador para decodificación
    print("Cargando tokenizador REMI...")
    tokenizer = load_remi_tokenizer(info['model_name'])
    
    for split_name in splits_to_check:
        split_path = dataset_dir / split_name
        print(f"\nCargando {split_name} desde: {split_path}")
        dataset = load_from_disk(str(split_path))
        
        # 4. Analizar estadísticas
        analyze_dataset_stats(dataset, split_name)
        
        # 5. Mostrar ejemplos
        if split_name == 'train' or args.split == 'val':
            show_examples(dataset, tokenizer, info, args.num_examples)
    
    # Resumen final
    print_section("RESUMEN")
    print("Dataset verificado exitosamente")
    print("")
    
    # Verificar si hay información de descarte
    if 'discard_stats' in info and info['discard_stats']:
        print("Estadísticas de archivos descartados:")
        for reason, count in info['discard_stats'].items():
            print(f"  - {reason}: {count} archivos")
        print("")
    
    print("Información del tokenizador:")
    print(f"  - Vocab size original: 20000")
    print(f"  - Vocab size con conditioning: {info.get('tokenizer_vocab_size', 'N/A')}")
    print(f"  - Conditioning tokens añadidos: {info.get('num_conditioning_tokens', 'N/A')}")
    print("")
    
    print("IMPORTANTE - Próximos pasos para entrenamiento:")
    print("1. Cargar tokenizador con conditioning tokens:")
    print("   tokenizer = load_remi_tokenizer(model_name)")
    print("   tokenizer.add_special_tokens({'additional_special_tokens': conditioning_tokens})")
    print("2. Cargar modelo y redimensionar embeddings:")
    print("   model = AutoModelForCausalLM.from_pretrained(model_name)")
    print("   model.resize_token_embeddings(len(tokenizer))")
    print("3. Entrenar con transformers.Trainer usando el dataset guardado")
    print("")
    
    return 0


if __name__ == '__main__':
    exit(main())
