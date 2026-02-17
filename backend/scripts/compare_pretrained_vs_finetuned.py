#!/usr/bin/env python3
"""
Script para comparar modelo pretrained vs finetuned en generación musical VA-condicionada.

Este script genera MIDIs usando ambos modelos (pretrained y finetuned) sobre una rejilla
de valores (valencia, activación), extrae características musicales, y exporta resultados
para análisis comparativo posterior.

Características:
- Múltiples seeds por combinación (V,A) para reproducibilidad
- Usa el mismo engine HF Maestro-REMI para benchmark justo
- Ambos modelos usan conditioning tokens VA
- Extracción de métricas musicales objetivas
- Exportación en CSV para análisis posterior

Usage:
    python compare_pretrained_vs_finetuned.py [options]
    
    Options:
        --output_dir PATH              Directorio para resultados (default: results/compare_pre_vs_finetuned)
        --num_samples_per_va INT       Número de seeds por combinación VA (default: 3)
        --seeds LIST                   Seeds separadas por comas (default: "42,43,44")
        --grid SIZE                    Grid VA: "4x4" o "3x3" (default: "4x4")
        --finetuned_path PATH          Path al modelo finetuned (default: models/maestro_finetuned/final)
        --pretrained_id STR            HF Hub ID del pretrained (default: Natooz/Maestro-REMI-bpe20k)
        --length_bars INT              Compases por MIDI (default: 8)
        --max_tokens INT               Tokens fijos (default: 512 para control)

Example:
    python compare_pretrained_vs_finetuned.py --grid 4x4 --num_samples_per_va 3
    python compare_pretrained_vs_finetuned.py --seeds "42,43,44" --output_dir results/comparison

Output:
    - output_dir/pretrained/v{v}_a{a}/seed{seed}.mid   MIDIs del modelo pretrained
    - output_dir/finetuned/v{v}_a{a}/seed{seed}.mid    MIDIs del modelo finetuned
    - output_dir/compare_raw.csv                        Resultados detallados
"""

import sys
import argparse
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import logging

# Agregar backend al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / "src"))

from core.music.mapping import va_to_music_params
from core.music.engines.hf_maestro_remi import generate_midi_hf_maestro_remi
from core.music.analysis.features import extract_midi_features

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Grids predefinidos
GRIDS = {
    '4x4': {
        'valence': [-0.8, -0.2, 0.2, 0.8],
        'arousal': [-0.8, -0.2, 0.2, 0.8]
    },
    '3x3': {
        'valence': [-0.6, 0.0, 0.6],
        'arousal': [-0.6, 0.0, 0.6]
    }
}


def parse_seeds(seeds_str: str) -> List[int]:
    """
    Parsea string de seeds separadas por comas.
    
    Args:
        seeds_str: String como "42,43,44"
        
    Returns:
        Lista de enteros [42, 43, 44]
    """
    try:
        seeds = [int(s.strip()) for s in seeds_str.split(',')]
        return seeds
    except ValueError as e:
        raise ValueError(f"Seeds inválidas: {seeds_str}. Usa formato '42,43,44'") from e


def generate_midi_with_model(
    model_source: str,
    model_id_or_path: str,
    valence: float,
    arousal: float,
    seed: int,
    output_path: str,
    length_bars: int = 8,
    max_new_tokens: int = 512
) -> str:
    """
    Genera MIDI usando modelo especificado (pretrained o finetuned).
    
    Args:
        model_source: "pretrained" o "finetuned"
        model_id_or_path: HF ID o path local
        valence: Valor de valencia en [-1, 1]
        arousal: Valor de activación en [-1, 1]
        seed: Semilla aleatoria
        output_path: Path donde guardar MIDI
        length_bars: Número de compases
        max_new_tokens: Tokens fijos a generar
        
    Returns:
        Path al MIDI generado
    """
    # Convertir VA a parámetros musicales
    music_params = va_to_music_params(valence, arousal)
    
    logger.debug(f"Generando con {model_source}: v={valence:.2f}, a={arousal:.2f}, seed={seed}")
    
    # Generar MIDI con engine HF Maestro-REMI
    generated_path = generate_midi_hf_maestro_remi(
        params=music_params,
        out_path=output_path,
        length_bars=length_bars,
        seed=seed,
        max_new_tokens=max_new_tokens,
        model_source=model_source,
        model_id_or_path=model_id_or_path
    )
    
    return generated_path


def run_comparison(
    output_dir: Path,
    grid_name: str,
    seeds: List[int],
    pretrained_id: str,
    finetuned_path: str,
    length_bars: int,
    max_tokens: int
) -> List[Dict]:
    """
    Ejecuta benchmark comparando pretrained vs finetuned.
    
    Para cada (v,a) en el grid, para cada seed:
    - Genera MIDI con pretrained
    - Genera MIDI con finetuned
    - Extrae métricas de ambos
    - Guarda resultados
    
    Args:
        output_dir: Directorio de salida
        grid_name: Nombre del grid ("4x4" o "3x3")
        seeds: Lista de seeds a usar
        pretrained_id: HF Hub ID del pretrained
        finetuned_path: Path al modelo finetuned
        length_bars: Compases por MIDI
        max_tokens: Tokens fijos a generar
        
    Returns:
        Lista de resultados (un dict por cada generación)
    """
    # Crear directorios
    pretrained_dir = output_dir / "pretrained"
    finetuned_dir = output_dir / "finetuned"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener grid
    if grid_name not in GRIDS:
        raise ValueError(f"Grid inválido: {grid_name}. Opciones: {list(GRIDS.keys())}")
    
    grid = GRIDS[grid_name]
    valence_values = grid['valence']
    arousal_values = grid['arousal']
    
    # Calcular total de samples
    total_samples = len(valence_values) * len(arousal_values) * 2 * len(seeds)  # x2 por pretrained+finetuned
    current_sample = 0
    
    logger.info("=" * 80)
    logger.info("COMPARACIÓN PRETRAINED VS FINETUNED")
    logger.info("=" * 80)
    logger.info(f"Grid: {grid_name} ({len(valence_values)}x{len(arousal_values)})")
    logger.info(f"Valencia: {valence_values}")
    logger.info(f"Activación: {arousal_values}")
    logger.info(f"Seeds: {seeds} ({len(seeds)} seeds)")
    logger.info(f"Pretrained: {pretrained_id}")
    logger.info(f"Finetuned: {finetuned_path}")
    logger.info(f"Total muestras: {total_samples}")
    logger.info(f"Longitud: {length_bars} compases, tokens: {max_tokens}")
    logger.info("=" * 80)
    
    results = []
    
    # Iterar sobre grid
    for valence in valence_values:
        for arousal in arousal_values:
            # Crear subdirectorio para esta combinación VA
            va_dirname = f"v{valence:+.1f}_a{arousal:+.1f}".replace('.', '_')
            
            for seed in seeds:
                # 1. Generar con PRETRAINED
                current_sample += 1
                logger.info(f"[{current_sample}/{total_samples}] Pretrained: v={valence:+.1f}, a={arousal:+.1f}, seed={seed}")
                
                pretrained_subdir = pretrained_dir / va_dirname
                pretrained_subdir.mkdir(parents=True, exist_ok=True)
                pretrained_path = pretrained_subdir / f"seed{seed}.mid"
                
                try:
                    generated_path = generate_midi_with_model(
                        model_source="pretrained",
                        model_id_or_path=pretrained_id,
                        valence=valence,
                        arousal=arousal,
                        seed=seed,
                        output_path=str(pretrained_path),
                        length_bars=length_bars,
                        max_new_tokens=max_tokens
                    )
                    
                    # Extraer métricas
                    features = extract_midi_features(generated_path)
                    
                    # Guardar resultado
                    result = {
                        'model_tag': 'pretrained',
                        'model_path': pretrained_id,
                        'valence': valence,
                        'arousal': arousal,
                        'seed': seed,
                        'midi_path': str(pretrained_path.relative_to(output_dir)),
                        **features
                    }
                    results.append(result)
                    logger.debug(f"  Completado: {features['total_notes']} notas, {features['note_density']:.2f} notas/s")
                    
                except Exception as e:
                    logger.error(f"  Error: {e}")
                    results.append({
                        'model_tag': 'pretrained',
                        'model_path': pretrained_id,
                        'valence': valence,
                        'arousal': arousal,
                        'seed': seed,
                        'error': str(e)
                    })
                
                # 2. Generar con FINETUNED
                current_sample += 1
                logger.info(f"[{current_sample}/{total_samples}] Finetuned: v={valence:+.1f}, a={arousal:+.1f}, seed={seed}")
                
                finetuned_subdir = finetuned_dir / va_dirname
                finetuned_subdir.mkdir(parents=True, exist_ok=True)
                finetuned_path_out = finetuned_subdir / f"seed{seed}.mid"
                
                try:
                    generated_path = generate_midi_with_model(
                        model_source="finetuned",
                        model_id_or_path=finetuned_path,
                        valence=valence,
                        arousal=arousal,
                        seed=seed,
                        output_path=str(finetuned_path_out),
                        length_bars=length_bars,
                        max_new_tokens=max_tokens
                    )
                    
                    # Extraer métricas
                    features = extract_midi_features(generated_path)
                    
                    # Guardar resultado
                    result = {
                        'model_tag': 'finetuned',
                        'model_path': finetuned_path,
                        'valence': valence,
                        'arousal': arousal,
                        'seed': seed,
                        'midi_path': str(finetuned_path_out.relative_to(output_dir)),
                        **features
                    }
                    results.append(result)
                    logger.debug(f"  Completado: {features['total_notes']} notas, {features['note_density']:.2f} notas/s")
                    
                except Exception as e:
                    logger.error(f"  Error: {e}")
                    results.append({
                        'model_tag': 'finetuned',
                        'model_path': finetuned_path,
                        'valence': valence,
                        'arousal': arousal,
                        'seed': seed,
                        'error': str(e)
                    })
    
    logger.info("=" * 80)
    logger.info(f"Comparación completada: {len(results)} muestras generadas")
    logger.info("=" * 80)
    
    return results


def save_results_csv(results: List[Dict], output_path: Path):
    """
    Guarda resultados en CSV.
    
    Args:
        results: Lista de dicts con resultados
        output_path: Path al CSV de salida
    """
    if not results:
        logger.warning("No hay resultados para guardar")
        return
    
    # Obtener todas las columnas (union de keys de todos los dicts)
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    
    # Ordenar columnas (importantes primero)
    priority_cols = ['model_tag', 'valence', 'arousal', 'seed', 'midi_path']
    other_cols = sorted(all_keys - set(priority_cols))
    fieldnames = priority_cols + other_cols
    
    # Escribir CSV
    logger.info(f"Guardando resultados en: {output_path}")
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"CSV guardado: {output_path} ({len(results)} filas)")


def main():
    parser = argparse.ArgumentParser(
        description='Comparar modelo pretrained vs finetuned en generación VA-condicionada'
    )
    
    parser.add_argument(
        '--output_dir',
        default='results/compare_pre_vs_finetuned',
        help='Directorio de salida (default: results/compare_pre_vs_finetuned)'
    )
    parser.add_argument(
        '--num_samples_per_va',
        type=int,
        default=3,
        help='Número de seeds por combinación VA (default: 3)'
    )
    parser.add_argument(
        '--seeds',
        default='42,43,44',
        help='Seeds separadas por comas (default: "42,43,44")'
    )
    parser.add_argument(
        '--grid',
        default='4x4',
        choices=['4x4', '3x3'],
        help='Grid de valores VA (default: 4x4)'
    )
    parser.add_argument(
        '--finetuned_path',
        default='models/maestro_finetuned/final',
        help='Path al modelo finetuned (default: models/maestro_finetuned/final)'
    )
    parser.add_argument(
        '--pretrained_id',
        default='Natooz/Maestro-REMI-bpe20k',
        help='HF Hub ID del modelo pretrained (default: Natooz/Maestro-REMI-bpe20k)'
    )
    parser.add_argument(
        '--length_bars',
        type=int,
        default=8,
        help='Compases por MIDI (default: 8)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=512,
        help='Tokens fijos a generar (default: 512)'
    )
    
    args = parser.parse_args()
    
    # Paths absolutos
    output_dir = backend_dir / args.output_dir
    finetuned_path = str(backend_dir / args.finetuned_path)
    
    # Parsear seeds
    seeds = parse_seeds(args.seeds)
    
    if len(seeds) != args.num_samples_per_va:
        logger.warning(
            f"Número de seeds ({len(seeds)}) no coincide con num_samples_per_va ({args.num_samples_per_va}). "
            f"Usando {len(seeds)} seeds."
        )
    
    # Ejecutar comparación
    results = run_comparison(
        output_dir=output_dir,
        grid_name=args.grid,
        seeds=seeds,
        pretrained_id=args.pretrained_id,
        finetuned_path=finetuned_path,
        length_bars=args.length_bars,
        max_tokens=args.max_tokens
    )
    
    # Guardar resultados
    results_csv = output_dir / "compare_raw.csv"
    save_results_csv(results, results_csv)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARACIÓN COMPLETADA")
    logger.info("=" * 80)
    logger.info(f"Resultados guardados en: {output_dir}")
    logger.info(f"  - compare_raw.csv: {len(results)} muestras")
    logger.info(f"  - pretrained/: MIDIs del modelo pretrained")
    logger.info(f"  - finetuned/: MIDIs del modelo finetuned")
    logger.info("")
    logger.info("Próximos pasos:")
    logger.info("  python scripts/analyze_compare_results.py")
    logger.info("")
    
    return 0


if __name__ == '__main__':
    exit(main())
