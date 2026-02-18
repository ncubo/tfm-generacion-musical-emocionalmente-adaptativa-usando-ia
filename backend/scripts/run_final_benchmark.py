#!/usr/bin/env python3
"""
Script de generación de benchmark final de motores de generación musical.

Genera MIDIs para los 3 engines (baseline, transformer_pretrained, transformer_finetuned)
sobre un grid VA reproducible, extrae métricas y genera CSV crudo.

Uso:
    python run_final_benchmark.py
    python run_final_benchmark.py --grid 3x3 --seeds "42,43"
    python run_final_benchmark.py --output_dir results/benchmark_test --max_items 10
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import csv
import json
import time
import traceback

# Añadir backend/src al path para imports
BACKEND_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

# Imports de módulos internos
from core.music.mapping import va_to_music_params
from core.music.engines.baseline import generate_midi_baseline
from core.music.engines.hf_maestro_remi import generate_midi_hf_maestro_remi
from core.music.analysis.features import extract_midi_features

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ===== CONFIGURACIÓN DE ENGINES =====

# Configuración de los 3 engines
ENGINES_CONFIG = {
    "baseline": {
        "name": "Baseline (heurístico)",
        "repo": "N/A (rule-based)",
        "generate_fn": lambda params, path, seed: generate_midi_baseline(
            params=params,
            out_path=path,
            length_bars=8,
            seed=seed
        )
    },
    "transformer_pretrained": {
        "name": "Transformer Pretrained",
        "repo": "Natooz/Maestro-REMI-bpe20k",
        "generate_fn": lambda params, path, seed: generate_midi_hf_maestro_remi(
            params=params,
            out_path=path,
            length_bars=8,
            seed=seed,
            model_source="pretrained",
            model_id_or_path="Natooz/Maestro-REMI-bpe20k"
        )
    },
    "transformer_finetuned": {
        "name": "Transformer Finetuned VA",
        "repo": "mmayorga/maestro-remi-finetuned-va",
        "generate_fn": lambda params, path, seed: generate_midi_hf_maestro_remi(
            params=params,
            out_path=path,
            length_bars=8,
            seed=seed,
            model_source="pretrained",
            model_id_or_path="mmayorga/maestro-remi-finetuned-va"
        )
    }
}


# ===== FUNCIONES AUXILIARES =====

def create_va_grid(grid_type: str = "4x4") -> List[Tuple[float, float]]:
    """
    Crea un grid de valores (valence, arousal).
    
    Args:
        grid_type: "4x4" o "3x3"
        
    Returns:
        Lista de tuplas (v, a)
    """
    if grid_type == "4x4":
        values = [-0.8, -0.2, 0.2, 0.8]
    elif grid_type == "3x3":
        values = [-0.7, 0.0, 0.7]
    else:
        raise ValueError(f"grid_type inválido: {grid_type}. Usar '4x4' o '3x3'")
    
    grid = [(v, a) for v in values for a in values]
    logger.info(f"Grid VA creado: {len(grid)} puntos ({grid_type})")
    return grid


def generate_single_midi(
    engine_name: str,
    valence: float,
    arousal: float,
    seed: int,
    output_dir: Path,
    skip_existing: bool = False
) -> Dict[str, Any]:
    """
    Genera un MIDI individual y extrae métricas.
    
    Args:
        engine_name: Nombre del engine ("baseline", "transformer_pretrained", etc.)
        valence: Valor de valencia [-1, 1]
        arousal: Valor de arousal [-1, 1]
        seed: Semilla aleatoria
        output_dir: Directorio base de resultados
        skip_existing: Si True, no regenera si el archivo ya existe
        
    Returns:
        Dict con resultados (status, metrics, path, generation_time_ms, error)
    """
    # Crear path del MIDI
    engine_dir = output_dir / engine_name / f"v{valence:+.1f}_a{arousal:+.1f}"
    engine_dir.mkdir(parents=True, exist_ok=True)
    midi_path = engine_dir / f"seed{seed}.mid"
    
    # Verificar si ya existe
    if skip_existing and midi_path.exists():
        logger.debug(f"Saltando {midi_path} (ya existe)")
        
        # Extraer métricas del existente
        try:
            features = extract_midi_features(str(midi_path))
            return {
                "status": "skipped",
                "midi_path": str(midi_path.relative_to(output_dir)),
                "generation_time_ms": 0,
                "metrics": features,
                "error": None
            }
        except Exception as e:
            logger.warning(f"Error al leer MIDI existente {midi_path}: {e}")
            # Continuar y regenerar
    
    # Obtener engine config
    engine_config = ENGINES_CONFIG.get(engine_name)
    if engine_config is None:
        return {
            "status": "error",
            "midi_path": None,
            "generation_time_ms": 0,
            "metrics": {},
            "error": f"Engine desconocido: {engine_name}"
        }
    
    try:
        # Derivar parámetros musicales
        params = va_to_music_params(valence, arousal)
        
        # Generar MIDI
        logger.info(f"Generando: {engine_name} | V={valence:+.1f} A={arousal:+.1f} | seed={seed}")
        
        start_time = time.perf_counter()
        engine_config["generate_fn"](params, str(midi_path), seed)
        end_time = time.perf_counter()
        
        generation_time_ms = (end_time - start_time) * 1000
        
        # Extraer métricas
        features = extract_midi_features(str(midi_path))
        
        logger.info(f"  Generado en {generation_time_ms:.0f}ms | "
                   f"Notas: {features['total_notes']} | "
                   f"Density: {features['note_density']:.2f} n/s")
        
        return {
            "status": "success",
            "midi_path": str(midi_path.relative_to(output_dir)),
            "generation_time_ms": generation_time_ms,
            "metrics": features,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"  ✗ Error generando MIDI: {error_msg}")
        logger.debug(traceback.format_exc())
        
        return {
            "status": "error",
            "midi_path": None,
            "generation_time_ms": 0,
            "metrics": {},
            "error": error_msg
        }


def run_benchmark(
    output_dir: Path,
    grid_type: str = "4x4",
    seeds: List[int] = [42, 43, 44],
    engines: List[str] = None,
    skip_existing: bool = False,
    max_items: Optional[int] = None
) -> Tuple[Path, Dict[str, Any]]:
    """
    Ejecuta el benchmark completo.
    
    Args:
        output_dir: Directorio de salida
        grid_type: "4x4" o "3x3"
        seeds: Lista de semillas
        engines: Lista de engines a usar (None = todos)
        skip_existing: No regenerar MIDIs existentes
        max_items: Máximo de items a generar (para pruebas)
        
    Returns:
        Tupla (csv_path, metadata_dict)
    """
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de resultados: {output_dir}")
    
    # Crear grid VA
    va_grid = create_va_grid(grid_type)
    
    # Seleccionar engines
    if engines is None:
        engines = list(ENGINES_CONFIG.keys())
    else:
        # Validar engines
        invalid = [e for e in engines if e not in ENGINES_CONFIG]
        if invalid:
            raise ValueError(f"Engines inválidos: {invalid}")
    
    logger.info(f"Engines a evaluar: {engines}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Total de combinaciones: {len(engines)} engines × {len(va_grid)} puntos VA × {len(seeds)} seeds = {len(engines) * len(va_grid) * len(seeds)}")
    
    # Preparar archivo CSV
    csv_path = output_dir / "benchmark_raw.csv"
    csv_fields = [
        "engine",
        "valence",
        "arousal",
        "seed",
        "status",
        "generation_time_ms",
        "note_density",
        "pitch_range",
        "mean_velocity",
        "mean_note_duration",
        "total_notes",
        "total_duration_seconds",
        "unique_pitches",
        "midi_path",
        "error"
    ]
    
    # Generar todos los MIDIs
    results = []
    item_count = 0
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    logger.info("=" * 60)
    logger.info("INICIANDO BENCHMARK")
    logger.info("=" * 60)
    
    benchmark_start = time.time()
    
    for engine_name in engines:
        for valence, arousal in va_grid:
            for seed in seeds:
                # Verificar límite de items
                if max_items is not None and item_count >= max_items:
                    logger.warning(f"Alcanzado límite de items: {max_items}")
                    break
                
                # Generar MIDI
                result = generate_single_midi(
                    engine_name=engine_name,
                    valence=valence,
                    arousal=arousal,
                    seed=seed,
                    output_dir=output_dir,
                    skip_existing=skip_existing
                )
                
                # Preparar fila CSV
                row = {
                    "engine": engine_name,
                    "valence": valence,
                    "arousal": arousal,
                    "seed": seed,
                    "status": result["status"],
                    "generation_time_ms": result["generation_time_ms"],
                    "midi_path": result["midi_path"] or "",
                    "error": result["error"] or ""
                }
                
                # Añadir métricas
                metrics = result["metrics"]
                row["note_density"] = metrics.get("note_density", "")
                row["pitch_range"] = metrics.get("pitch_range", "")
                row["mean_velocity"] = metrics.get("mean_velocity", "")
                row["mean_note_duration"] = metrics.get("mean_note_duration", "")
                row["total_notes"] = metrics.get("total_notes", "")
                row["total_duration_seconds"] = metrics.get("total_duration_seconds", "")
                row["unique_pitches"] = metrics.get("unique_pitches", "")
                
                results.append(row)
                item_count += 1
                
                # Contadores
                if result["status"] == "success":
                    success_count += 1
                elif result["status"] == "error":
                    error_count += 1
                elif result["status"] == "skipped":
                    skipped_count += 1
            
            if max_items is not None and item_count >= max_items:
                break
        
        if max_items is not None and item_count >= max_items:
            break
    
    benchmark_end = time.time()
    total_time_seconds = benchmark_end - benchmark_start
    
    # Guardar CSV
    logger.info("=" * 60)
    logger.info("Guardando resultados...")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"CSV guardado: {csv_path} ({len(results)} filas)")
    
    # Crear metadata
    metadata = {
        "benchmark_date": datetime.now().isoformat(),
        "total_time_seconds": round(total_time_seconds, 2),
        "config": {
            "grid_type": grid_type,
            "grid_size": len(va_grid),
            "seeds": seeds,
            "engines": engines,
            "max_items": max_items,
            "skip_existing": skip_existing
        },
        "engines_info": {
            name: {
                "name": cfg["name"],
                "repo": cfg["repo"]
            }
            for name, cfg in ENGINES_CONFIG.items()
            if name in engines
        },
        "results_summary": {
            "total_items": item_count,
            "success": success_count,
            "errors": error_count,
            "skipped": skipped_count
        },
        "output_dir": str(output_dir),
        "csv_file": str(csv_path.name)
    }
    
    # Guardar metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Metadata guardada: {metadata_path}")
    
    # Resumen final
    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETADO")
    logger.info("=" * 60)
    logger.info(f"Tiempo total: {total_time_seconds:.1f}s ({total_time_seconds/60:.1f} min)")
    logger.info(f"Items procesados: {item_count}")
    logger.info(f"  - Exitosos: {success_count}")
    logger.info(f"  - Errores: {error_count}")
    logger.info(f"  - Saltados: {skipped_count}")
    
    if error_count > 0:
        logger.warning(f"⚠️  {error_count} items fallaron. Revisar columna 'error' en CSV.")
    
    return csv_path, metadata


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Benchmark final de motores de generación musical",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Benchmark completo (4x4 grid, 3 seeds, 3 engines = 144 MIDIs)
  python run_final_benchmark.py
  
  # Benchmark rápido (3x3 grid, 1 seed = 27 MIDIs)
  python run_final_benchmark.py --grid 3x3 --seeds "42"
  
  # Prueba con límite (solo 10 MIDIs)
  python run_final_benchmark.py --max_items 10
  
  # Benchmark solo baseline
  python run_final_benchmark.py --engines baseline
  
  # No regenerar existentes
  python run_final_benchmark.py --skip_existing
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio de salida (default: results/final_benchmark_TIMESTAMP)'
    )
    
    parser.add_argument(
        '--grid',
        type=str,
        default='4x4',
        choices=['4x4', '3x3'],
        help='Tamaño del grid VA (default: 4x4)'
    )
    
    parser.add_argument(
        '--seeds',
        type=str,
        default='42,43,44',
        help='Seeds separadas por comas (default: "42,43,44")'
    )
    
    parser.add_argument(
        '--engines',
        type=str,
        default=None,
        help='Engines separados por comas (default: todos). Opciones: baseline,transformer_pretrained,transformer_finetuned'
    )
    
    parser.add_argument(
        '--max_items',
        type=int,
        default=None,
        help='Máximo de items a generar (para pruebas)'
    )
    
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='No regenerar MIDIs que ya existen'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device para modelos transformer (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Configurar device si se especifica
    if args.device:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.device == 'cpu' else '0'
        logger.info(f"Device forzado: {args.device}")
    
    # Parsear seeds
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        logger.error(f"Error: seeds inválidas '{args.seeds}'. Usar formato: 42,43,44")
        sys.exit(1)
    
    # Parsear engines
    engines = None
    if args.engines:
        engines = [e.strip() for e in args.engines.split(',')]
    
    # Crear directorio de salida
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BACKEND_ROOT / "results" / f"final_benchmark_{timestamp}"
    
    # Ejecutar benchmark
    try:
        csv_path, metadata = run_benchmark(
            output_dir=output_dir,
            grid_type=args.grid,
            seeds=seeds,
            engines=engines,
            skip_existing=args.skip_existing,
            max_items=args.max_items
        )
        
        logger.info("=" * 60)
        logger.info("Para analizar los resultados, ejecuta:")
        logger.info(f"  python scripts/analyze_final_benchmark.py {output_dir}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
