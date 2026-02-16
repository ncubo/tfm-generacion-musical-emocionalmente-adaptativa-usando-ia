"""
Script de benchmark automático para comparar motores de generación musical.

Este script compara el motor baseline (reglas) vs transformer_pretrained (HF Maestro)
generando MIDIs sobre una rejilla de valores (valencia, activación), extrayendo características
musicales objetivas y exportando resultados en múltiples formatos.

Usage:
    python run_benchmark_models.py [options]
    
    Options:
        --grid default|custom    Grid de valores VA a usar (default: default)
        --output_dir PATH        Directorio para guardar resultados (default: results)
        --seed_base INT          Semilla base para reproducibilidad (default: 42)
        --length_bars INT        Número de compases por MIDI (default: 8)
        --verbose               Mostrar logs detallados

Example:
    python run_benchmark_models.py --grid default --seed_base 42 --verbose

Output:
    - data/benchmark_midis/       MIDIs generados
    - results/benchmark_raw.json  Resultados crudos
    - results/benchmark_table.csv Tabla CSV
    - results/benchmark_table.tex Tabla LaTeX lista para TFM
"""

import sys
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging

# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.music.mapping import va_to_music_params
from src.core.music.engines.baseline import generate_midi_baseline
from src.core.music.engines.hf_maestro_remi import generate_midi_hf_maestro_remi
from src.core.music.analysis.features import extract_midi_features


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Definición de grids de valores VA
GRIDS = {
    'default': {
        'valence': [-0.8, -0.2, 0.2, 0.8],
        'arousal': [-0.8, -0.2, 0.2, 0.8]
    },
    'custom': {
        'valence': [-1.0, -0.5, 0.0, 0.5, 1.0],
        'arousal': [-1.0, -0.5, 0.0, 0.5, 1.0]
    }
}


def create_output_directories(output_dir: Path, benchmark_dir: Path):
    """
    Crea directorios necesarios para el benchmark.
    
    Args:
        output_dir: Directorio para resultados (JSON, CSV, LaTeX)
        benchmark_dir: Directorio para MIDIs generados
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorios creados: {output_dir}, {benchmark_dir}")


def generate_midi_for_va(
    engine: str,
    valence: float,
    arousal: float,
    seed: int,
    output_path: str,
    length_bars: int = 8
) -> str:
    """
    Genera un archivo MIDI usando el motor especificado.
    
    Args:
        engine: "baseline" o "transformer_pretrained"
        valence: Valor de valencia [-1, 1]
        arousal: Valor de activación [-1, 1]
        seed: Semilla para reproducibilidad
        output_path: Ruta donde guardar el MIDI
        length_bars: Número de compases a generar
    
    Returns:
        Ruta del archivo MIDI generado
    
    Raises:
        ValueError: Si el motor no es válido
        Exception: Si falla la generación
    """
    # Convertir VA a parámetros musicales
    music_params = va_to_music_params(valence, arousal)
    
    # Generar según motor
    if engine == 'baseline':
        return generate_midi_baseline(
            params=music_params,
            out_path=output_path,
            length_bars=length_bars,
            seed=seed
        )
    elif engine == 'transformer_pretrained':
        return generate_midi_hf_maestro_remi(
            params=music_params,
            out_path=output_path,
            length_bars=length_bars,
            seed=seed
        )
    else:
        raise ValueError(f"Motor no válido: {engine}. Usar 'baseline' o 'transformer_pretrained'")


def run_benchmark(
    grid_name: str = 'default',
    seed_base: int = 42,
    output_dir: Path = None,
    length_bars: int = 8,
    verbose: bool = False
) -> List[Dict]:
    """
    Ejecuta el benchmark completo sobre la rejilla especificada.
    
    Args:
        grid_name: Nombre del grid a usar (default, custom)
        seed_base: Semilla base para reproducibilidad
        output_dir: Directorio para resultados
        length_bars: Número de compases por MIDI
        verbose: Mostrar logs detallados
    
    Returns:
        Lista de diccionarios con resultados de cada muestra
    """
    # Configurar logging
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Obtener grid
    if grid_name not in GRIDS:
        raise ValueError(f"Grid '{grid_name}' no existe. Opciones: {list(GRIDS.keys())}")
    
    grid = GRIDS[grid_name]
    valence_values = grid['valence']
    arousal_values = grid['arousal']
    
    # Directorios
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'results'
    
    benchmark_dir = Path(__file__).parent.parent / 'data' / 'benchmark_midis'
    
    create_output_directories(output_dir, benchmark_dir)
    
    # Lista para resultados
    results = []
    
    # Engines a comparar
    engines = ['baseline', 'transformer_pretrained']
    
    # Contador de muestras
    total_samples = len(valence_values) * len(arousal_values) * len(engines)
    current_sample = 0
    
    logger.info(f"Iniciando benchmark con grid '{grid_name}'")
    logger.info(f"Total de muestras a generar: {total_samples}")
    logger.info(f"Valores de valencia: {valence_values}")
    logger.info(f"Valores de activación: {arousal_values}")
    logger.info(f"Motores: {engines}")
    logger.info("=" * 60)
    
    # Iterar sobre grid
    for valence in valence_values:
        for arousal in arousal_values:
            for engine in engines:
                current_sample += 1
                
                # Crear nombre de archivo único
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{engine}_v{valence:+.1f}_a{arousal:+.1f}_seed{seed_base}_{timestamp}.mid"
                midi_path = benchmark_dir / filename
                
                logger.info(f"[{current_sample}/{total_samples}] Generando: {filename}")
                logger.debug(f"  Motor: {engine}")
                logger.debug(f"  Valencia: {valence:.2f}, Activación: {arousal:.2f}")
                logger.debug(f"  Semilla: {seed_base}, Longitud: {length_bars} compases")
                
                try:
                    # Generar MIDI
                    generated_path = generate_midi_for_va(
                        engine=engine,
                        valence=valence,
                        arousal=arousal,
                        seed=seed_base,
                        output_path=str(midi_path),
                        length_bars=length_bars
                    )
                    
                    logger.debug(f"  MIDI generado: {generated_path}")
                    
                    # Extraer features
                    features = extract_midi_features(generated_path)
                    
                    logger.debug(f"  Características extraídas: {features['total_notes']} notas, "
                               f"{features['note_density']:.2f} notas/s")
                    
                    # Construir resultado
                    result = {
                        'engine': engine,
                        'valence': valence,
                        'arousal': arousal,
                        'seed': seed_base,
                        'midi_path': str(midi_path),
                        'length_bars': length_bars,
                        **features  # Incluir todas las features
                    }
                    
                    results.append(result)
                    logger.info(f"  Completado exitosamente")
                
                except Exception as e:
                    logger.error(f"  Error generando {filename}: {e}")
                    # Registrar error pero continuar
                    results.append({
                        'engine': engine,
                        'valence': valence,
                        'arousal': arousal,
                        'seed': seed_base,
                        'midi_path': str(midi_path),
                        'length_bars': length_bars,
                        'error': str(e)
                    })
    
    logger.info("=" * 60)
    logger.info(f"Benchmark completado: {len(results)} muestras procesadas")
    
    return results


def save_results_json(results: List[Dict], output_dir: Path):
    """
    Guarda resultados en formato JSON crudo.
    
    Args:
        results: Lista de resultados del benchmark
        output_dir: Directorio donde guardar el archivo
    """
    json_path = output_dir / 'benchmark_raw.json'
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resultados JSON guardados: {json_path}")


def save_results_csv(results: List[Dict], output_dir: Path):
    """
    Guarda resultados en formato CSV tabular.
    
    Args:
        results: Lista de resultados del benchmark
        output_dir: Directorio donde guardar el archivo
    """
    csv_path = output_dir / 'benchmark_table.csv'
    
    # Filtrar resultados con errores
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        logger.warning("No hay resultados válidos para exportar a CSV")
        return
    
    # Definir columnas (excluir algunas menos relevantes para la tabla)
    columns = [
        'engine', 'valence', 'arousal', 'seed', 'length_bars',
        'note_density', 'pitch_range', 'mean_velocity', 
        'mean_note_duration', 'total_notes', 'total_duration_seconds',
        'unique_pitches'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(valid_results)
    
    logger.info(f"Tabla CSV guardada: {csv_path} ({len(valid_results)} filas)")


def save_results_latex(results: List[Dict], output_dir: Path):
    """
    Genera tabla LaTeX lista para pegar en el TFM.
    
    Args:
        results: Lista de resultados del benchmark
        output_dir: Directorio donde guardar el archivo
    """
    tex_path = output_dir / 'benchmark_table.tex'
    
    # Filtrar resultados válidos
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        logger.warning("No hay resultados válidos para exportar a LaTeX")
        return
    
    # Obtener metadata común (seed, length_bars)
    if valid_results:
        seed_base = valid_results[0].get('seed', 'N/A')
        length_bars = valid_results[0].get('length_bars', 'N/A')
    else:
        seed_base = 'N/A'
        length_bars = 'N/A'
    
    # Mapeo de nombres de engines a versiones cortas para LaTeX
    engine_names = {
        'baseline': 'Baseline',
        'transformer_pretrained': 'Transformer'
    }
    
    # Generar LaTeX
    latex_content = []
    latex_content.append("% Tabla de benchmark de motores de generación musical")
    latex_content.append("% Generada automáticamente por run_benchmark_models.py")
    latex_content.append("")
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append(f"\\caption{{Comparación de motores de generación musical sobre grid Valencia-Activación "
                        f"(seed={seed_base}, {length_bars} compases)}}")
    latex_content.append("\\label{tab:benchmark_engines}")
    latex_content.append("\\resizebox{\\textwidth}{!}{%")
    latex_content.append("\\begin{tabular}{llrrrrrrr}")
    latex_content.append("\\hline")
    latex_content.append("\\textbf{Motor} & \\textbf{V/A} & \\textbf{Densidad} & \\textbf{Rango} & "
                        "\\textbf{Velocidad} & \\textbf{Duración} & \\textbf{Notas} & "
                        "\\textbf{Total (s)} & \\textbf{Únicas} \\\\")
    latex_content.append("& & \\textbf{(n/s)} & \\textbf{(st)} & "
                        "\\textbf{(vel)} & \\textbf{(s)} & & & \\\\")
    latex_content.append("\\hline")
    
    # Ordenar por (valencia, activación) primero, luego por motor
    # Esto agrupa baseline y transformer para cada combinación VA
    sorted_results = sorted(valid_results, key=lambda r: (r['valence'], r['arousal'], r['engine']))
    
    # Track previous VA for separación visual
    prev_va = None
    
    for result in sorted_results:
        current_va = (result['valence'], result['arousal'])
        
        # Agregar línea separadora entre diferentes combinaciones VA
        if prev_va is not None and prev_va != current_va:
            latex_content.append("\\hline")
        
        va_label = f"({result['valence']:+.1f}, {result['arousal']:+.1f})"
        engine_label = engine_names.get(result['engine'], result['engine'])
        
        row = (
            f"{engine_label} & "
            f"{va_label} & "
            f"{result['note_density']:.2f} & "
            f"{result['pitch_range']} & "
            f"{result['mean_velocity']:.1f} & "
            f"{result['mean_note_duration']:.2f} & "
            f"{result['total_notes']} & "
            f"{result['total_duration_seconds']:.1f} & "
            f"{result['unique_pitches']} \\\\"
        )
        latex_content.append(row)
        prev_va = current_va
    
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("}")
    latex_content.append("\\end{table}")
    
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    logger.info(f"Tabla LaTeX guardada: {tex_path}")


def print_summary(results: List[Dict]):
    """
    Imprime resumen del benchmark en consola.
    
    Args:
        results: Lista de resultados del benchmark
    """
    valid_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]
    
    print("\n" + "=" * 60)
    print("RESUMEN DEL BENCHMARK")
    print("=" * 60)
    print(f"Total de muestras: {len(results)}")
    print(f"Exitosas: {len(valid_results)}")
    print(f"Con errores: {len(error_results)}")
    
    if error_results:
        print("\nErrores encontrados:")
        for err in error_results:
            print(f"  - {err['engine']} (V={err['valence']}, A={err['arousal']}): {err['error']}")
    
    if valid_results:
        # Calcular estadísticas por motor
        engines = set(r['engine'] for r in valid_results)
        
        print("\nEstadísticas por engine:")
        for engine in sorted(engines):
            engine_results = [r for r in valid_results if r['engine'] == engine]
            
            avg_density = sum(r['note_density'] for r in engine_results) / len(engine_results)
            avg_range = sum(r['pitch_range'] for r in engine_results) / len(engine_results)
            avg_velocity = sum(r['mean_velocity'] for r in engine_results) / len(engine_results)
            
            print(f"\n  {engine}:")
            print(f"    Muestras: {len(engine_results)}")
            print(f"    Densidad promedio: {avg_density:.2f} notas/s")
            print(f"    Rango tonal promedio: {avg_range:.1f} semitonos")
            print(f"    Velocity promedio: {avg_velocity:.1f}")
    
    print("=" * 60 + "\n")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Benchmark automático de motores de generación musical',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--grid',
        type=str,
        default='default',
        choices=list(GRIDS.keys()),
        help='Grid de valores VA a usar (default: default)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Directorio para guardar resultados (default: backend/results)'
    )
    
    parser.add_argument(
        '--seed_base',
        type=int,
        default=42,
        help='Semilla base para reproducibilidad (default: 42)'
    )
    
    parser.add_argument(
        '--length_bars',
        type=int,
        default=8,
        help='Número de compases por MIDI (default: 8)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mostrar logs detallados'
    )
    
    args = parser.parse_args()
    
    try:
        # Ejecutar benchmark
        results = run_benchmark(
            grid_name=args.grid,
            seed_base=args.seed_base,
            output_dir=args.output_dir,
            length_bars=args.length_bars,
            verbose=args.verbose
        )
        
        # Determinar output_dir para guardar resultados
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'results'
        
        # Guardar resultados en múltiples formatos
        save_results_json(results, output_dir)
        save_results_csv(results, output_dir)
        save_results_latex(results, output_dir)
        
        # Mostrar resumen
        print_summary(results)
        
        logger.info("¡Benchmark completado exitosamente!")
        logger.info(f"Resultados guardados en: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error ejecutando benchmark: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
