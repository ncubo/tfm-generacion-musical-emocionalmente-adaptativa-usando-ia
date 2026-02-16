"""
Script de benchmark automático para comparar motores de generación musical.

Este script compara el motor baseline (reglas) vs transformer_pretrained (HF Maestro)
generando MIDIs sobre una rejilla de valores (valencia, activación), extrayendo características
musicales objetivas y exportando resultados en múltiples formatos.

Características:
- Múltiples seeds por combinación (V,A) para rigor estadístico
- Control de longitud fija para Transformer (max_new_tokens)
- Agregación estadística con mean ± std
- Tablas LaTeX listas para TFM
- Tabla resumen por nivel de arousal

Usage:
    python run_benchmark_models.py [options]
    
    Options:
        --grid default|custom           Grid de valores VA a usar (default: default)
        --output_dir PATH              Directorio para guardar resultados (default: results)
        --seed_base INT                Semilla base para reproducibilidad (default: 42)
        --num_seeds INT                Número de seeds por combinación (default: 1, recomendado: 5)
        --max_tokens INT               Tokens fijos para Transformer (default: None=auto, recomendado: 512)
        --length_bars INT              Número de compases por MIDI (default: 8)
        --verbose                      Mostrar logs detallados

Example:
    # Benchmark simple (1 seed)
    python run_benchmark_models.py --grid default --seed_base 42
    
    # Benchmark estadísticamente robusto (5 seeds, longitud fija)
    python run_benchmark_models.py --grid default --seed_base 42 --num_seeds 5 --max_tokens 512 --verbose

Output:
    - data/benchmark_midis/              MIDIs generados individuales
    - results/benchmark_raw.json         Resultados crudos de todas las ejecuciones
    - results/benchmark_table.csv        Tabla CSV individual
    - results/benchmark_table.tex        Tabla LaTeX individual
    - results/benchmark_aggregated.csv   Tabla CSV agregada (mean ± std)
    - results/benchmark_aggregated.tex   Tabla LaTeX agregada (mean ± std)
    - results/benchmark_arousal_summary.tex  Tabla resumen por arousal
"""

import sys
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging
import numpy as np
from collections import defaultdict

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
    length_bars: int = 8,
    max_new_tokens: int = None
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
        max_new_tokens: Tokens fijos para transformer (benchmark mode)
    
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
            seed=seed,
            max_new_tokens=max_new_tokens  # Pasar para control de longitud
        )
    else:
        raise ValueError(f"Motor no válido: {engine}. Usar 'baseline' o 'transformer_pretrained'")


def run_benchmark(
    grid_name: str = 'default',
    seed_base: int = 42,
    num_seeds: int = 1,
    max_new_tokens: int = None,
    output_dir: Path = None,
    length_bars: int = 8,
    verbose: bool = False
) -> List[Dict]:
    """
    Ejecuta el benchmark completo sobre la rejilla especificada.
    
    Args:
        grid_name: Nombre del grid a usar (default, custom)
        seed_base: Semilla base para reproducibilidad
        num_seeds: Número de seeds a ejecutar por combinación (V,A)
        max_new_tokens: Tokens fijos para transformer (None = auto)
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
    total_samples = len(valence_values) * len(arousal_values) * len(engines) * num_seeds
    current_sample = 0
    
    logger.info(f"Iniciando benchmark con grid '{grid_name}'")
    logger.info(f"Total de muestras a generar: {total_samples}")
    logger.info(f"Valores de valencia: {valence_values}")
    logger.info(f"Valores de activación: {arousal_values}")
    logger.info(f"Motores: {engines}")
    logger.info(f"Seeds por combinación: {num_seeds} (base={seed_base})")
    if max_new_tokens:
        logger.info(f"Tokens fijos para Transformer: {max_new_tokens}")
    logger.info("=" * 60)
    
    # Iterar sobre grid con múltiples seeds
    for valence in valence_values:
        for arousal in arousal_values:
            for engine in engines:
                for seed_idx in range(num_seeds):
                    current_sample += 1
                    
                    # Calcular seed actual: seed_base + seed_idx
                    current_seed = seed_base + seed_idx
                    
                    # Crear nombre de archivo único
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{engine}_v{valence:+.1f}_a{arousal:+.1f}_seed{current_seed}_{timestamp}.mid"
                    midi_path = benchmark_dir / filename
                    
                    logger.info(f"[{current_sample}/{total_samples}] Generando: {filename}")
                    logger.debug(f"  Motor: {engine}")
                    logger.debug(f"  Valencia: {valence:.2f}, Activación: {arousal:.2f}")
                    logger.debug(f"  Semilla: {current_seed}, Longitud: {length_bars} compases")
                    
                    try:
                        # Generar MIDI
                        generated_path = generate_midi_for_va(
                            engine=engine,
                            valence=valence,
                            arousal=arousal,
                            seed=current_seed,
                            output_path=str(midi_path),
                            length_bars=length_bars,
                            max_new_tokens=max_new_tokens
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
                            'seed': current_seed,
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
                            'seed': current_seed,
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


def aggregate_results(results: List[Dict]) -> List[Dict]:
    """
    Agrega resultados por (engine, valence, arousal), calculando mean ± std.
    
    Agrupa todas las ejecuciones con diferentes seeds para una misma combinación
    (engine, V, A) y calcula estadísticas (media, desviación estándar) para
    cada métrica musical.
    
    Args:
        results: Lista de resultados individuales del benchmark
    
    Returns:
        Lista de diccionarios agregados con formato:
        {
            'engine': str,
            'valence': float,
            'arousal': float,
            'num_samples': int,
            '<metric>_mean': float,
            '<metric>_std': float,
            ...
        }
    
    Example:
        >>> raw_results = run_benchmark(..., num_seeds=5)
        >>> aggregated = aggregate_results(raw_results)
        >>> print(aggregated[0]['note_density_mean'], '±', aggregated[0]['note_density_std'])
    """
    # Filtrar resultados válidos (sin errores)
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        logger.warning("No hay resultados válidos para agregar")
        return []
    
    # Agrupar por (engine, valence, arousal)
    groups = defaultdict(list)
    
    for result in valid_results:
        key = (result['engine'], result['valence'], result['arousal'])
        groups[key].append(result)
    
    # Métricas a agregar (todas las features numéricas)
    metrics = [
        'note_density', 'pitch_range', 'mean_velocity', 'mean_note_duration',
        'total_notes', 'total_duration_seconds', 'min_pitch', 'max_pitch',
        'unique_pitches'
    ]
    
    # Calcular estadísticas para cada grupo
    aggregated = []
    
    for (engine, valence, arousal), group_results in sorted(groups.items()):
        agg_row = {
            'engine': engine,
            'valence': valence,
            'arousal': arousal,
            'num_samples': len(group_results)
        }
        
        # Calcular mean y std para cada métrica
        for metric in metrics:
            values = [r[metric] for r in group_results]
            
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            
            agg_row[f'{metric}_mean'] = round(mean_val, 3)
            agg_row[f'{metric}_std'] = round(std_val, 3)
        
        aggregated.append(agg_row)
    
    logger.info(f"Resultados agregados: {len(aggregated)} combinaciones (V,A,engine)")
    
    return aggregated


def aggregate_by_arousal(aggregated_results: List[Dict]) -> List[Dict]:
    """
    Agrega resultados por (engine, arousal), promediando sobre valencia.
    
    Útil para analizar el efecto principal de arousal sobre las métricas,
    independientemente de valencia.
    
    Args:
        aggregated_results: Resultados ya agregados por (engine, V, A)
    
    Returns:
        Lista de diccionarios agregados por (engine, arousal) con mean ± std
    """
    if not aggregated_results:
        return []
    
    # Agrupar por (engine, arousal)
    groups = defaultdict(list)
    
    for result in aggregated_results:
        key = (result['engine'], result['arousal'])
        groups[key].append(result)
    
    # Métricas a promediar (usamos las medias ya calculadas)
    metrics = [
        'note_density', 'pitch_range', 'mean_velocity', 'mean_note_duration',
        'total_notes', 'total_duration_seconds', 'unique_pitches'
    ]
    
    # Calcular promedios sobre valencia
    arousal_aggregated = []
    
    for (engine, arousal), group_results in sorted(groups.items()):
        agg_row = {
            'engine': engine,
            'arousal': arousal,
            'num_valence_levels': len(group_results)
        }
        
        # Promediar las medias de cada métrica
        for metric in metrics:
            mean_key = f'{metric}_mean'
            values = [r[mean_key] for r in group_results]
            
            overall_mean = np.mean(values)
            overall_std = np.std(values, ddof=1) if len(values) > 1 else 0.0
            
            agg_row[f'{metric}_mean'] = round(overall_mean, 3)
            agg_row[f'{metric}_std'] = round(overall_std, 3)
        
        arousal_aggregated.append(agg_row)
    
    logger.info(f"Agregación por arousal: {len(arousal_aggregated)} combinaciones (A,engine)")
    
    return arousal_aggregated


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


def save_aggregated_csv(aggregated_results: List[Dict], output_dir: Path):
    """
    Guarda resultados agregados (mean ± std) en formato CSV.
    
    Args:
        aggregated_results: Lista de resultados agregados
        output_dir: Directorio donde guardar el archivo
    """
    csv_path = output_dir / 'benchmark_aggregated.csv'
    
    if not aggregated_results:
        logger.warning("No hay resultados agregados para exportar a CSV")
        return
    
    # Definir columnas ordenadas
    columns = ['engine', 'valence', 'arousal', 'num_samples']
    
    # Agregar columnas mean/std para cada métrica
    metrics = [
        'note_density', 'pitch_range', 'mean_velocity', 'mean_note_duration',
        'total_notes', 'total_duration_seconds', 'unique_pitches'
    ]
    
    for metric in metrics:
        columns.append(f'{metric}_mean')
        columns.append(f'{metric}_std')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(aggregated_results)
    
    logger.info(f"CSV agregado guardado: {csv_path} ({len(aggregated_results)} filas)")


def save_aggregated_latex(aggregated_results: List[Dict], output_dir: Path):
    """
    Genera tabla LaTeX con formato mean ± std para TFM.
    
    Args:
        aggregated_results: Lista de resultados agregados
        output_dir: Directorio donde guardar el archivo
    """
    tex_path = output_dir / 'benchmark_aggregated.tex'
    
    if not aggregated_results:
        logger.warning("No hay resultados agregados para exportar a LaTeX")
        return
    
    # Mapeo de nombres de engines
    engine_names = {
        'baseline': 'Baseline',
        'transformer_pretrained': 'Transformer'
    }
    
    # Generar LaTeX
    latex_content = []
    latex_content.append("% Tabla de benchmark agregado (mean ± std)")
    latex_content.append("% Generada automáticamente por run_benchmark_models.py")
    latex_content.append("")
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Comparación estadística de motores de generación musical}")
    latex_content.append("\\label{tab:benchmark_aggregated}")
    latex_content.append("\\resizebox{\\textwidth}{!}{%")
    latex_content.append("\\begin{tabular}{llccccccc}")
    latex_content.append("\\hline")
    latex_content.append("\\textbf{Motor} & \\textbf{V/A} & \\textbf{N} & \\textbf{Densidad} & \\textbf{Rango} & "
                        "\\textbf{Velocidad} & \\textbf{Duración} & \\textbf{Notas} & \\textbf{Únicas} \\\\")
    latex_content.append("& & & \\textbf{(n/s)} & \\textbf{(st)} & "
                        "\\textbf{(vel)} & \\textbf{(s)} & & \\\\")
    latex_content.append("\\hline")
    
    # Ordenar por (valencia, arousal, engine)
    sorted_results = sorted(aggregated_results, 
                           key=lambda r: (r['valence'], r['arousal'], r['engine']))
    
    prev_va = None
    
    for result in sorted_results:
        current_va = (result['valence'], result['arousal'])
        
        # Separador visual entre combinaciones VA
        if prev_va is not None and prev_va != current_va:
            latex_content.append("\\hline")
        
        va_label = f"({result['valence']:+.1f}, {result['arousal']:+.1f})"
        engine_label = engine_names.get(result['engine'], result['engine'])
        
        # Formato: mean ± std con máximo 2 decimales
        row = (
            f"{engine_label} & "
            f"{va_label} & "
            f"{result['num_samples']} & "
            f"${result['note_density_mean']:.2f} \\pm {result['note_density_std']:.2f}$ & "
            f"${result['pitch_range_mean']:.0f} \\pm {result['pitch_range_std']:.0f}$ & "
            f"${result['mean_velocity_mean']:.0f} \\pm {result['mean_velocity_std']:.0f}$ & "
            f"${result['mean_note_duration_mean']:.2f} \\pm {result['mean_note_duration_std']:.2f}$ & "
            f"${result['total_notes_mean']:.0f} \\pm {result['total_notes_std']:.0f}$ & "
            f"${result['unique_pitches_mean']:.0f} \\pm {result['unique_pitches_std']:.0f}$ \\\\"
        )
        latex_content.append(row)
        prev_va = current_va
    
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("}")
    latex_content.append("\\end{table}")
    
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    logger.info(f"Tabla LaTeX agregada guardada: {tex_path}")


def save_arousal_summary_latex(arousal_results: List[Dict], output_dir: Path):
    """
    Genera tabla LaTeX resumida por nivel de arousal.
    
    Args:
        arousal_results: Resultados agregados por (engine, arousal)
        output_dir: Directorio donde guardar el archivo
    """
    tex_path = output_dir / 'benchmark_arousal_summary.tex'
    
    if not arousal_results:
        logger.warning("No hay resultados de arousal para exportar")
        return
    
    engine_names = {
        'baseline': 'Baseline',
        'transformer_pretrained': 'Transformer'
    }
    
    latex_content = []
    latex_content.append("% Tabla resumen por nivel de arousal")
    latex_content.append("% Generada automáticamente por run_benchmark_models.py")
    latex_content.append("")
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Efecto del arousal en métricas musicales (promedio sobre valencia)}")
    latex_content.append("\\label{tab:arousal_summary}")
    latex_content.append("\\begin{tabular}{llcccc}")
    latex_content.append("\\hline")
    latex_content.append("\\textbf{Motor} & \\textbf{Arousal} & \\textbf{Densidad} & \\textbf{Velocidad} & "
                        "\\textbf{Duración} & \\textbf{Notas} \\\\")
    latex_content.append("& & \\textbf{(n/s)} & \\textbf{(vel)} & \\textbf{(s)} & \\\\")
    latex_content.append("\\hline")
    
    sorted_results = sorted(arousal_results, key=lambda r: (r['arousal'], r['engine']))
    
    prev_arousal = None
    
    for result in sorted_results:
        current_arousal = result['arousal']
        
        if prev_arousal is not None and prev_arousal != current_arousal:
            latex_content.append("\\hline")
        
        engine_label = engine_names.get(result['engine'], result['engine'])
        
        row = (
            f"{engine_label} & "
            f"{current_arousal:+.1f} & "
            f"${result['note_density_mean']:.2f} \\pm {result['note_density_std']:.2f}$ & "
            f"${result['mean_velocity_mean']:.0f} \\pm {result['mean_velocity_std']:.0f}$ & "
            f"${result['mean_note_duration_mean']:.2f} \\pm {result['mean_note_duration_std']:.2f}$ & "
            f"${result['total_notes_mean']:.0f} \\pm {result['total_notes_std']:.0f}$ \\\\"
        )
        latex_content.append(row)
        prev_arousal = current_arousal
    
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    logger.info(f"Tabla resumen de arousal guardada: {tex_path}")


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
        '--num_seeds',
        type=int,
        default=1,
        help='Número de seeds por combinación (V,A) (default: 1, recomendado: 5)'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=None,
        help='Tokens fijos para Transformer (default: None=auto, recomendado: 512)'
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
            num_seeds=args.num_seeds,
            max_new_tokens=args.max_tokens,
            output_dir=args.output_dir,
            length_bars=args.length_bars,
            verbose=args.verbose
        )
        
        # Determinar output_dir para guardar resultados
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'results'
        
        # Guardar resultados crudos individuales
        save_results_json(results, output_dir)
        save_results_csv(results, output_dir)
        save_results_latex(results, output_dir)
        
        # Si hay múltiples seeds, generar agregaciones estadísticas
        if args.num_seeds > 1:
            logger.info("=" * 60)
            logger.info("Generando agregaciones estadísticas...")
            logger.info("=" * 60)
            
            # Agregar por (engine, V, A)
            aggregated = aggregate_results(results)
            
            # Guardar agregaciones
            save_aggregated_csv(aggregated, output_dir)
            save_aggregated_latex(aggregated, output_dir)
            
            # Agregar por (engine, A) - resumen de arousal
            arousal_summary = aggregate_by_arousal(aggregated)
            save_arousal_summary_latex(arousal_summary, output_dir)
            
            logger.info(f"Agregaciones guardadas en: {output_dir}")
        
        # Mostrar resumen
        print_summary(results)
        
        logger.info("¡Benchmark completado exitosamente!")
        logger.info(f"Resultados guardados en: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error ejecutando benchmark: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
