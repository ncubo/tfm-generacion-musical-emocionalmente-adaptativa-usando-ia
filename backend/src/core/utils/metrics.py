"""
Módulo de instrumentación y medición de rendimiento.

Proporciona herramientas para medir latencias en el pipeline emocional
y de generación musical, con el objetivo de evaluar el rendimiento del
sistema de forma reproducible y defendible académicamente.
"""

import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from collections import defaultdict
import statistics


class PerformanceMetrics:
    """
    Gestor de métricas de rendimiento del sistema.
    
    Permite medir tiempos de ejecución de diferentes etapas del pipeline
    y almacenar los resultados para posterior análisis estadístico.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Inicializa el gestor de métricas.
        
        Args:
            output_dir: Directorio donde guardar los resultados
        """
        self.measurements: Dict[str, List[float]] = defaultdict(list)
        self.metadata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.output_dir = output_dir or Path('metrics')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    @contextmanager
    def measure(self, stage_name: str, metadata: Optional[Dict] = None):
        """
        Context manager para medir el tiempo de ejecución de una etapa.
        
        Args:
            stage_name: Nombre de la etapa a medir
            metadata: Información adicional sobre la medición
            
        Yields:
            Diccionario donde se guardará el tiempo medido
            
        Example:
            with metrics.measure('emotion_detection') as timing:
                result = detect_emotion()
            print(f"Tardó {timing['duration']} segundos")
        """
        start_time = time.perf_counter()
        timing_info = {'stage': stage_name}
        
        try:
            yield timing_info
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            timing_info['duration'] = duration
            timing_info['timestamp'] = datetime.now().isoformat()
            
            # Guardar la medición
            self.measurements[stage_name].append(duration)
            
            # Guardar metadata si se proporciona
            if metadata:
                timing_info.update(metadata)
            self.metadata[stage_name].append(timing_info)
    
    def measure_function(self, stage_name: str):
        """
        Decorador para medir automáticamente funciones.
        
        Args:
            stage_name: Nombre de la etapa a medir
            
        Returns:
            Decorador de función
            
        Example:
            @metrics.measure_function('midi_generation')
            def generate_midi():
                # código de generación
                pass
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure(stage_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_statistics(self, stage_name: Optional[str] = None) -> Dict:
        """
        Calcula estadísticas sobre las mediciones realizadas.
        
        Args:
            stage_name: Etapa específica (None para todas)
            
        Returns:
            Diccionario con estadísticas por etapa
        """
        if stage_name:
            stages = {stage_name: self.measurements[stage_name]}
        else:
            stages = self.measurements
            
        stats = {}
        for name, times in stages.items():
            if not times:
                continue
                
            stats[name] = {
                'count': len(times),
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0.0,
                'min': min(times),
                'max': max(times),
                'total': sum(times)
            }
            
        return stats
    
    def save_to_csv(self, filename: Optional[str] = None):
        """
        Guarda las mediciones en formato CSV.
        
        Args:
            filename: Nombre del archivo (generado automáticamente si None)
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'metrics_{timestamp}.csv'
            
        filepath = self.output_dir / filename
        
        # Preparar datos para CSV
        rows = []
        for stage_name, metadata_list in self.metadata.items():
            for entry in metadata_list:
                row = {
                    'stage': stage_name,
                    'duration': entry['duration'],
                    'timestamp': entry['timestamp']
                }
                # Añadir metadata adicional
                for key, value in entry.items():
                    if key not in ['stage', 'duration', 'timestamp']:
                        row[key] = value
                rows.append(row)
        
        if not rows:
            print("No hay mediciones para guardar")
            return
            
        # Escribir CSV
        fieldnames = list(rows[0].keys())
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"[OK] Métricas guardadas en: {filepath}")
        
    def save_to_json(self, filename: Optional[str] = None):
        """
        Guarda las mediciones y estadísticas en formato JSON.
        
        Args:
            filename: Nombre del archivo (generado automáticamente si None)
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'metrics_{timestamp}.json'
            
        filepath = self.output_dir / filename
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'raw_measurements': dict(self.measurements),
            'metadata': dict(self.metadata)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"[OK] Métricas guardadas en: {filepath}")
    
    def print_summary(self):
        """
        Imprime un resumen de las estadísticas por consola.
        """
        stats = self.get_statistics()
        
        if not stats:
            print("No hay mediciones disponibles")
            return
            
        print("\n" + "="*70)
        print("RESUMEN DE MÉTRICAS DE RENDIMIENTO")
        print("="*70)
        
        for stage_name, stage_stats in stats.items():
            print(f"\n[{stage_name.upper()}]")
            print(f"  Mediciones: {stage_stats['count']}")
            print(f"  Media:      {stage_stats['mean']*1000:.2f} ms")
            print(f"  Mediana:    {stage_stats['median']*1000:.2f} ms")
            print(f"  Desv. Est.: {stage_stats['stdev']*1000:.2f} ms")
            print(f"  Mínimo:     {stage_stats['min']*1000:.2f} ms")
            print(f"  Máximo:     {stage_stats['max']*1000:.2f} ms")
            
        # Calcular latencia total si existen ambas etapas
        if 'emotion_detection' in stats and 'midi_generation' in stats:
            total_mean = stats['emotion_detection']['mean'] + stats['midi_generation']['mean']
            print(f"\n[LATENCIA TOTAL - Emoción + MIDI]")
            print(f"  Media:      {total_mean*1000:.2f} ms")
            
        print("\n" + "="*70 + "\n")
    
    def clear(self):
        """Limpia todas las mediciones almacenadas."""
        self.measurements.clear()
        self.metadata.clear()


# Instancia global para uso en la aplicación
_global_metrics = None


def get_metrics(output_dir: Optional[Path] = None) -> PerformanceMetrics:
    """
    Obtiene la instancia global de métricas.
    
    Args:
        output_dir: Directorio de salida (solo para primera inicialización)
        
    Returns:
        Instancia de PerformanceMetrics
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = PerformanceMetrics(output_dir)
    return _global_metrics


def reset_metrics():
    """Reinicia la instancia global de métricas."""
    global _global_metrics
    _global_metrics = None
