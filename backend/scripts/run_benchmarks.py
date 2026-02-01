#!/usr/bin/env python3
"""
Script de benchmarking para evaluación de rendimiento del sistema.

Ejecuta múltiples mediciones de los endpoints /emotion y /generate-midi
para obtener estadísticas representativas de latencia del sistema completo.

Uso:
    python run_benchmarks.py [--iterations N] [--url URL]

Ejemplo:
    python run_benchmarks.py --iterations 30 --url http://localhost:5000
"""

import argparse
import requests
import time
import json
from pathlib import Path
from datetime import datetime
from statistics import mean, median, stdev


class BenchmarkRunner:
    """Ejecutor de benchmarks para el sistema de generación musical."""
    
    def __init__(self, base_url: str = 'http://localhost:5000', iterations: int = 25):
        """
        Inicializa el runner de benchmarks.
        
        Args:
            base_url: URL base del servidor Flask
            iterations: Número de iteraciones por endpoint
        """
        self.base_url = base_url.rstrip('/')
        self.iterations = iterations
        self.results = {
            'emotion': [],
            'generate_midi': [],
            'total_pipeline': []
        }
        
    def check_server(self) -> bool:
        """
        Verifica que el servidor esté disponible.
        
        Returns:
            True si el servidor responde, False en caso contrario
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def benchmark_emotion(self) -> dict:
        """
        Ejecuta benchmark del endpoint /emotion.
        
        Returns:
            Diccionario con tiempo de respuesta y datos de la respuesta
        """
        start_time = time.perf_counter()
        
        try:
            response = requests.post(
                f"{self.base_url}/emotion",
                timeout=30
            )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'duration': duration,
                    'emotion': data.get('emotion'),
                    'valence': data.get('valence'),
                    'arousal': data.get('arousal')
                }
            else:
                return {
                    'success': False,
                    'duration': duration,
                    'error': f"HTTP {response.status_code}"
                }
                
        except requests.RequestException as e:
            end_time = time.perf_counter()
            return {
                'success': False,
                'duration': end_time - start_time,
                'error': str(e)
            }
    
    def benchmark_generate_midi(self) -> dict:
        """
        Ejecuta benchmark del endpoint /generate-midi.
        
        Returns:
            Diccionario con tiempo de respuesta y datos de la respuesta
        """
        start_time = time.perf_counter()
        
        try:
            response = requests.post(
                f"{self.base_url}/generate-midi",
                timeout=30
            )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'duration': duration,
                    'emotion': data.get('emotion'),
                    'valence': data.get('valence'),
                    'arousal': data.get('arousal'),
                    'midi_path': data.get('midi_path')
                }
            else:
                return {
                    'success': False,
                    'duration': duration,
                    'error': f"HTTP {response.status_code}"
                }
                
        except requests.RequestException as e:
            end_time = time.perf_counter()
            return {
                'success': False,
                'duration': end_time - start_time,
                'error': str(e)
            }
    
    def run_benchmarks(self):
        """
        Ejecuta todos los benchmarks según el número de iteraciones configurado.
        """
        print(f"\n{'='*70}")
        print(f"BENCHMARK DE RENDIMIENTO - {self.iterations} iteraciones por endpoint")
        print(f"{'='*70}\n")
        
        # Verificar servidor
        print("[INFO] Verificando servidor...")
        if not self.check_server():
            print("[ERROR] El servidor no está disponible en", self.base_url)
            print("        Asegúrate de que Flask esté ejecutándose.")
            return
        print("[OK] Servidor disponible\n")
        
        # Benchmark endpoint /emotion
        print(f"[BENCHMARK] Ejecutando {self.iterations} mediciones de /emotion...")
        for i in range(self.iterations):
            result = self.benchmark_emotion()
            if result['success']:
                self.results['emotion'].append(result['duration'])
                print(f"  [{i+1}/{self.iterations}] {result['duration']*1000:.2f} ms - {result['emotion']}")
            else:
                print(f"  [{i+1}/{self.iterations}] ERROR: {result.get('error')}")
            
            # Pequeña pausa entre requests para no saturar
            time.sleep(0.1)
        
        print(f"\n[BENCHMARK] Ejecutando {self.iterations} mediciones de /generate-midi...")
        for i in range(self.iterations):
            result = self.benchmark_generate_midi()
            if result['success']:
                self.results['generate_midi'].append(result['duration'])
                self.results['total_pipeline'].append(result['duration'])
                print(f"  [{i+1}/{self.iterations}] {result['duration']*1000:.2f} ms - {result['emotion']}")
            else:
                print(f"  [{i+1}/{self.iterations}] ERROR: {result.get('error')}")
            
            # Pausa entre requests
            time.sleep(0.1)
        
        print("\n[OK] Benchmarks completados\n")
    
    def calculate_statistics(self) -> dict:
        """
        Calcula estadísticas sobre los resultados obtenidos.
        
        Returns:
            Diccionario con estadísticas por endpoint
        """
        stats = {}
        
        for endpoint, times in self.results.items():
            if not times:
                continue
                
            stats[endpoint] = {
                'count': len(times),
                'mean_ms': mean(times) * 1000,
                'median_ms': median(times) * 1000,
                'stdev_ms': stdev(times) * 1000 if len(times) > 1 else 0.0,
                'min_ms': min(times) * 1000,
                'max_ms': max(times) * 1000,
                'p95_ms': self._percentile(times, 95) * 1000,
                'p99_ms': self._percentile(times, 99) * 1000
            }
            
        return stats
    
    def _percentile(self, data: list, percentile: int) -> float:
        """Calcula el percentil especificado."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def print_results(self):
        """Imprime resultados en consola de forma legible."""
        stats = self.calculate_statistics()
        
        print(f"{'='*70}")
        print("RESULTADOS DE BENCHMARKS")
        print(f"{'='*70}\n")
        
        for endpoint, endpoint_stats in stats.items():
            print(f"[{endpoint.upper().replace('_', ' ')}]")
            print(f"  Mediciones:      {endpoint_stats['count']}")
            print(f"  Media:           {endpoint_stats['mean_ms']:.2f} ms")
            print(f"  Mediana:         {endpoint_stats['median_ms']:.2f} ms")
            print(f"  Desv. Estándar:  {endpoint_stats['stdev_ms']:.2f} ms")
            print(f"  Mínimo:          {endpoint_stats['min_ms']:.2f} ms")
            print(f"  Máximo:          {endpoint_stats['max_ms']:.2f} ms")
            print(f"  Percentil 95:    {endpoint_stats['p95_ms']:.2f} ms")
            print(f"  Percentil 99:    {endpoint_stats['p99_ms']:.2f} ms")
            print()
        
        # Análisis de tiempo real
        if 'total_pipeline' in stats:
            avg_latency = stats['total_pipeline']['mean_ms']
            is_realtime = avg_latency < 1000  # < 1 segundo
            
            print(f"[EVALUACION DE TIEMPO REAL]")
            print(f"  Latencia promedio: {avg_latency:.2f} ms")
            print(f"  ¿Tiempo real?:     {'SI' if is_realtime else 'NO'}")
            print(f"  Criterio:          < 1000 ms para interacción aceptable")
            print()
        
        print(f"{'='*70}\n")
    
    def save_results(self, output_dir: str = 'metrics'):
        """
        Guarda los resultados en archivos JSON y CSV.
        
        Args:
            output_dir: Directorio donde guardar los resultados
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar JSON
        json_file = output_path / f'benchmark_{timestamp}.json'
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'base_url': self.base_url,
            'iterations': self.iterations,
            'statistics': self.calculate_statistics(),
            'raw_results': self.results
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Resultados guardados en: {json_file}")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Benchmark de rendimiento del sistema de generación musical'
    )
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=25,
        help='Número de iteraciones por endpoint (default: 25)'
    )
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:5000',
        help='URL base del servidor (default: http://localhost:5000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='metrics',
        help='Directorio de salida para resultados (default: metrics)'
    )
    
    args = parser.parse_args()
    
    # Ejecutar benchmarks
    runner = BenchmarkRunner(base_url=args.url, iterations=args.iterations)
    runner.run_benchmarks()
    runner.print_results()
    runner.save_results(output_dir=args.output)


if __name__ == '__main__':
    main()
