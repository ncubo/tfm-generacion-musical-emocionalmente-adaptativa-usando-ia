#!/usr/bin/env python3
"""
Script de análisis y visualización de métricas de rendimiento.

Genera reportes y visualizaciones a partir de los datos de benchmarking
recolectados.

Uso:
    python analyze_metrics.py [--input INPUT_FILE] [--output OUTPUT_DIR]

Ejemplo:
    python analyze_metrics.py --input metrics/benchmark_20260201_120000.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


class MetricsAnalyzer:
    """Analizador de métricas de rendimiento."""
    
    def __init__(self, input_file: str):
        """
        Inicializa el analizador.
        
        Args:
            input_file: Archivo JSON con resultados de benchmarking
        """
        self.input_file = Path(input_file)
        self.data = self._load_data()
        
    def _load_data(self) -> dict:
        """Carga los datos del archivo JSON."""
        if not self.input_file.exists():
            print(f"[ERROR] No se encuentra el archivo {self.input_file}")
            sys.exit(1)
            
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_text_report(self, output_dir: str = 'metrics'):
        """
        Genera un reporte de texto detallado.
        
        Args:
            output_dir: Directorio donde guardar el reporte
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_path / f'reporte_rendimiento_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DE EVALUACIÓN DE RENDIMIENTO\n")
            f.write("Sistema de Generación Musical Emocionalmente Adaptativa\n")
            f.write("="*80 + "\n\n")
            
            # Información general
            f.write("1. CONFIGURACIÓN DEL BENCHMARK\n")
            f.write("-" * 80 + "\n")
            f.write(f"Fecha de ejecución:  {self.data['timestamp']}\n")
            f.write(f"URL del servidor:    {self.data['base_url']}\n")
            f.write(f"Iteraciones:         {self.data['iterations']}\n\n")
            
            # Estadísticas por endpoint
            f.write("2. RESULTADOS POR COMPONENTE\n")
            f.write("-" * 80 + "\n\n")
            
            stats = self.data['statistics']
            
            for endpoint, endpoint_stats in stats.items():
                endpoint_name = endpoint.replace('_', ' ').title()
                f.write(f"2.{list(stats.keys()).index(endpoint)+1} {endpoint_name}\n")
                f.write(f"    Número de mediciones: {endpoint_stats['count']}\n")
                f.write(f"    Media:                {endpoint_stats['mean_ms']:.2f} ms\n")
                f.write(f"    Mediana:              {endpoint_stats['median_ms']:.2f} ms\n")
                f.write(f"    Desviación estándar:  {endpoint_stats['stdev_ms']:.2f} ms\n")
                f.write(f"    Valor mínimo:         {endpoint_stats['min_ms']:.2f} ms\n")
                f.write(f"    Valor máximo:         {endpoint_stats['max_ms']:.2f} ms\n")
                f.write(f"    Percentil 95:         {endpoint_stats['p95_ms']:.2f} ms\n")
                f.write(f"    Percentil 99:         {endpoint_stats['p99_ms']:.2f} ms\n\n")
            
            # Análisis de tiempo real
            f.write("3. ANÁLISIS DE CAPACIDAD DE TIEMPO REAL\n")
            f.write("-" * 80 + "\n")
            
            if not stats:
                f.write("No hay estadísticas disponibles para analizar.\n\n")
            elif 'total_pipeline' in stats:
                avg_latency = stats['total_pipeline']['mean_ms']
                max_latency = stats['total_pipeline']['max_ms']
                p95_latency = stats['total_pipeline']['p95_ms']
                
                f.write(f"Latencia media del pipeline completo: {avg_latency:.2f} ms\n")
                f.write(f"Latencia máxima observada:            {max_latency:.2f} ms\n")
                f.write(f"Percentil 95 de latencia:             {p95_latency:.2f} ms\n\n")
                
                # Criterios de evaluación
                f.write("Criterios de evaluación:\n")
                
                realtime_threshold = 1000  # 1 segundo
                interactive_threshold = 500  # 500 ms
                
                if avg_latency < interactive_threshold:
                    verdict = "EXCELENTE - Interacción fluida"
                elif avg_latency < realtime_threshold:
                    verdict = "BUENO - Tiempo real aceptable"
                else:
                    verdict = "MEJORABLE - Latencia perceptible"
                
                f.write(f"  • Latencia < 500 ms:  Interacción fluida (excelente)\n")
                f.write(f"  • Latencia < 1000 ms: Tiempo real aceptable (bueno)\n")
                f.write(f"  • Latencia > 1000 ms: Latencia perceptible (mejorable)\n\n")
                f.write(f"VEREDICTO: {verdict}\n\n")
            
            # Conclusiones
            f.write("4. CONCLUSIONES\n")
            f.write("-" * 80 + "\n")
            
            if 'total_pipeline' in stats:
                avg_latency = stats['total_pipeline']['mean_ms']
                
                if avg_latency < 500:
                    f.write("El sistema demuestra capacidad de operación en tiempo real con latencias\n")
                    f.write("muy bajas, permitiendo una interacción fluida y natural. La generación\n")
                    f.write("musical se produce prácticamente de forma instantánea tras la detección\n")
                    f.write("emocional, cumpliendo con los requisitos de un sistema adaptativo.\n")
                elif avg_latency < 1000:
                    f.write("El sistema opera en tiempo real con latencias aceptables. Aunque existe\n")
                    f.write("un pequeño delay perceptible, este no compromete significativamente la\n")
                    f.write("experiencia de usuario y es apropiado para una aplicación de generación\n")
                    f.write("musical adaptativa.\n")
                else:
                    f.write("El sistema presenta latencias que pueden ser perceptibles en algunos casos.\n")
                    f.write("Se recomienda optimización adicional para mejorar la respuesta en tiempo real.\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Fin del reporte\n")
            f.write("="*80 + "\n")
        
        print(f"[OK] Reporte generado en: {report_file}")
        return report_file
    
    def generate_latex_table(self, output_dir: str = 'metrics'):
        """
        Genera tabla en formato LaTeX para inclusión en memoria.
        
        Args:
            output_dir: Directorio donde guardar la tabla
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        latex_file = output_path / f'tabla_latex_{timestamp}.tex'
        
        stats = self.data['statistics']
        
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write("% Tabla de métricas de rendimiento\n")
            f.write("% Generada automáticamente\n\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Métricas de rendimiento del sistema}\n")
            f.write("\\label{tab:metricas_rendimiento}\n")
            f.write("\\begin{tabular}{lrrr}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Métrica} & \\textbf{Emoción} & \\textbf{MIDI} & \\textbf{Total} \\\\\n")
            f.write("\\hline\n")
            
            # Fila de media
            emotion_mean = stats.get('emotion', {}).get('mean_ms', 0)
            midi_mean = stats.get('generate_midi', {}).get('mean_ms', 0)
            total_mean = stats.get('total_pipeline', {}).get('mean_ms', 0)
            
            f.write(f"Media (ms) & {emotion_mean:.2f} & {midi_mean:.2f} & {total_mean:.2f} \\\\\n")
            
            # Fila de mediana
            emotion_median = stats.get('emotion', {}).get('median_ms', 0)
            midi_median = stats.get('generate_midi', {}).get('median_ms', 0)
            total_median = stats.get('total_pipeline', {}).get('median_ms', 0)
            
            f.write(f"Mediana (ms) & {emotion_median:.2f} & {midi_median:.2f} & {total_median:.2f} \\\\\n")
            
            # Fila de desviación estándar
            emotion_stdev = stats.get('emotion', {}).get('stdev_ms', 0)
            midi_stdev = stats.get('generate_midi', {}).get('stdev_ms', 0)
            total_stdev = stats.get('total_pipeline', {}).get('stdev_ms', 0)
            
            f.write(f"Desv. Est. (ms) & {emotion_stdev:.2f} & {midi_stdev:.2f} & {total_stdev:.2f} \\\\\n")
            
            # Fila de percentil 95
            emotion_p95 = stats.get('emotion', {}).get('p95_ms', 0)
            midi_p95 = stats.get('generate_midi', {}).get('p95_ms', 0)
            total_p95 = stats.get('total_pipeline', {}).get('p95_ms', 0)
            
            f.write(f"P95 (ms) & {emotion_p95:.2f} & {midi_p95:.2f} & {total_p95:.2f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"[OK] Tabla LaTeX generada en: {latex_file}")
        return latex_file
    
    def generate_markdown_table(self, output_dir: str = 'metrics'):
        """
        Genera tabla en formato Markdown.
        
        Args:
            output_dir: Directorio donde guardar la tabla
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_file = output_path / f'tabla_markdown_{timestamp}.md'
        
        stats = self.data['statistics']
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Métricas de Rendimiento del Sistema\n\n")
            f.write("## Estadísticas de Latencia\n\n")
            
            f.write("| Métrica | Detección Emocional | Generación MIDI | Pipeline Completo |\n")
            f.write("|---------|--------------------:|----------------:|------------------:|\n")
            
            # Fila de media
            emotion_mean = stats.get('emotion', {}).get('mean_ms', 0)
            midi_mean = stats.get('generate_midi', {}).get('mean_ms', 0)
            total_mean = stats.get('total_pipeline', {}).get('mean_ms', 0)
            
            f.write(f"| Media (ms) | {emotion_mean:.2f} | {midi_mean:.2f} | {total_mean:.2f} |\n")
            
            # Fila de mediana
            emotion_median = stats.get('emotion', {}).get('median_ms', 0)
            midi_median = stats.get('generate_midi', {}).get('median_ms', 0)
            total_median = stats.get('total_pipeline', {}).get('median_ms', 0)
            
            f.write(f"| Mediana (ms) | {emotion_median:.2f} | {midi_median:.2f} | {total_median:.2f} |\n")
            
            # Fila de desviación estándar
            emotion_stdev = stats.get('emotion', {}).get('stdev_ms', 0)
            midi_stdev = stats.get('generate_midi', {}).get('stdev_ms', 0)
            total_stdev = stats.get('total_pipeline', {}).get('stdev_ms', 0)
            
            f.write(f"| Desv. Estándar (ms) | {emotion_stdev:.2f} | {midi_stdev:.2f} | {total_stdev:.2f} |\n")
            
            # Fila de mínimo
            emotion_min = stats.get('emotion', {}).get('min_ms', 0)
            midi_min = stats.get('generate_midi', {}).get('min_ms', 0)
            total_min = stats.get('total_pipeline', {}).get('min_ms', 0)
            
            f.write(f"| Mínimo (ms) | {emotion_min:.2f} | {midi_min:.2f} | {total_min:.2f} |\n")
            
            # Fila de máximo
            emotion_max = stats.get('emotion', {}).get('max_ms', 0)
            midi_max = stats.get('generate_midi', {}).get('max_ms', 0)
            total_max = stats.get('total_pipeline', {}).get('max_ms', 0)
            
            f.write(f"| Máximo (ms) | {emotion_max:.2f} | {midi_max:.2f} | {total_max:.2f} |\n")
            
            # Fila de percentil 95
            emotion_p95 = stats.get('emotion', {}).get('p95_ms', 0)
            midi_p95 = stats.get('generate_midi', {}).get('p95_ms', 0)
            total_p95 = stats.get('total_pipeline', {}).get('p95_ms', 0)
            
            f.write(f"| Percentil 95 (ms) | {emotion_p95:.2f} | {midi_p95:.2f} | {total_p95:.2f} |\n")
            
            f.write("\n## Interpretación\n\n")
            
            if total_mean < 500:
                f.write("**Excelente**: El sistema opera con latencias muy bajas, "
                       "permitiendo interacción fluida en tiempo real.\n")
            elif total_mean < 1000:
                f.write("**Bueno**: El sistema cumple con los requisitos de tiempo real "
                       "con latencias aceptables para una aplicación interactiva.\n")
            else:
                f.write("**Mejorable**: Las latencias son perceptibles. "
                       "Se recomienda optimización adicional.\n")
        
        print(f"[OK] Tabla Markdown generada en: {md_file}")
        return md_file
    
    def print_summary(self):
        """Imprime resumen en consola."""
        stats = self.data['statistics']
        
        print("\n" + "="*70)
        print("RESUMEN DE ANÁLISIS DE MÉTRICAS")
        print("="*70 + "\n")
        
        for endpoint, endpoint_stats in stats.items():
            print(f"[{endpoint.replace('_', ' ').upper()}]")
            print(f"  Media:     {endpoint_stats['mean_ms']:.2f} ms")
            print(f"  Mediana:   {endpoint_stats['median_ms']:.2f} ms")
            print(f"  Desv.Est.: {endpoint_stats['stdev_ms']:.2f} ms")
            print(f"  P95:       {endpoint_stats['p95_ms']:.2f} ms\n")
        
        print("="*70 + "\n")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Análisis de métricas de rendimiento'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Archivo JSON con resultados de benchmarking'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='metrics',
        help='Directorio de salida (default: metrics)'
    )
    
    args = parser.parse_args()
    
    # Analizar métricas
    analyzer = MetricsAnalyzer(input_file=args.input)
    analyzer.print_summary()
    
    # Generar reportes
    print("Generando reportes...")
    analyzer.generate_text_report(output_dir=args.output)
    analyzer.generate_latex_table(output_dir=args.output)
    analyzer.generate_markdown_table(output_dir=args.output)
    
    print("\n[OK] Análisis completado")


if __name__ == '__main__':
    main()
