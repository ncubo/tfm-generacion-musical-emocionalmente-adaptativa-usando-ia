#!/usr/bin/env python3
"""
Script de demostración rápida del sistema de métricas.

Ejecuta un benchmark pequeño y muestra los resultados inmediatamente.
Útil para verificar que todo funciona correctamente.

Uso:
    python demo_metrics.py
"""

import sys
import time
from pathlib import Path

# Añadir el directorio src al path para poder importar los módulos
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.utils.metrics import PerformanceMetrics


def simulate_emotion_detection():
    """Simula la detección emocional."""
    time.sleep(0.15)  # Simular ~150ms
    return {'emotion': 'happy', 'valence': 0.7, 'arousal': 0.6}


def simulate_midi_generation():
    """Simula la generación MIDI."""
    time.sleep(0.35)  # Simular ~350ms
    return {'midi_path': '/path/to/output.mid'}


def main():
    print("\n" + "="*70)
    print("DEMO - Sistema de Métricas de Rendimiento")
    print("="*70 + "\n")
    
    # Crear instancia de métricas
    metrics = PerformanceMetrics(output_dir=Path('metrics_demo'))
    
    print("Ejecutando 10 simulaciones de cada operación...\n")
    
    # Simular múltiples ejecuciones
    for i in range(10):
        # Medir detección emocional
        with metrics.measure('emotion_detection', metadata={'iteration': i+1}):
            simulate_emotion_detection()
        
        # Medir generación MIDI
        with metrics.measure('midi_generation', metadata={'iteration': i+1}):
            simulate_midi_generation()
        
        print(f"  Iteración {i+1}/10 completada")
    
    print("\n" + "="*70)
    print("Resultados:")
    print("="*70 + "\n")
    
    # Mostrar estadísticas
    metrics.print_summary()
    
    # Guardar resultados
    print("Guardando resultados...")
    metrics.save_to_csv('demo_metrics.csv')
    metrics.save_to_json('demo_metrics.json')
    
    print("\n[OK] Demo completada. Archivos guardados en metrics_demo/\n")


if __name__ == '__main__':
    main()
