"""
Script de análisis cuantitativo de estabilidad temporal.

Este script mide métricas objetivas de estabilidad del sistema emocional,
comparando diferentes configuraciones de estabilización.

Métricas calculadas:
- Frecuencia de cambios de emoción (cambios/segundo)
- Varianza de valores V/A
- Rate of change de V/A (|Δ| promedio)
- Tiempo de convergencia

Uso:
    python backend/scripts/analyze_stability.py --duration 30
    
Output:
    - Tabla comparativa de métricas
    - Gráficas de evolución temporal (opcional)
"""

import sys
import os
import argparse
from typing import Dict, List
import statistics

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.camera import WebcamCapture
from core.emotion import DeepFaceEmotionDetector
from core.pipeline import EmotionPipeline


def calculate_metrics(data: List[Dict]) -> Dict:
    """
    Calcula métricas de estabilidad a partir de datos capturados.
    
    Args:
        data: Lista de resultados del pipeline
        
    Returns:
        Diccionario con métricas calculadas
    """
    if not data or len(data) < 2:
        return {}
    
    # Extraer series temporales
    emotions = [d['emotion'] for d in data]
    valences = [d['valence'] for d in data]
    arousals = [d['arousal'] for d in data]
    
    # 1. Frecuencia de cambios de emoción
    emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
    
    # 2. Varianza de V/A
    valence_variance = statistics.variance(valences) if len(valences) > 1 else 0.0
    arousal_variance = statistics.variance(arousals) if len(arousals) > 1 else 0.0
    
    # 3. Rate of change (promedio de |Δ|)
    valence_deltas = [abs(valences[i] - valences[i-1]) for i in range(1, len(valences))]
    arousal_deltas = [abs(arousals[i] - arousals[i-1]) for i in range(1, len(arousals))]
    
    valence_roc = statistics.mean(valence_deltas) if valence_deltas else 0.0
    arousal_roc = statistics.mean(arousal_deltas) if arousal_deltas else 0.0
    
    # 4. Estabilidad de emoción (% de frames con misma emoción que anterior)
    emotion_stability = (len(emotions) - emotion_changes) / len(emotions) * 100
    
    return {
        'n_samples': len(data),
        'emotion_changes': emotion_changes,
        'emotion_stability_pct': emotion_stability,
        'valence_variance': valence_variance,
        'arousal_variance': arousal_variance,
        'valence_roc': valence_roc,
        'arousal_roc': arousal_roc,
        'avg_valence': statistics.mean(valences),
        'avg_arousal': statistics.mean(arousals)
    }


def capture_data(pipeline: EmotionPipeline, duration: int, label: str) -> List[Dict]:
    """
    Captura datos del pipeline durante un tiempo determinado.
    
    Args:
        pipeline: Pipeline a evaluar
        duration: Duración en segundos
        label: Etiqueta descriptiva
        
    Returns:
        Lista de resultados capturados
    """
    print(f"\nCapturando datos para: {label}")
    print(f"Duración: {duration} segundos")
    print("Mantén una expresión neutral y luego cambia a otra emoción...\n")
    
    data = []
    import time
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        result = pipeline.step()
        data.append(result)
        frame_count += 1
        
        # Mostrar progreso cada segundo
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            print(f"  Progreso: {elapsed:.1f}s / {duration}s (quedan {remaining:.1f}s)")
    
    print(f"[OK] Capturados {len(data)} samples")
    return data


def print_comparison_table(results: Dict[str, Dict]):
    """
    Imprime tabla comparativa de resultados.
    
    Args:
        results: Diccionario con métricas por configuración
    """
    print("\n" + "=" * 90)
    print("COMPARACIÓN DE ESTABILIDAD TEMPORAL")
    print("=" * 90)
    print()
    
    # Encabezados
    print(f"{'Métrica':<35} | {'Sin Estabilización':<20} | {'Con Estabilización':<20}")
    print("-" * 90)
    
    # Comparar métricas
    if 'minimal' in results and 'optimized' in results:
        minimal = results['minimal']
        optimized = results['optimized']
        
        print(f"{'Samples capturados':<35} | {minimal['n_samples']:<20} | {optimized['n_samples']:<20}")
        print(f"{'Cambios de emoción (total)':<35} | {minimal['emotion_changes']:<20} | {optimized['emotion_changes']:<20}")
        print(f"{'Estabilidad emoción (%)':<35} | {minimal['emotion_stability_pct']:<20.1f} | {optimized['emotion_stability_pct']:<20.1f}")
        print()
        print(f"{'Varianza Valencia':<35} | {minimal['valence_variance']:<20.4f} | {optimized['valence_variance']:<20.4f}")
        print(f"{'Varianza Arousal':<35} | {minimal['arousal_variance']:<20.4f} | {optimized['arousal_variance']:<20.4f}")
        print()
        print(f"{'Rate of Change Valencia':<35} | {minimal['valence_roc']:<20.4f} | {optimized['valence_roc']:<20.4f}")
        print(f"{'Rate of Change Arousal':<35} | {minimal['arousal_roc']:<20.4f} | {optimized['arousal_roc']:<20.4f}")
        print()
        print(f"{'Valencia promedio':<35} | {minimal['avg_valence']:<20.3f} | {optimized['avg_valence']:<20.3f}")
        print(f"{'Arousal promedio':<35} | {minimal['avg_arousal']:<20.3f} | {optimized['avg_arousal']:<20.3f}")
    
    print("=" * 90)
    print()
    print("INTERPRETACIÓN:")
    print("  - Estabilidad emoción: Mayor % = menos cambios abruptos")
    print("  - Varianza V/A: Menor valor = más estable")
    print("  - Rate of Change: Menor valor = transiciones más suaves")
    print()


def main():
    """
    Función principal de análisis.
    """
    parser = argparse.ArgumentParser(
        description='Análisis cuantitativo de estabilidad temporal'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Duración de captura por configuración (segundos)'
    )
    
    args = parser.parse_args()
    
    print("=" * 90)
    print("ANÁLISIS CUANTITATIVO DE ESTABILIDAD TEMPORAL")
    print("=" * 90)
    print()
    print(f"Duración de captura: {args.duration} segundos por configuración")
    print()
    print("INSTRUCCIONES:")
    print("  1. Mantén una expresión facial neutral durante 10-15 segundos")
    print("  2. Cambia a una emoción clara (ej: sonrisa amplia)")
    print("  3. Mantén esa emoción durante 10-15 segundos")
    print("  4. Repite para ambas configuraciones")
    print()
    
    # Crear componentes compartidos
    webcam = WebcamCapture(camera_index=0)
    detector = DeepFaceEmotionDetector(enforce_detection=False)
    
    # Configuraciones a comparar
    configs = {
        'minimal': {
            'window_size': 1,
            'alpha': 1.0,
            'min_confidence': 0.0,
            'label': 'Sin Estabilización'
        },
        'optimized': {
            'window_size': 7,
            'alpha': 0.3,
            'min_confidence': 60.0,
            'label': 'Con Estabilización Optimizada'
        }
    }
    
    try:
        # Iniciar cámara una sola vez
        webcam.start()
        print("[OK] Cámara iniciada\n")
        
        results = {}
        
        # Evaluar cada configuración
        for key, config in configs.items():
            print(f"\n{'='*50}")
            print(f"Configuración: {config['label']}")
            print(f"  window_size={config['window_size']}")
            print(f"  alpha={config['alpha']}")
            print(f"  min_confidence={config['min_confidence']}")
            print('='*50)
            
            # Crear pipeline
            pipeline = EmotionPipeline(
                camera=webcam,
                detector=detector,
                window_size=config['window_size'],
                alpha=config['alpha'],
                min_confidence=config['min_confidence']
            )
            
            # Capturar datos
            data = capture_data(pipeline, args.duration, config['label'])
            
            # Calcular métricas
            metrics = calculate_metrics(data)
            results[key] = metrics
            
            # Pequeña pausa entre configuraciones
            if key != list(configs.keys())[-1]:
                print("\nPreparándose para siguiente configuración...")
                import time
                time.sleep(2)
        
        # Mostrar resultados comparativos
        print_comparison_table(results)
        
    except RuntimeError as e:
        print(f"[ERROR] Error: {e}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n[OK] Interrumpido por el usuario")
        
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Liberar recursos
        webcam.release()
        print("\n[OK] Recursos liberados correctamente")


if __name__ == "__main__":
    main()
