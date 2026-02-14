#!/usr/bin/env python3
"""
Script de test para el engine HF Maestro-REMI.

Genera archivos MIDI de prueba para diferentes emociones y verifica que:
- El engine carga correctamente
- Genera archivos MIDI válidos
- Responde a diferentes parámetros musicales
- Es reproducible con seeds

Uso:
    python scripts/test_hf_maestro_remi.py
    python scripts/test_hf_maestro_remi.py --emotion happy
    python scripts/test_hf_maestro_remi.py --all
"""

import sys
import argparse
from pathlib import Path

# Añadir src al path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir / "src"))

from core.music.engines.hf_maestro_remi import generate_midi_hf_maestro_remi
from core.music.engines.baseline import generate_midi_baseline

# Parámetros musicales para diferentes emociones
EMOTION_PARAMS = {
    'happy': {
        'tempo_bpm': 130,
        'mode': 'major',
        'density': 0.75,
        'pitch_low': 60,
        'pitch_high': 84,
        'rhythm_complexity': 0.6,
        'velocity_mean': 90,
        'velocity_spread': 20
    },
    'sad': {
        'tempo_bpm': 70,
        'mode': 'minor',
        'density': 0.4,
        'pitch_low': 48,
        'pitch_high': 72,
        'rhythm_complexity': 0.3,
        'velocity_mean': 60,
        'velocity_spread': 15
    },
    'excited': {
        'tempo_bpm': 150,
        'mode': 'major',
        'density': 0.85,
        'pitch_low': 64,
        'pitch_high': 84,
        'rhythm_complexity': 0.8,
        'velocity_mean': 100,
        'velocity_spread': 25
    },
    'calm': {
        'tempo_bpm': 80,
        'mode': 'major',
        'density': 0.5,
        'pitch_low': 55,
        'pitch_high': 75,
        'rhythm_complexity': 0.3,
        'velocity_mean': 70,
        'velocity_spread': 10
    },
    'angry': {
        'tempo_bpm': 140,
        'mode': 'minor',
        'density': 0.8,
        'pitch_low': 55,
        'pitch_high': 80,
        'rhythm_complexity': 0.75,
        'velocity_mean': 95,
        'velocity_spread': 30
    },
}


def test_single_emotion(emotion: str, seed: int = 42, length_bars: int = 8):
    """
    Genera un MIDI con HF engine para una emoción específica.
    
    Args:
        emotion: Nombre de la emoción
        seed: Semilla aleatoria
        length_bars: Número de compases
        
    Returns:
        bool: True si el test pasó
    """
    if emotion not in EMOTION_PARAMS:
        print(f"Emoción desconocida: {emotion}")
        print(f"   Emociones disponibles: {list(EMOTION_PARAMS.keys())}")
        return False
    
    params = EMOTION_PARAMS[emotion]
    output_dir = backend_dir / "output" / "test_hf"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{emotion}_hf.mid"
    
    print(f"\n{'='*60}")
    print(f"TEST: {emotion.upper()}")
    print(f"{'='*60}")
    print(f"Parámetros:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"Seed: {seed}")
    print(f"Compases: {length_bars}")
    print(f"Output: {output_path}")
    print()
    
    try:
        # Generar con HF engine
        result = generate_midi_hf_maestro_remi(
            params=params,
            out_path=str(output_path),
            seed=seed,
            length_bars=length_bars
        )
        
        # Verificar que el archivo existe
        if not output_path.exists():
            print(f"FAIL: Archivo no generado")
            return False
        
        file_size = output_path.stat().st_size
        print(f"PASS: Generado {output_path.name} ({file_size} bytes)")
        return True
        
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_baseline(emotion: str = 'happy', seed: int = 42, length_bars: int = 4):
    """
    Genera con HF engine y baseline para comparar.
    
    Args:
        emotion: Nombre de la emoción
        seed: Semilla aleatoria
        length_bars: Número de compases
    """
    params = EMOTION_PARAMS[emotion]
    output_dir = backend_dir / "output" / "test_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hf_path = output_dir / f"{emotion}_hf.mid"
    baseline_path = output_dir / f"{emotion}_baseline.mid"
    
    print(f"\n{'='*60}")
    print(f"COMPARACIÓN: {emotion.upper()}")
    print(f"{'='*60}")
    
    # Generar con HF
    print(f"\n[1/2] Generando con HF Maestro-REMI...")
    try:
        generate_midi_hf_maestro_remi(
            params=params,
            out_path=str(hf_path),
            seed=seed,
            length_bars=length_bars
        )
        hf_size = hf_path.stat().st_size
        print(f"HF: {hf_path.name} ({hf_size} bytes)")
    except Exception as e:
        print(f"HF falló: {e}")
        hf_size = 0
    
    # Generar con baseline
    print(f"\n[2/2] Generando con baseline...")
    try:
        generate_midi_baseline(
            params=params,
            out_path=str(baseline_path),
            seed=seed,
            length_bars=length_bars
        )
        baseline_size = baseline_path.stat().st_size
        print(f"Baseline: {baseline_path.name} ({baseline_size} bytes)")
    except Exception as e:
        print(f"Baseline falló: {e}")
        baseline_size = 0
    
    # Comparar tamaños
    print(f"\nComparación:")
    print(f"   HF:       {hf_size:,} bytes")
    print(f"   Baseline: {baseline_size:,} bytes")
    if hf_size > 0 and baseline_size > 0:
        ratio = hf_size / baseline_size
        print(f"   Ratio:    {ratio:.2f}x (HF típicamente genera más datos)")


def test_reproducibility(emotion: str = 'happy', length_bars: int = 4):
    """
    Verifica que el engine sea reproducible con la misma seed.
    
    Args:
        emotion: Nombre de la emoción
        length_bars: Número de compases
        
    Returns:
        bool: True si es reproducible
    """
    params = EMOTION_PARAMS[emotion]
    output_dir = backend_dir / "output" / "test_reproducibility"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path1 = output_dir / f"{emotion}_run1.mid"
    path2 = output_dir / f"{emotion}_run2.mid"
    
    print(f"\n{'='*60}")
    print(f"TEST REPRODUCIBILIDAD: {emotion.upper()}")
    print(f"{'='*60}")
    
    seed = 12345
    
    # Primera ejecución
    print(f"\n[1/2] Primera generación (seed={seed})...")
    try:
        generate_midi_hf_maestro_remi(params, str(path1), seed=seed, length_bars=length_bars)
        size1 = path1.stat().st_size
        print(f"Run 1: {size1} bytes")
    except Exception as e:
        print(f"Run 1 falló: {e}")
        return False
    
    # Segunda ejecución
    print(f"\n[2/2] Segunda generación (seed={seed})...")
    try:
        generate_midi_hf_maestro_remi(params, str(path2), seed=seed, length_bars=length_bars)
        size2 = path2.stat().st_size
        print(f"Run 2: {size2} bytes")
    except Exception as e:
        print(f"Run 2 falló: {e}")
        return False
    
    # Comparar archivos
    with open(path1, 'rb') as f1, open(path2, 'rb') as f2:
        content1 = f1.read()
        content2 = f2.read()
    
    if content1 == content2:
        print(f"\nREPRODUCIBLE: Archivos idénticos")
        return True
    else:
        print(f"\nNO REPRODUCIBLE: Archivos diferentes")
        print(f"   Diferencia: {abs(size1 - size2)} bytes")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test del engine HF Maestro-REMI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/test_hf_maestro_remi.py --emotion happy
  python scripts/test_hf_maestro_remi.py --all
  python scripts/test_hf_maestro_remi.py --compare
  python scripts/test_hf_maestro_remi.py --reproducibility
        """
    )
    
    parser.add_argument('--emotion', type=str, 
                       help=f'Emoción a testear: {list(EMOTION_PARAMS.keys())}')
    parser.add_argument('--all', action='store_true',
                       help='Testear todas las emociones')
    parser.add_argument('--compare', action='store_true',
                       help='Comparar HF engine con baseline')
    parser.add_argument('--reproducibility', action='store_true',
                       help='Test de reproducibilidad con seeds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Semilla aleatoria (default: 42)')
    parser.add_argument('--bars', type=int, default=8,
                       help='Número de compases (default: 8)')
    
    args = parser.parse_args()
    
    print("\nTEST HF MAESTRO-REMI ENGINE")
    print("=" * 60)
    
    results = []
    
    # Test individual
    if args.emotion:
        success = test_single_emotion(args.emotion, seed=args.seed, length_bars=args.bars)
        results.append((args.emotion, success))
    
    # Test todas las emociones
    elif args.all:
        for emotion in EMOTION_PARAMS:
            success = test_single_emotion(emotion, seed=args.seed, length_bars=args.bars)
            results.append((emotion, success))
    
    # Comparación
    elif args.compare:
        compare_with_baseline(seed=args.seed, length_bars=args.bars)
    
    # Reproducibilidad
    elif args.reproducibility:
        success = test_reproducibility(length_bars=args.bars)
        results.append(('reproducibility', success))
    
    # Por defecto: test happy
    else:
        success = test_single_emotion('happy', seed=args.seed, length_bars=args.bars)
        results.append(('happy', success))
    
    # Resumen
    if results:
        print(f"\n{'='*60}")
        print("RESUMEN")
        print(f"{'='*60}")
        
        for name, success in results:
            status = "PASS" if success else "FAIL"
            print(f"{status} - {name}")
        
        total = len(results)
        passed = sum(1 for _, s in results if s)
        
        print(f"\nResultado: {passed}/{total} tests pasaron")
        
        if passed == total:
            print("\n¡Todos los tests pasaron!")
            return 0
        else:
            print("\nAlgunos tests fallaron")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
