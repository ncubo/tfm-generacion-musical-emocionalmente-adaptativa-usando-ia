#!/usr/bin/env python3
"""
Script de prueba para generación de MIDI con HF Maestro-REMI desde coordenadas V/A.

Este script permite generar archivos MIDI usando el modelo Hugging Face Maestro-REMI,
condicionado indirectamente por valores manuales de Valence-Arousal.

Modelo usado: https://huggingface.co/NathanFradet/Maestro-REMI-bpe20k

Uso:
    python backend/scripts/generate_transformer_from_va.py --v 0.7 --a 0.6 --out data/outputs/midis/happy_transformer.mid
    python backend/scripts/generate_transformer_from_va.py --v -0.7 --a -0.4 --out data/outputs/midis/sad_transformer.mid
    python backend/scripts/generate_transformer_from_va.py --v 0.0 --a 0.0 --out data/outputs/midis/neutral_transformer.mid

Opciones:
    --v VALENCE         Valence en [-1, 1] (default: 0.0)
    --a AROUSAL         Arousal en [-1, 1] (default: 0.0)
    --out PATH          Path de salida para el MIDI (default: data/outputs/midis/transformer_va.mid)
    --length-bars INT   Número de compases (default: 8)
    --seed INT          Semilla aleatoria (default: None)
    
Ejemplo completo:
    python backend/scripts/generate_transformer_from_va.py \\
        --v 0.8 --a 0.7 \\
        --out data/outputs/midis/excited_transformer.mid \\
        --length-bars 16 \\
        --seed 42

Prerequisitos:
    1. Instalar dependencias: pip install -r backend/requirements.txt
    2. El modelo se descarga automáticamente desde Hugging Face
"""

import sys
import argparse
import logging
from pathlib import Path

# Añadir src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.music.mapping import va_to_music_params
from core.music.engines.hf_maestro_remi import generate_midi_hf_maestro_remi

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Genera MIDI con HF Maestro-REMI desde coordenadas Valence-Arousal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--v', '--valence',
        type=float,
        default=0.0,
        dest='valence',
        help='Valence en [-1, 1] (default: 0.0)'
    )
    
    parser.add_argument(
        '--a', '--arousal',
        type=float,
        default=0.0,
        dest='arousal',
        help='Arousal en [-1, 1] (default: 0.0)'
    )
    
    parser.add_argument(
        '--out', '--output',
        type=str,
        default='data/outputs/midis/transformer_va.mid',
        dest='output',
        help='Path de salida para el MIDI (default: data/outputs/midis/transformer_va.mid)'
    )
    
    parser.add_argument(
        '--length-bars',
        type=int,
        default=8,
        dest='length_bars',
        help='Número de compases a generar (default: 8)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Semilla aleatoria para reproducibilidad (default: None)'
    )
    
    args = parser.parse_args()
    
    # Validar V y A
    if not (-1 <= args.valence <= 1):
        parser.error(f"Valence debe estar en [-1, 1]. Got: {args.valence}")
    
    if not (-1 <= args.arousal <= 1):
        parser.error(f"Arousal debe estar en [-1, 1]. Got: {args.arousal}")
    
    # Mostrar configuración
    logger.info("=" * 60)
    logger.info("GENERACIÓN MIDI CON HF MAESTRO-REMI")
    logger.info("=" * 60)
    logger.info(f"Valence: {args.valence:.2f}")
    logger.info(f"Arousal: {args.arousal:.2f}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Length bars: {args.length_bars}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 60)
    
    # 1. Mapear V/A a parámetros musicales
    logger.info("\n[1/2] Mapeando V/A a parámetros musicales...")
    music_params = va_to_music_params(args.valence, args.arousal)
    
    logger.info(f"  Tempo: {music_params['tempo_bpm']} BPM")
    logger.info(f"  Modo: {music_params['mode']}")
    logger.info(f"  Rango tonal: MIDI {music_params['pitch_low']}-{music_params['pitch_high']}")
    logger.info(f"  Densidad: {music_params['density']:.2f}")
    logger.info(f"  Complejidad rítmica: {music_params['rhythm_complexity']:.2f}")
    
    # 2. Generar MIDI con HF Maestro-REMI
    logger.info("\n[2/2] Generando MIDI con HF Maestro-REMI...")
    logger.info("  (Primera ejecución puede tardar mientras se descarga el modelo)")
    
    try:
        generated_path = generate_midi_hf_maestro_remi(
            params=music_params,
            out_path=args.output,
            length_bars=args.length_bars,
            seed=args.seed
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("GENERACIÓN COMPLETADA")
        logger.info("=" * 60)
        logger.info(f"Archivo MIDI: {generated_path}")
        logger.info(f"Compases generados: {args.length_bars}")
        logger.info("=" * 60)
        
        return 0
    
    except Exception as e:
        logger.error(f"\nError durante la generación: {e}")
        import traceback
        traceback.print_exc()
        logger.error("\nPosibles soluciones:")
        logger.error("1. Verificar instalación: pip install transformers miditok torch")
        logger.error("2. Verificar conexión a internet (para descargar modelo)")
        logger.error("3. Verificar espacio en disco para cache del modelo")
        return 1


if __name__ == '__main__':
    sys.exit(main())

