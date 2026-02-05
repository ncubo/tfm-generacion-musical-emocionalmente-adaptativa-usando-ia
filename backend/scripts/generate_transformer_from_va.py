#!/usr/bin/env python3
"""
Script de prueba para generación de MIDI con Music Transformer desde coordenadas V/A.

Este script permite generar archivos MIDI usando un Music Transformer preentrenado,
condicionado indirectamente por valores manuales de Valence-Arousal.

Uso:
    python backend/scripts/generate_transformer_from_va.py --v 0.7 --a 0.6 --out data/outputs/midis/happy_transformer.mid
    python backend/scripts/generate_transformer_from_va.py --v -0.7 --a -0.4 --out data/outputs/midis/sad_transformer.mid
    python backend/scripts/generate_transformer_from_va.py --v 0.0 --a 0.0 --out data/outputs/midis/neutral_transformer.mid

Opciones:
    --v VALENCE         Valence en [-1, 1] (default: 0.0)
    --a AROUSAL         Arousal en [-1, 1] (default: 0.0)
    --out PATH          Path de salida para el MIDI (default: data/outputs/midis/transformer_va.mid)
    --checkpoint PATH   Path al checkpoint del transformer (default: None = pesos aleatorios)
    --max-length INT    Longitud máxima en tokens (default: 512)
    --seed INT          Semilla aleatoria (default: None)
    --use-primer        Activar uso de primer melody
    
Ejemplo completo:
    python backend/scripts/generate_transformer_from_va.py \\
        --v 0.8 --a 0.7 \\
        --out data/outputs/midis/excited_transformer.mid \\
        --max-length 1024 \\
        --seed 42 \\
        --use-primer

Prerequisitos:
    1. Instalar dependencias: pip install -r backend/requirements.txt
    2. (Opcional) Descargar checkpoint preentrenado
"""

import sys
import argparse
import logging
from pathlib import Path

# Añadir src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.music.mapping import va_to_music_params
from core.music.transformer.transformer_infer import TransformerMusicGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Genera MIDI con Music Transformer desde coordenadas Valence-Arousal',
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
        '--checkpoint',
        type=str,
        default=None,
        help='Path al checkpoint del transformer (default: None)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        dest='max_length',
        help='Longitud máxima en tokens (default: 512)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Semilla aleatoria para reproducibilidad (default: None)'
    )
    
    parser.add_argument(
        '--use-primer',
        action='store_true',
        dest='use_primer',
        help='Activar uso de primer melody'
    )
    
    args = parser.parse_args()
    
    # Validar V y A
    if not (-1 <= args.valence <= 1):
        parser.error(f"Valence debe estar en [-1, 1]. Got: {args.valence}")
    
    if not (-1 <= args.arousal <= 1):
        parser.error(f"Arousal debe estar en [-1, 1]. Got: {args.arousal}")
    
    # Mostrar configuración
    logger.info("=" * 60)
    logger.info("GENERACIÓN MIDI CON MUSIC TRANSFORMER")
    logger.info("=" * 60)
    logger.info(f"Valence: {args.valence:.2f}")
    logger.info(f"Arousal: {args.arousal:.2f}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Checkpoint: {args.checkpoint or 'None (pesos aleatorios)'}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Usar primer: {args.use_primer}")
    logger.info("=" * 60)
    
    # 1. Mapear V/A a parámetros musicales
    logger.info("\n[1/3] Mapeando V/A a parámetros musicales...")
    music_params = va_to_music_params(args.valence, args.arousal)
    
    logger.info(f"  Tempo: {music_params['tempo_bpm']} BPM")
    logger.info(f"  Modo: {music_params['mode']}")
    logger.info(f"  Rango tonal: MIDI {music_params['pitch_low']}-{music_params['pitch_high']}")
    logger.info(f"  Densidad: {music_params['density']:.2f}")
    logger.info(f"  Complejidad rítmica: {music_params['rhythm_complexity']:.2f}")
    
    # 2. Inicializar generador
    logger.info("\n[2/3] Inicializando generador Music Transformer...")
    try:
        generator = TransformerMusicGenerator(
            checkpoint_path=args.checkpoint,
            device=None  # Auto-detectar
        )
    except Exception as e:
        logger.error(f"Error inicializando generador: {e}")
        logger.error("\nPosibles soluciones:")
        logger.error("1. Verificar instalación de PyTorch: pip install torch")
        logger.error("2. Verificar instalación de miditok: pip install miditok")
        logger.error("3. Si se especificó checkpoint, verificar que existe")
        sys.exit(1)
    
    # 3. Generar MIDI
    logger.info("\n[3/3] Generando MIDI...")
    try:
        result = generator.generate(
            v=args.valence,
            a=args.arousal,
            out_path=args.output,
            max_length=args.max_length,
            seed=args.seed,
            use_primer=args.use_primer,
            music_params=music_params if args.use_primer else None
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("GENERACIÓN COMPLETADA")
        logger.info("=" * 60)
        logger.info(f"Archivo MIDI: {result['midi_path']}")
        logger.info(f"Tokens generados: {result['num_tokens']}")
        logger.info("\nParámetros de generación:")
        for key, value in result['generation_params'].items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
        
        return 0
    
    except Exception as e:
        logger.error(f"\nError durante la generación: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
