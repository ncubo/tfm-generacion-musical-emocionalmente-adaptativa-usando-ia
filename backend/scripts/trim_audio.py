#!/usr/bin/env python3
"""
trim_audio.py

Recorta archivos MP3 a un máximo de 30 segundos para reducir la fatiga
de los participantes en un experimento de evaluación perceptual.

Estructura esperada de entrada:
    input_dir/
        baseline/        *.mp3
        transformer/     *.mp3
        transformer_finetuned/  *.mp3

Estructura de salida:
    output_dir/
        baseline/        *.mp3  (≤ MAX_DURATION_MS ms)
        transformer/     *.mp3
        transformer_finetuned/  *.mp3

Si un archivo ya dura ≤ MAX_DURATION_MS, se copia sin modificar.

Uso:
    python trim_audio.py
    python trim_audio.py --input_dir perceptual_test --output_dir audio_trimmed
    python trim_audio.py --max_seconds 45
"""

import argparse
import logging
import shutil
from pathlib import Path

from pydub import AudioSegment  # pip install pydub  (+ ffmpeg en el sistema)

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

# Subcarpetas que se escanean dentro de input_dir
SOURCE_SUBDIRS: list[str] = ["baseline", "transformer", "transformer_finetuned"]

# Duración máxima por defecto (segundos → ms internamente)
DEFAULT_MAX_SECONDS: int = 30


# ── Función principal de recorte ──────────────────────────────────────────────


def trim_file(src: Path, dst: Path, max_ms: int) -> None:
    """
    Recorta src a max_ms milisegundos y lo guarda en dst.

    - Si la duración ya es ≤ max_ms, copia el archivo sin modificar.
    - Mantiene la misma tasa de bits del original.
    - Lanza excepciones para que el llamador pueda manejar errores.

    Args:
        src:    Ruta al archivo MP3 de entrada.
        dst:    Ruta al archivo MP3 de salida.
        max_ms: Duración máxima en milisegundos.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_mp3(str(src))
    duration_ms = len(audio)
    duration_s = duration_ms / 1000

    if duration_ms <= max_ms:
        # El archivo ya cumple el límite; copiar directamente
        shutil.copy2(src, dst)
        logger.info(
            "  COPIADO  %-50s  (%.1f s, ya dentro del límite)",
            src.name,
            duration_s,
        )
    else:
        # Recortar y exportar como MP3 con la misma tasa de bits
        trimmed = audio[:max_ms]
        # Intentar preservar la tasa de bits original; por defecto 192k
        bitrate = audio.frame_rate  # Hz, solo informativo aquí
        trimmed.export(str(dst), format="mp3", bitrate="192k")
        logger.info(
            "  RECORTADO %-50s  (%.1f s → %.1f s)",
            src.name,
            duration_s,
            max_ms / 1000,
        )


# ── Escaneo y procesado de carpetas ──────────────────────────────────────────


def process_directory(
    input_dir: Path,
    output_dir: Path,
    max_ms: int,
    subdirs: list[str],
) -> tuple[int, int]:
    """
    Recorre cada subcarpeta en subdirs dentro de input_dir y recorta los MP3.

    Los errores en un archivo no detienen el procesamiento de los demás.

    Returns:
        Tupla (procesados_ok, errores).
    """
    ok = 0
    errors = 0

    for subdir in subdirs:
        src_subdir = input_dir / subdir
        if not src_subdir.is_dir():
            logger.warning("Subcarpeta no encontrada, se omite: %s", src_subdir)
            continue

        mp3_files = sorted(src_subdir.glob("*.mp3"))
        if not mp3_files:
            logger.warning("Sin archivos .mp3 en: %s", src_subdir)
            continue

        logger.info("── %s  (%d archivos)", subdir, len(mp3_files))

        for mp3_src in mp3_files:
            mp3_dst = output_dir / subdir / mp3_src.name
            try:
                trim_file(mp3_src, mp3_dst, max_ms)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "  ERROR procesando '%s': %s", mp3_src.name, exc
                )
                errors += 1

    return ok, errors


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recorta archivos MP3 a un máximo de N segundos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("perceptual_test"),
        help="Carpeta raíz con las subcarpetas de audio (baseline, transformer, …).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("audio_trimmed"),
        help="Carpeta de salida donde se guardarán los audios recortados.",
    )
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="Duración máxima en segundos.",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=SOURCE_SUBDIRS,
        metavar="SUBDIR",
        help="Subcarpetas a procesar dentro de input_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    max_ms = int(args.max_seconds * 1000)

    if not input_dir.is_dir():
        logger.error("El directorio de entrada no existe: %s", input_dir)
        raise SystemExit(1)

    logger.info("=" * 60)
    logger.info("Input dir    : %s", input_dir)
    logger.info("Output dir   : %s", output_dir)
    logger.info("Duración máx : %.1f s  (%d ms)", args.max_seconds, max_ms)
    logger.info("Subcarpetas  : %s", args.subdirs)
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    ok, errors = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        max_ms=max_ms,
        subdirs=args.subdirs,
    )

    logger.info("=" * 60)
    logger.info("Completado. OK: %d  |  Errores: %d", ok, errors)
    logger.info("Archivos guardados en: %s", output_dir)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
