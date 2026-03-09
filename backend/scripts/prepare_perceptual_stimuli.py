#!/usr/bin/env python3
"""
prepare_perceptual_stimuli.py

Prepara automáticamente un conjunto de estímulos MIDI (y opcionalmente WAV)
para una evaluación perceptual de música generada por IA.

Lee un directorio de benchmark con la estructura:

    benchmark_dir/
        baseline/
            v+0.8_a+0.8/
                seed0_bars16.mid
                seed2_bars16.mid
                ...
            ...
        transformer_pretrained/   (o "transformer")
            ...
        transformer_finetuned/
            ...

Genera:

    output_dir/
        baseline/
            baseline_v+0.8_a+0.8.mid
            ...
        transformer/
            transformer_v+0.8_a+0.8.mid
            ...
        transformer_finetuned/
            transformer_finetuned_v+0.8_a+0.8.mid
            ...
        stimuli_metadata.csv

Uso:
    python prepare_perceptual_stimuli.py \\
        --benchmark_dir results/final_benchmark_20260301_161909 \\
        --output_dir perceptual_test

    # Con conversión a MP3 (por defecto, recomendado para Google Forms/Drive):
    python prepare_perceptual_stimuli.py \\
        --benchmark_dir results/final_benchmark_20260301_161909 \\
        --output_dir perceptual_test \\
        --convert_audio

    # Con conversión a WAV y SoundFont personalizado:
    python prepare_perceptual_stimuli.py \\
        --benchmark_dir results/final_benchmark_20260301_161909 \\
        --output_dir perceptual_test \\
        --convert_audio --format wav \\
        --soundfont /usr/share/sounds/sf2/FluidR3_GM.sf2

Dependencias para conversión de audio:
    pip install pretty_midi pyFluidSynth pydub
    macOS:  brew install fluidsynth ffmpeg
    Ubuntu: sudo apt install fluidsynth ffmpeg
"""

import argparse
import csv
import logging
import shutil
from pathlib import Path
from typing import Optional

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

# Las cuatro esquinas del espacio valence-arousal
TARGET_COORDINATES: list[str] = [
    "v+0.8_a+0.8",
    "v+0.8_a-0.8",
    "v-0.8_a+0.8",
    "v-0.8_a-0.8",
]

# Motores a procesar por defecto.
# "transformer" se resuelve automáticamente a "transformer_pretrained" si no
# existe una carpeta con ese nombre exacto (ver resolve_engine_dir).
DEFAULT_ENGINES: list[str] = ["baseline", "transformer", "transformer_finetuned"]

# Semilla preferida y longitud de pieza
PREFERRED_SEED = "seed2"
BARS_SUFFIX = "bars16"

# Alias de carpetas para cada nombre de motor lógico.
# Si la carpeta exacta no existe, se prueban los alias en orden.
ENGINE_FOLDER_ALIASES: dict[str, list[str]] = {
    "transformer": ["transformer", "transformer_pretrained"],
}


# ── Resolución de directorios ─────────────────────────────────────────────────


def resolve_engine_dir(benchmark_dir: Path, engine_name: str) -> Optional[Path]:
    """
    Devuelve el directorio fuente del motor dentro de benchmark_dir.

    Si la carpeta no existe con el nombre exacto, prueba los alias definidos
    en ENGINE_FOLDER_ALIASES (p.ej. "transformer" → "transformer_pretrained").

    Returns:
        Path al directorio si existe, None en caso contrario.
    """
    candidates = ENGINE_FOLDER_ALIASES.get(engine_name, [engine_name])
    for name in candidates:
        candidate = benchmark_dir / name
        if candidate.is_dir():
            logger.debug("Motor '%s' resuelto como carpeta '%s'.", engine_name, name)
            return candidate
    return None


# ── Selección de archivo MIDI ─────────────────────────────────────────────────


def select_midi_file(coord_dir: Path) -> Optional[Path]:
    """
    Selecciona el archivo MIDI preferido dentro de una carpeta de coordenada VA.

    Prioridad:
      1. {PREFERRED_SEED}_{BARS_SUFFIX}.mid  (p.ej. seed2_bars16.mid)
      2. Cualquier seedX_{BARS_SUFFIX}.mid disponible (orden alfabético)

    Returns:
        Path al archivo MIDI seleccionado, o None si no se encuentra ninguno.
    """
    preferred = coord_dir / f"{PREFERRED_SEED}_{BARS_SUFFIX}.mid"
    if preferred.exists():
        return preferred

    # Fallback: primer bars16 disponible (orden determinista)
    candidates = sorted(coord_dir.glob(f"*_{BARS_SUFFIX}.mid"))
    if candidates:
        logger.warning(
            "  '%s' no encontrado en '%s'. Usando '%s' como fallback.",
            preferred.name,
            coord_dir.name,
            candidates[0].name,
        )
        return candidates[0]

    logger.warning(
        "  No se encontró ningún archivo *_%s.mid en: %s", BARS_SUFFIX, coord_dir
    )
    return None


# ── Copia de archivos ─────────────────────────────────────────────────────────


def copy_midi(src: Path, dst: Path) -> bool:
    """
    Copia src → dst, creando los directorios necesarios.

    Returns:
        True si la copia fue exitosa, False en caso de error.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        logger.info("  Copiado: %s → %s", src.name, dst)
        return True
    except OSError as exc:
        logger.error("  Error al copiar '%s': %s", src, exc)
        return False


# ── Utilidades ────────────────────────────────────────────────────────────────


def parse_valence_arousal(coord: str) -> tuple[str, str]:
    """
    Extrae los valores numéricos de valencia y activación de una cadena como
    'v+0.8_a-0.2'.

    Returns:
        Tupla (valence, arousal) como strings con signo, p.ej. ('+0.8', '-0.2').
    """
    parts = coord.split("_")
    valence = parts[0][1:]  # elimina la 'v' inicial
    arousal = parts[1][1:]  # elimina la 'a' inicial
    return valence, arousal


# ── Conversión MIDI → MP3 / WAV (opcional) ───────────────────────────────────

SAMPLE_RATE = 44100
MP3_BITRATE = "192k"  # 192 kbps: calidad óptima para evaluación perceptual


def _render_audio_array(
    midi_path: Path, soundfont: Optional[str]
) -> Optional["numpy.ndarray"]:
    """
    Sintetiza un MIDI con pretty_midi + fluidsynth y devuelve el array de audio
    (float32, mono/stereo, SAMPLE_RATE Hz).

    Retorna None si falla la importación o la síntesis.
    """
    try:
        import pretty_midi  # type: ignore  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "'pretty_midi' no instalado. Ejecuta: pip install pretty_midi pyFluidSynth"
        )
        return None
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        audio = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=soundfont)
        return audio
    except Exception as exc:  # noqa: BLE001
        logger.error("  Error sintetizando '%s': %s", midi_path.name, exc)
        return None


def midi_to_audio(
    midi_path: Path,
    out_path: Path,
    fmt: str = "mp3",
    soundfont: Optional[str] = None,
) -> bool:
    """
    Convierte un MIDI a MP3 o WAV.

    MP3 (recomendado para Google Forms/Drive):
      ~10× más pequeño que WAV a 192 kbps sin pérdida perceptible.
      Requiere: pydub + ffmpeg en el sistema.

    WAV:
      Sin compresión, máxima fidelidad. Requiere solo scipy.

    Dependencias del sistema:
      macOS:  brew install fluidsynth ffmpeg
      Ubuntu: sudo apt install fluidsynth ffmpeg

    Returns:
        True si la conversión fue exitosa, False en caso contrario.
    """
    audio = _render_audio_array(midi_path, soundfont)
    if audio is None:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "wav":
        try:
            import scipy.io.wavfile as wavfile  # noqa: PLC0415
        except ImportError:
            logger.warning("'scipy' no instalado. Ejecuta: pip install scipy")
            return False
        try:
            # Normalizar a int16 para WAV estándar
            import numpy as np  # noqa: PLC0415
            audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            wavfile.write(str(out_path), SAMPLE_RATE, audio_int16)
            logger.info("  WAV generado: %s", out_path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("  Error escribiendo WAV '%s': %s", out_path.name, exc)
            return False

    # ── MP3 ──
    try:
        from pydub import AudioSegment  # type: ignore  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "'pydub' no instalado. Ejecuta: pip install pydub\n"
            "  Y asegúrate de tener ffmpeg: brew install ffmpeg"
        )
        return False
    try:
        import numpy as np  # noqa: PLC0415

        # pydub trabaja con int16
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,   # int16 → 2 bytes
            channels=1,
        )
        segment.export(str(out_path), format="mp3", bitrate=MP3_BITRATE)
        logger.info("  MP3 generado: %s", out_path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("  Error convirtiendo a MP3 '%s': %s", midi_path.name, exc)
        return False


# ── Proceso principal ─────────────────────────────────────────────────────────


def prepare_stimuli(
    benchmark_dir: Path,
    output_dir: Path,
    engines: list[str],
    coordinates: list[str],
    convert_audio: bool,
    audio_format: str,
    soundfont: Optional[str],
) -> list[dict]:
    """
    Itera sobre motores y coordenadas, selecciona los MIDIs, los copia y
    opcionalmente los convierte a MP3 o WAV.

    Los errores en un motor no interrumpen el procesamiento del resto.

    Returns:
        Lista de dicts con los metadatos de cada estímulo procesado.
    """
    metadata_rows: list[dict] = []

    for engine in engines:
        logger.info("── Motor: %s", engine)

        engine_src = resolve_engine_dir(benchmark_dir, engine)
        if engine_src is None:
            logger.warning(
                "Motor '%s' no encontrado en '%s'. Se omite.",
                engine,
                benchmark_dir,
            )
            continue

        engine_dst = output_dir / engine
        engine_dst.mkdir(parents=True, exist_ok=True)

        for coord in coordinates:
            coord_dir = engine_src / coord
            if not coord_dir.is_dir():
                logger.warning(
                    "  Coordenada '%s' no encontrada para motor '%s'. Se omite.",
                    coord,
                    engine,
                )
                continue

            midi_src = select_midi_file(coord_dir)
            if midi_src is None:
                continue

            filename = f"{engine}_{coord}.mid"
            midi_dst = engine_dst / filename

            if not copy_midi(midi_src, midi_dst):
                continue

            valence, arousal = parse_valence_arousal(coord)
            metadata_rows.append(
                {
                    "filename": filename,
                    "engine": engine,
                    "valence": valence,
                    "arousal": arousal,
                    "source_path": str(midi_src.resolve()),
                }
            )

            if convert_audio:
                audio_dst = engine_dst / filename.replace(".mid", f".{audio_format}")
                midi_to_audio(midi_src, audio_dst, fmt=audio_format, soundfont=soundfont)

    return metadata_rows


# ── CSV de metadatos ──────────────────────────────────────────────────────────


def write_metadata_csv(rows: list[dict], output_dir: Path) -> None:
    """Escribe stimuli_metadata.csv en output_dir."""
    csv_path = output_dir / "stimuli_metadata.csv"
    fieldnames = ["filename", "engine", "valence", "arousal", "source_path"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV de metadatos guardado en: %s", csv_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara estímulos MIDI (y opcionalmente WAV) para una evaluación "
            "perceptual de música generada por IA."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        required=True,
        help=(
            "Ruta a la carpeta del benchmark "
            "(p.ej. results/final_benchmark_20260301_161909)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("perceptual_test"),
        help="Carpeta de salida donde se guardarán los estímulos.",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=DEFAULT_ENGINES,
        metavar="ENGINE",
        help=(
            "Motores a incluir. 'transformer' se resuelve automáticamente "
            "a 'transformer_pretrained' si esa carpeta existe."
        ),
    )
    parser.add_argument(
        "--coordinates",
        nargs="+",
        default=TARGET_COORDINATES,
        metavar="COORD",
        help="Coordenadas VA a incluir (formato: v±N_a±N).",
    )
    parser.add_argument(
        "--convert_audio",
        action="store_true",
        help=(
            "Convierte los MIDI a audio (MP3 por defecto). "
            "Requiere: pip install pretty_midi pyFluidSynth pydub "
            "y fluidsynth + ffmpeg en el sistema."
        ),
    )
    parser.add_argument(
        "--format",
        dest="audio_format",
        choices=["mp3", "wav"],
        default="mp3",
        help=(
            "Formato de audio de salida. "
            "'mp3' (~10× más pequeño, compatible con Google Drive/Forms). "
            "'wav' (sin compresión, máxima fidelidad)."
        ),
    )
    parser.add_argument(
        "--soundfont",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Ruta a un SoundFont .sf2 para la síntesis. "
            "Si se omite, fluidsynth usa su SoundFont por defecto."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not benchmark_dir.is_dir():
        logger.error("El directorio de benchmark no existe: %s", benchmark_dir)
        raise SystemExit(1)

    logger.info("=" * 60)
    logger.info("Benchmark dir : %s", benchmark_dir)
    logger.info("Output dir    : %s", output_dir)
    logger.info("Motores       : %s", args.engines)
    logger.info("Coordenadas   : %s", args.coordinates)
    logger.info("Convertir audio: %s", args.convert_audio)
    if args.convert_audio:
        logger.info("Formato audio : %s (%s)", args.audio_format, MP3_BITRATE if args.audio_format == "mp3" else "sin compresión")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = prepare_stimuli(
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        engines=args.engines,
        coordinates=args.coordinates,
        convert_audio=args.convert_audio,
        audio_format=args.audio_format,
        soundfont=args.soundfont,
    )

    if rows:
        write_metadata_csv(rows, output_dir)
        logger.info("=" * 60)
        logger.info(
            "Proceso completado. %d estímulos preparados en: %s",
            len(rows),
            output_dir,
        )
    else:
        logger.error(
            "No se procesó ningún estímulo. "
            "Revisa las rutas y los nombres de motores/coordenadas."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
