"""
Entrada/salida de MIDI para generación con Transformer.

Maneja la conversión entre tokens y archivos MIDI usando miditok (REMI encoding).
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict
import mido

logger = logging.getLogger(__name__)


try:
    from miditok import REMI, TokenizerConfig
    MIDITOK_AVAILABLE = True
except ImportError:
    MIDITOK_AVAILABLE = False
    logger.warning("miditok no disponible. Instalar con: pip install miditok")


class MIDITokenConverter:
    """
    Conversor entre tokens y MIDI usando tokenización REMI.
    
    REMI (REpresentation for Music with Instruments) representa música como
    una secuencia de eventos: Bar, Position, Tempo, Chord, Pitch, Velocity, Duration.
    """
    
    def __init__(self, vocab_size: int = 512, beat_resolution: int = 4):
        """
        Inicializa el conversor de tokens.
        
        Args:
            vocab_size: Tamaño del vocabulario
            beat_resolution: Resolución de beat (ticks por beat)
        """
        if not MIDITOK_AVAILABLE:
            raise RuntimeError("miditok no está instalado. Ejecuta: pip install miditok")
        
        # Configuración de tokenización REMI (miditok 3.0+)
        config = TokenizerConfig(
            pitch_range=(21, 109),  # A0 a C8 (piano completo)
            beat_res={(0, 4): 8, (4, 12): 4},  # Resolución variable por compás
            num_velocities=32,  # 32 niveles de velocidad
            use_chords=True,  # Detectar acordes
            use_rests=True,   # Silencios
            use_tempos=True,  # Cambios de tempo
            use_programs=False,  # Sin cambios de instrumento por simplicidad
        )
        
        self.tokenizer = REMI(tokenizer_config=config)
        logger.info("REMI tokenizer inicializado")
    
    def tokens_to_midi(
        self,
        tokens: List[int],
        out_path: str,
        tempo: int = 120
    ) -> str:
        """
        Convierte una secuencia de tokens a archivo MIDI.
        
        Args:
            tokens: Lista de IDs de tokens
            out_path: Path donde guardar el MIDI
            tempo: Tempo inicial (BPM)
        
        Returns:
            Path al archivo MIDI generado
        
        Raises:
            ValueError: Si los tokens no son válidos o el MIDI está vacío
        """
        try:
            # Convertir tokens a MIDI usando miditok 3.0+ (usa 'decode' en lugar de 'tokens_to_midi')
            # Retorna un objeto symusic.Score en lugar de mido.MidiFile
            midi_score = self.tokenizer.decode([tokens])
            
            # Validar que el MIDI no esté vacío
            if not self._validate_midi_score(midi_score):
                raise ValueError("El MIDI generado está vacío o es inválido")
            
            # Guardar MIDI
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # symusic.Score tiene un método dump_midi para guardar
            midi_score.dump_midi(str(out_path))
            
            logger.info(f"MIDI guardado: {out_path}")
            return str(out_path)
        
        except Exception as e:
            logger.error(f"Error convirtiendo tokens a MIDI: {e}")
            raise
    
    def midi_to_tokens(self, midi_path: str) -> List[int]:
        """
        Convierte un archivo MIDI a secuencia de tokens.
        
        Útil para crear primers o entrenar modelos.
        
        Args:
            midi_path: Path al archivo MIDI
        
        Returns:
            Lista de IDs de tokens
        """
        try:
            # miditok 3.0+ usa 'encode' en lugar de 'midi_to_tokens'
            # y acepta directamente el path del archivo
            tokens = self.tokenizer.encode(midi_path)
            return tokens[0] if tokens else []
        
        except Exception as e:
            logger.error(f"Error convirtiendo MIDI a tokens: {e}")
            raise
    
    def _validate_midi_score(self, midi_score) -> bool:
        """
        Valida que un score de symusic tenga contenido válido.
        
        Args:
            midi_score: Objeto symusic.Score
        
        Returns:
            True si el MIDI es válido
        """
        # Verificar que tenga al menos un track
        if not hasattr(midi_score, 'tracks') or not midi_score.tracks:
            return False
        
        # Contar eventos de nota
        note_count = 0
        for track in midi_score.tracks:
            if hasattr(track, 'notes'):
                note_count += len(track.notes)
        
        # Debe tener al menos algunas notas
        return note_count > 0
    
    @property
    def vocab_size(self) -> int:
        """Retorna el tamaño del vocabulario."""
        return len(self.tokenizer.vocab)
    
    @property
    def pad_token(self) -> int:
        """Token de padding."""
        return self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0
    
    @property
    def eos_token(self) -> Optional[int]:
        """Token de fin de secuencia."""
        return self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None


def create_simple_primer(
    mode: str = 'major',
    pitch_low: int = 60,
    num_notes: int = 4,
    duration: float = 0.5
) -> str:
    """
    Crea un archivo MIDI simple para usar como primer.
    
    Args:
        mode: Modo musical ('major' o 'minor')
        pitch_low: Nota más grave (MIDI pitch)
        num_notes: Número de notas en el primer
        duration: Duración de cada nota (segundos)
    
    Returns:
        Path temporal al archivo MIDI del primer
    """
    import tempfile
    
    # Crear MIDI vacío
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Tempo 120 BPM
    track.append(mido.MetaMessage('set_tempo', tempo=500000))
    
    # Escala según modo
    if mode == 'major':
        scale_intervals = [0, 2, 4, 5, 7]  # Mayor
    else:
        scale_intervals = [0, 2, 3, 5, 7]  # Menor natural
    
    # Generar notas del primer
    time = 0
    ticks_per_beat = mid.ticks_per_beat
    note_ticks = int(duration * ticks_per_beat * 2)  # Asumiendo 120 BPM
    
    for i in range(min(num_notes, len(scale_intervals))):
        pitch = pitch_low + scale_intervals[i]
        
        # Note on
        track.append(mido.Message('note_on', note=pitch, velocity=80, time=time))
        # Note off
        track.append(mido.Message('note_off', note=pitch, velocity=0, time=note_ticks))
        
        time = 0  # Tiempo relativo para el siguiente evento
    
    # Guardar en archivo temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mid')
    mid.save(temp_file.name)
    
    return temp_file.name


def validate_midi_file(midi_path: str) -> Dict[str, any]:
    """
    Valida y retorna estadísticas de un archivo MIDI.
    
    Args:
        midi_path: Path al archivo MIDI
    
    Returns:
        Dict con estadísticas: num_tracks, num_notes, duration, etc.
    """
    try:
        mid = mido.MidiFile(midi_path)
        
        # Contar notas
        num_notes = 0
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    num_notes += 1
        
        # Calcular duración total
        duration = mid.length
        
        return {
            'valid': True,
            'num_tracks': len(mid.tracks),
            'num_notes': num_notes,
            'duration': duration,
            'ticks_per_beat': mid.ticks_per_beat
        }
    
    except Exception as e:
        logger.error(f"Error validando MIDI: {e}")
        return {
            'valid': False,
            'error': str(e)
        }
