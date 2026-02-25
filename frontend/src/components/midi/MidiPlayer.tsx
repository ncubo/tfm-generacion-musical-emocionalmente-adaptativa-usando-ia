import { useEffect, useRef, useState } from 'react';
import { Play, Pause, Loader2 } from 'lucide-react';
import Soundfont from 'soundfont-player';
import { Midi } from '@tonejs/midi';

interface MidiPlayerProps {
  midiUrl: string;
  autoPlay?: boolean; // Si true, reproduce automáticamente al cargar
}

interface ScheduledNote {
  time: number;
  note: string;
  duration: number;
  velocity: number;
}

export function MidiPlayer({ midiUrl, autoPlay = false }: MidiPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [instrumentLoading, setInstrumentLoading] = useState(true);
  const [midiReady, setMidiReady] = useState(false);

  const instrumentRef = useRef<Soundfont.Player | null>(null);
  const midiDataRef = useRef<Midi | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const scheduledNotesRef = useRef<ScheduledNote[]>([]);
  const playbackStartTimeRef = useRef<number>(0);
  const animationFrameRef = useRef<number | null>(null);
  const isPlayingRef = useRef<boolean>(false); // Ref para controlar el loop de progreso
  const pausedAtRef = useRef<number>(0); // Tiempo en segundos donde se pausó
  const scheduledNotesIdsRef = useRef<unknown[]>([]); // IDs de notas programadas (para poder cancelarlas)
  const shouldAutoPlayRef = useRef<boolean>(autoPlay); // Trackear si debemos auto-play

  // Actualizar la ref cuando cambie autoPlay
  useEffect(() => {
    shouldAutoPlayRef.current = autoPlay;
  }, [autoPlay, midiUrl]); // También actualizar cuando cambie el midiUrl

  useEffect(() => {
    // Inicializar AudioContext y cargar instrumento de piano
    const initAudio = async () => {
      try {
        setInstrumentLoading(true);

        // Crear AudioContext
        audioContextRef.current = new AudioContext();

        // Cargar soundfont de piano acústico
        instrumentRef.current = await Soundfont.instrument(
          audioContextRef.current,
          'acoustic_grand_piano',
          {
            soundfont: 'MusyngKite', // SoundFont de alta calidad
          }
        );

        console.log('Piano cargado exitosamente');
        setInstrumentLoading(false);
      } catch (err) {
        console.error('Error al cargar instrumento:', err);
        setError('Error al cargar el instrumento de piano');
        setInstrumentLoading(false);
      }
    };

    initAudio();

    // Resetear estados y cargar nuevo MIDI
    setMidiReady(false);
    stopPlayback();
    loadMidi();

    // Cleanup
    return () => {
      stopPlayback();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [midiUrl]);

  const loadMidi = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Descargar archivo MIDI
      const response = await fetch(midiUrl);
      const arrayBuffer = await response.arrayBuffer();

      // Parsear MIDI
      const midi = new Midi(arrayBuffer);
      midiDataRef.current = midi;
      setDuration(midi.duration);

      // Preparar notas para reproducción
      const notes: ScheduledNote[] = [];
      midi.tracks.forEach(track => {
        track.notes.forEach(note => {
          notes.push({
            time: note.time,
            note: note.name,
            duration: note.duration,
            velocity: note.velocity,
          });
        });
      });

      // Ordenar por tiempo
      notes.sort((a, b) => a.time - b.time);
      scheduledNotesRef.current = notes;

      console.log('MIDI cargado:', {
        nombre: midi.name,
        duración: midi.duration,
        pistas: midi.tracks.length,
        notas: notes.length,
      });

      setMidiReady(true);
    } catch (err) {
      console.error('Error al cargar MIDI:', err);
      setError(err instanceof Error ? err.message : 'Error al cargar MIDI');
      setMidiReady(false);
    } finally {
      setIsLoading(false);
    }
  };

  const pausePlayback = () => {
    if (!audioContextRef.current) return;

    // Guardar posición actual
    const currentTime = audioContextRef.current.currentTime - playbackStartTimeRef.current;
    pausedAtRef.current = currentTime;

    isPlayingRef.current = false;
    setIsPlaying(false);

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Detener todas las notas programadas
    if (instrumentRef.current) {
      instrumentRef.current.stop();
    }

    // Limpiar notas programadas
    scheduledNotesIdsRef.current = [];
  };

  const stopPlayback = () => {
    pausePlayback();
    setProgress(0);
    pausedAtRef.current = 0;
  };

  const playMidi = async () => {
    if (!midiDataRef.current || !instrumentRef.current || !audioContextRef.current) {
      setError('MIDI o instrumento no cargado');
      return;
    }

    if (instrumentLoading) {
      setError('Instrumento aún cargando...');
      return;
    }

    try {
      // Si estaba reproduciéndose, pausar
      if (isPlaying) {
        pausePlayback();
        return;
      }

      // Reanudar AudioContext si está suspendido (requisito del navegador)
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      const instrument = instrumentRef.current;
      const audioContext = audioContextRef.current;
      const notes = scheduledNotesRef.current;

      // Tiempo de inicio (ajustado si es resume)
      const startTime = audioContext.currentTime;
      const resumeOffset = pausedAtRef.current; // Offset de donde se pausó
      playbackStartTimeRef.current = startTime - resumeOffset;

      // Programar solo las notas que aún no han sonado
      const scheduledIds: unknown[] = [];
      notes.forEach(note => {
        // Solo programar notas que aún no han pasado
        if (note.time >= resumeOffset) {
          const scheduleTime = startTime + (note.time - resumeOffset);
          const nodeId = instrument.play(note.note, scheduleTime, {
            duration: note.duration,
            gain: note.velocity,
          });
          scheduledIds.push(nodeId);
        }
      });

      scheduledNotesIdsRef.current = scheduledIds;

      isPlayingRef.current = true;
      setIsPlaying(true);
      setError(null);

      // Actualizar progreso
      const totalDuration = duration; // Capturar duración en el closure
      const updateProgress = () => {
        if (!isPlayingRef.current || !audioContextRef.current) return;

        const currentTime = audioContextRef.current.currentTime - playbackStartTimeRef.current;
        const newProgress = (currentTime / totalDuration) * 100;

        setProgress(Math.min(newProgress, 100));

        // Detener al finalizar
        if (currentTime >= totalDuration) {
          stopPlayback();
        } else {
          animationFrameRef.current = requestAnimationFrame(updateProgress);
        }
      };

      updateProgress();
    } catch (err) {
      console.error('Error al reproducir MIDI:', err);
      setError(err instanceof Error ? err.message : 'Error al reproducir');
      stopPlayback();
    }
  };

  // UseEffect para auto-play cuando todo está listo (instrumento + MIDI)
  useEffect(() => {
    if (
      shouldAutoPlayRef.current &&
      !instrumentLoading &&
      midiReady &&
      instrumentRef.current &&
      audioContextRef.current &&
      !isPlaying &&
      !isLoading
    ) {
      console.log('[MidiPlayer] Auto-play activado - reproduciendo...');
      // Pequeño delay para asegurar que todo está estabilizado
      const timeout = setTimeout(() => {
        playMidi();
      }, 300);

      return () => clearTimeout(timeout);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [instrumentLoading, midiReady, isPlaying, isLoading]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-gray-50 rounded-lg p-6">
      {/* Controles principales */}
      <div className="flex items-center space-x-4 mb-4">
        {/* Botón Play/Pause */}
        <button
          onClick={playMidi}
          disabled={isLoading || instrumentLoading || !!error}
          className={`w-14 h-14 rounded-full flex items-center justify-center transition-colors ${
            isLoading || instrumentLoading || error
              ? 'bg-gray-400 cursor-not-allowed'
              : isPlaying
                ? 'bg-red-500 hover:bg-red-600'
                : 'bg-blue-600 hover:bg-blue-700'
          } text-white shadow-lg`}
        >
          {isLoading || instrumentLoading ? (
            <Loader2 className="h-6 w-6 animate-spin" />
          ) : isPlaying ? (
            <Pause className="h-6 w-6" />
          ) : (
            <Play className="h-6 w-6 ml-1" />
          )}
        </button>

        {/* Info */}
        <div className="flex-1">
          <p className="text-xs text-gray-500 mt-1">
            {duration > 0 && !instrumentLoading && (
              <>
                {formatTime((progress / 100) * duration)} / {formatTime(duration)}
                {' • '}
                {scheduledNotesRef.current.length} notas
              </>
            )}
          </p>
        </div>
      </div>

      {/* Barra de progreso */}
      {duration > 0 && (
        <div className="mb-4">
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              key={`progress-${progress}`}
              className="h-full bg-blue-600 transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}

      {/* Info del MIDI */}
      {midiDataRef.current && (
        <div className="mt-4 text-xs text-gray-600 space-y-1">
          <div className="flex justify-between">
            <span>Pistas:</span>
            <span className="font-medium">{midiDataRef.current.tracks.length}</span>
          </div>
          <div className="flex justify-between">
            <span>Tempo:</span>
            <span className="font-medium">
              {midiDataRef.current.header.tempos[0]?.bpm.toFixed(0) || 120} BPM
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
