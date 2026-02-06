import { useState, useEffect, useRef, useCallback } from 'react';
import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';
import { Play, Pause, Square, Loader2 } from 'lucide-react';

interface MidiPlayerProps {
  midiBase64: string;
  onError?: (error: string) => void;
}

export function MidiPlayer({ midiBase64, onError }: MidiPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  
  const synthsRef = useRef<Tone.PolySynth[]>([]);
  const partsRef = useRef<Tone.Part[]>([]);
  const midiRef = useRef<Midi | null>(null);
  const animationFrameRef = useRef<number>(0);

  const cleanupSynths = useCallback(() => {
    // Detener y limpiar Parts
    partsRef.current.forEach(part => {
      part.stop();
      part.dispose();
    });
    partsRef.current = [];
    
    // Limpiar sintetizadores
    synthsRef.current.forEach(synth => {
      synth.releaseAll();
      synth.dispose();
    });
    synthsRef.current = [];
  }, []);

  const stopPlayback = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    Tone.Transport.stop();
    Tone.Transport.position = 0; // Resetear posición sin cancelar eventos
    setIsPlaying(false);
    setCurrentTime(0);
  }, []);

  // Decodificar y preparar MIDI
  const loadMidi = useCallback(async () => {
    try {
      setIsLoading(true);
      
      // Decodificar base64 a ArrayBuffer
      const binaryString = atob(midiBase64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Parsear MIDI
      const midi = new Midi(bytes.buffer);
      midiRef.current = midi;
      setDuration(midi.duration);
      
      // Limpiar sintetizadores anteriores
      cleanupSynths();
      
      // Crear un sintetizador para cada track
      midi.tracks.forEach((track) => {
        const synth = new Tone.PolySynth(Tone.Synth, {
          envelope: {
            attack: 0.02,
            decay: 0.1,
            sustain: 0.3,
            release: 1,
          },
        }).toDestination();
        
        synthsRef.current.push(synth);
        
        // Crear Part para programar las notas
        const part = new Tone.Part((time, note) => {
          synth.triggerAttackRelease(
            note.name,
            note.duration,
            time,
            note.velocity
          );
        }, track.notes.map(note => ({
          time: note.time,
          name: note.name,
          duration: note.duration,
          velocity: note.velocity,
        }))).start(0);
        
        partsRef.current.push(part);
      });
      
      setIsLoading(false);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Error al cargar MIDI';
      onError?.(message);
      setIsLoading(false);
    }
  }, [midiBase64, cleanupSynths, onError]);

  // Limpiar recursos al desmontar o cambiar MIDI
  useEffect(() => {
    // Limpiar y recargar cuando cambia el MIDI
    stopPlayback();
    cleanupSynths();
    midiRef.current = null;
    setDuration(0);
    setCurrentTime(0);
    
    // Cargar el nuevo MIDI automáticamente
    if (midiBase64) {
      loadMidi();
    }
    
    return () => {
      stopPlayback();
      cleanupSynths();
    };
  }, [midiBase64, loadMidi, stopPlayback, cleanupSynths]);

  const updateProgress = () => {
    setCurrentTime(Tone.Transport.seconds);
    
    if (Tone.Transport.state === 'started') {
      animationFrameRef.current = requestAnimationFrame(updateProgress);
    }
  };

  const handlePlay = async () => {
    if (!midiRef.current) {
      await loadMidi();
    }
    
    // Inicializar AudioContext (requerido por navegadores)
    await Tone.start();
    
    if (isPlaying) {
      // Pausar
      Tone.Transport.pause();
      setIsPlaying(false);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    } else {
      // Reproducir
      Tone.Transport.start();
      setIsPlaying(true);
      updateProgress();
    }
  };

  const handleStop = () => {
    stopPlayback();
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = parseFloat(e.target.value);
    Tone.Transport.seconds = newTime;
    setCurrentTime(newTime);
  };

  // Evento cuando termina la reproducción
  useEffect(() => {
    if (currentTime >= duration && duration > 0 && isPlaying) {
      stopPlayback();
    }
  }, [currentTime, duration, isPlaying]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="p-4 bg-linear-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700">Reproductor MIDI</h3>
        <span className="text-xs text-gray-500">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>

      {/* Barra de progreso */}
      <input
        type="range"
        min="0"
        max={duration || 100}
        value={currentTime}
        onChange={handleSeek}
        disabled={!midiRef.current || isLoading}
        className="w-full mb-4 h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
        style={{
          background: `linear-gradient(to right, #9333ea 0%, #9333ea ${(currentTime / duration) * 100}%, #d1d5db ${(currentTime / duration) * 100}%, #d1d5db 100%)`
        }}
      />

      {/* Controles */}
      <div className="flex gap-2">
        <button
          onClick={handlePlay}
          disabled={isLoading}
          className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded transition-colors duration-200 flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Cargando...
            </>
          ) : isPlaying ? (
            <>
              <Pause className="w-4 h-4" />
              Pausar
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Reproducir
            </>
          )}
        </button>
        
        <button
          onClick={handleStop}
          disabled={!isPlaying && currentTime === 0}
          className="bg-red-600 hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded transition-colors duration-200 flex items-center justify-center gap-2"
        >
          <Square className="w-4 h-4" />
          Detener
        </button>
      </div>

      {/* Info de tracks */}
      {midiRef.current && (
        <div className="mt-3 text-xs text-gray-600">
          {midiRef.current.tracks.length} track(s) · {Math.floor(duration)}s
        </div>
      )}
    </div>
  );
}
