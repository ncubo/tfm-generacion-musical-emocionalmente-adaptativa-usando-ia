import { useEffect, useRef, useState } from 'react';
import type { EmotionFromFrameResponse } from '../../types';
import { MidiGenerator } from '../midi/MidiGenerator';
import { CameraView } from './CameraView';
import { RealtimeCapture } from '../emotion/RealtimeCapture';
import { useRealtimeEmotion } from '../../hooks/useRealtimeEmotion';
import { useAutoMidiGeneration } from '../../hooks/useAutoMidiGeneration';

type WebcamStatus = 'idle' | 'requesting' | 'active' | 'error';
type MusicEngine = 'baseline' | 'transformer_pretrained' | 'transformer_finetuned';

interface WebcamViewProps {
  onSnapshot?: (blob: Blob) => void;
}

// Opciones de frecuencia para captura en tiempo real
const FPS_OPTIONS = [
  { value: 1000, label: '1 captura/seg (1 FPS)' },
  { value: 2000, label: '1 captura/2s (0.5 FPS)' },
  { value: 3000, label: '1 captura/3s' },
  { value: 5000, label: '1 captura/5s' },
] as const;

export function WebcamView({ onSnapshot: _onSnapshot }: WebcamViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [status, setStatus] = useState<WebcamStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [emotionResult] = useState<EmotionFromFrameResponse | null>(null);

  // Estados para modo tiempo real
  const [realtimeEnabled, setRealtimeEnabled] = useState(false);
  const [realtimeInterval, setRealtimeInterval] = useState(2000); // Default 2s

  // Estados para modo Auto MIDI
  const [autoMidiEnabled, setAutoMidiEnabled] = useState(false);
  const [selectedEngine, setSelectedEngine] = useState<MusicEngine>('transformer_finetuned');
  const [lengthBars, setLengthBars] = useState(4);
  const [deltaValenceThreshold, setDeltaValenceThreshold] = useState(0.2);
  const [deltaArousalThreshold, setDeltaArousalThreshold] = useState(0.2);
  const [cooldownMs, setCooldownMs] = useState(4000);

  // Estados para MIDI generado
  const [midiUrl, setMidiUrl] = useState<string | null>(null);
  const [generatedInfo, setGeneratedInfo] = useState<{
    engine: string;
    seed: number;
    lengthBars: number;
  } | null>(null);
  const [autoMidiError, setAutoMidiError] = useState<string | null>(null);

  // Hook para detección emocional en tiempo real
  const realtimeEmotionState = useRealtimeEmotion({
    videoRef,
    canvasRef,
    intervalMs: realtimeInterval,
    enabled: realtimeEnabled,
  });

  // Hook para generación automática de MIDI
  const autoMidiState = useAutoMidiGeneration({
    enabled: autoMidiEnabled,
    currentEmotion: {
      valence: realtimeEmotionState.valence,
      arousal: realtimeEmotionState.arousal,
      emotion: realtimeEmotionState.emotion,
    },
    engine: selectedEngine,
    lengthBars: autoMidiEnabled ? 4 : lengthBars, // Siempre 4 compases en modo auto
    deltaValenceThreshold,
    deltaArousalThreshold,
    cooldownMs,
    onMidiGenerated: result => {
      // Limpiar MIDI anterior
      if (midiUrl) {
        URL.revokeObjectURL(midiUrl);
      }

      // Crear URL para el nuevo MIDI
      const url = URL.createObjectURL(result.midiBlob);
      setMidiUrl(url);

      setGeneratedInfo({
        engine: result.engine,
        seed: result.seed,
        lengthBars: result.length_bars,
      });

      setAutoMidiError(null);

      console.log(`[WebcamView] MIDI auto-generado: engine=${result.engine}, seed=${result.seed}`);
    },
    onError: error => {
      setAutoMidiError(error);
      console.error('[WebcamView] Error al auto-generar MIDI:', error);
    },
  });

  useEffect(() => {
    // Solicitar acceso a la cámara al montar el componente
    const startWebcam = async () => {
      setStatus('requesting');
      setError(null);

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
        });

        // Guardar referencia del stream para limpieza posterior
        streamRef.current = stream;

        // Cambiar estado a 'active' para renderizar el elemento video
        setStatus('active');
      } catch (err) {
        console.error('Error al acceder a la cámara:', err);

        // Mensajes de error específicos según el tipo
        if (err instanceof Error) {
          if (err.name === 'NotAllowedError') {
            setError('Permiso denegado. Por favor, permite el acceso a la cámara.');
          } else if (err.name === 'NotFoundError') {
            setError('No se encontró ninguna cámara en este dispositivo.');
          } else {
            setError(`Error al acceder a la cámara: ${err.message}`);
          }
        } else {
          setError('Error desconocido al acceder a la cámara.');
        }

        setStatus('error');
      }
    };

    startWebcam();

    // Cleanup: liberar recursos al desmontar el componente
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    };
  }, []); // Solo ejecutar una vez al montar

  // Asignar stream al video cuando esté en el DOM
  useEffect(() => {
    if (status === 'active' && streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [status]); // Ejecutar cuando cambie el status

  return (
    <div className="space-y-6">
      {/* Grid principal: cámara + captura manual */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Columna izquierda: Cámara */}
        <CameraView status={status} error={error} videoRef={videoRef} canvasRef={canvasRef} />

        {/* Columna derecha: Captura manual y análisis */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-bold mb-4 text-gray-800">Control de Captura</h2>

          {status === 'active' && (
            <>
              {/* Selector de Motor de Generación */}
              <div className="mb-4">
                <label
                  htmlFor="engine-select"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Motor de Generación Musical
                </label>
                <select
                  id="engine-select"
                  value={selectedEngine}
                  onChange={e => setSelectedEngine(e.target.value as MusicEngine)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg bg-white text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                >
                  <option value="transformer_finetuned">Transformer (fine-tuned)</option>
                  <option value="transformer_pretrained">HF Maestro-REMI</option>
                  <option value="baseline">Baseline (Reglas)</option>
                </select>
              </div>

              <div className="mb-4 border-t border-gray-300"></div>

              {/* Modo Tiempo Real + Auto MIDI */}
              <RealtimeCapture
                realtimeState={realtimeEmotionState}
                realtimeEnabled={realtimeEnabled}
                realtimeInterval={realtimeInterval}
                onToggleRealtime={setRealtimeEnabled}
                onIntervalChange={setRealtimeInterval}
                fpsOptions={FPS_OPTIONS}
                autoMidiState={{
                  enabled: autoMidiEnabled,
                  isGenerating: autoMidiState.isGenerating,
                  generationCount: autoMidiState.generationCount,
                  timeSinceLastGeneration: autoMidiState.timeSinceLastGeneration,
                  deltaValenceThreshold,
                  deltaArousalThreshold,
                  cooldownMs,
                }}
                onToggleAutoMidi={setAutoMidiEnabled}
                onThresholdChange={(type, value) => {
                  if (type === 'valence') {
                    setDeltaValenceThreshold(value);
                  } else {
                    setDeltaArousalThreshold(value);
                  }
                }}
                onCooldownChange={setCooldownMs}
                selectedEngine={selectedEngine}
              />

              <div className="my-4 border-t border-gray-300"></div>

              {/* Modo Manual */}
              {/* <ManualCapture
                snapshotBlob={snapshotBlob}
                previewUrl={previewUrl}
                analyzing={analyzing}
                emotionResult={emotionResult}
                analysisError={analysisError}
                onCapture={captureSnapshot}
                onAnalyze={analyzeImage}
                onClear={clearSnapshot}
              /> */}
            </>
          )}
        </div>
      </div>

      {/* Generador MIDI - Ancho completo abajo */}
      {status === 'active' && (emotionResult || realtimeEmotionState.emotion) && (
        <div className="bg-white rounded-lg shadow-md">
          <MidiGenerator
            emotionData={
              emotionResult || {
                emotion: realtimeEmotionState.emotion!,
                valence: realtimeEmotionState.valence!,
                arousal: realtimeEmotionState.arousal!,
                face_detected: realtimeEmotionState.faceDetected,
              }
            }
            selectedEngine={selectedEngine}
            onEngineChange={setSelectedEngine}
            lengthBars={lengthBars}
            onLengthBarsChange={setLengthBars}
            midiUrl={midiUrl}
            generatedInfo={generatedInfo}
            onMidiGenerated={result => {
              // Manejar generación manual
              if (midiUrl) {
                URL.revokeObjectURL(midiUrl);
              }
              const url = URL.createObjectURL(result.midiBlob);
              setMidiUrl(url);
              setGeneratedInfo({
                engine: result.engine,
                seed: result.seed,
                lengthBars: result.length_bars,
              });
            }}
            isAutoGenerating={autoMidiState.isGenerating}
            autoPlayMidi={autoMidiEnabled}
            autoMidiEnabled={autoMidiEnabled}
          />

          {/* Mostrar error de auto-generación si existe */}
          {autoMidiError && (
            <div className="px-6 pb-4">
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-red-800 text-sm font-semibold">
                  Error en generación automática:
                </p>
                <p className="text-red-700 text-sm mt-1">{autoMidiError}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
