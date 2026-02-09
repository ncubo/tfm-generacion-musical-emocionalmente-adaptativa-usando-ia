import { useEffect, useRef, useState } from 'react';
import { apiClient } from '../../api/client';
import type { EmotionFromFrameResponse } from '../../types';
import { useRealtimeEmotion } from '../../hooks/useRealtimeEmotion';
import { MidiGenerator } from '../midi/MidiGenerator';
import { CameraView } from './CameraView';
import { RealtimeCapture } from '../emotion/RealtimeCapture';
import { ManualCapture } from '../emotion/ManualCapture';

type WebcamStatus = 'idle' | 'requesting' | 'active' | 'error';

// Opciones de frecuencia para tiempo real
const FPS_OPTIONS = [
  { label: '0.5 FPS (2s)', value: 2000 },
  { label: '1 FPS (1s)', value: 1000 },
  { label: '2 FPS (0.5s)', value: 500 },
] as const;

interface WebcamViewProps {
  onSnapshot?: (blob: Blob) => void;
}

export function WebcamView({ onSnapshot }: WebcamViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  const [status, setStatus] = useState<WebcamStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [snapshotBlob, setSnapshotBlob] = useState<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  
  // Estados para análisis emocional
  const [analyzing, setAnalyzing] = useState(false);
  const [emotionResult, setEmotionResult] = useState<EmotionFromFrameResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Estados para modo tiempo real
  const [realtimeEnabled, setRealtimeEnabled] = useState(false);
  const [realtimeInterval, setRealtimeInterval] = useState(1000); // 1 FPS por defecto

  // Hook de tiempo real
  const realtimeState = useRealtimeEmotion({
    videoRef,
    canvasRef,
    intervalMs: realtimeInterval,
    enabled: realtimeEnabled,
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
            height: { ideal: 480 }
          } 
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

  // Cleanup del preview URL cuando cambia o al desmontar
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const captureSnapshot = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) {
      console.error('Video o canvas no disponibles');
      return;
    }

    // Obtener dimensiones reales del video
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    if (videoWidth === 0 || videoHeight === 0) {
      console.error('Video no tiene dimensiones válidas');
      return;
    }

    // Configurar canvas con las dimensiones del video
    canvas.width = videoWidth;
    canvas.height = videoHeight;

    // Dibujar frame actual del video en el canvas
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('No se pudo obtener contexto 2D del canvas');
      return;
    }

    ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

    // Convertir canvas a Blob
    canvas.toBlob(
      (blob) => {
        if (blob) {
          // Revocar URL anterior si existe
          if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
          }

          // Crear nueva URL para preview
          const newPreviewUrl = URL.createObjectURL(blob);
          
          setSnapshotBlob(blob);
          setPreviewUrl(newPreviewUrl);

          // Llamar callback si existe
          if (onSnapshot) {
            onSnapshot(blob);
          }
        }
      },
      'image/jpeg',
      0.95 // Calidad JPEG
    );
  };

  const clearSnapshot = () => {
    // Revocar URL del preview
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    
    setSnapshotBlob(null);
    setPreviewUrl(null);
    setEmotionResult(null);
    setAnalysisError(null);
  };

  const analyzeImage = async () => {
    if (!snapshotBlob) {
      setAnalysisError('No hay imagen capturada para analizar');
      return;
    }

    setAnalyzing(true);
    setAnalysisError(null);

    try {
      // Enviar imagen al backend
      const result = await apiClient.analyzeImageEmotion(snapshotBlob);
      setEmotionResult(result);

      // Llamar callback si existe
      if (onSnapshot) {
        onSnapshot(snapshotBlob);
      }
    } catch (err) {
      console.error('Error al analizar imagen:', err);
      setAnalysisError(
        err instanceof Error ? err.message : 'Error al analizar la imagen'
      );
      setEmotionResult(null);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Grid principal: cámara + captura manual */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Columna izquierda: Cámara */}
        <CameraView
          status={status}
          error={error}
          videoRef={videoRef}
          canvasRef={canvasRef}
        />

        {/* Columna derecha: Captura manual y análisis */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-2xl font-bold mb-4 text-gray-800">Captura Manual</h2>

          {status === 'active' && (
            <>
              {/* <RealtimeCapture
                realtimeState={realtimeState}
                realtimeEnabled={realtimeEnabled}
                realtimeInterval={realtimeInterval}
                onToggleRealtime={setRealtimeEnabled}
                onIntervalChange={setRealtimeInterval}
                fpsOptions={FPS_OPTIONS}
              />

              <div className="my-4 border-t border-gray-300"></div> */}

              <ManualCapture
                snapshotBlob={snapshotBlob}
                previewUrl={previewUrl}
                analyzing={analyzing}
                emotionResult={emotionResult}
                analysisError={analysisError}
                onCapture={captureSnapshot}
                onAnalyze={analyzeImage}
                onClear={clearSnapshot}
              />
            </>
          )}
        </div>
      </div>

      {/* Generador MIDI - Ancho completo abajo */}
      {status === 'active' && emotionResult && (
        <div className="bg-white rounded-lg shadow-md">
          <MidiGenerator emotionData={emotionResult} />
        </div>
      )}
    </div>
  );
}
