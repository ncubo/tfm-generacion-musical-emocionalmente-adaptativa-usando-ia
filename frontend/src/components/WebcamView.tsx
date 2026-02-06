import { useEffect, useRef, useState } from 'react';
import { apiClient } from '../api/client';
import type { EmotionFromFrameResponse } from '../types';
import { useRealtimeEmotion } from '../hooks/useRealtimeEmotion';

type WebcamStatus = 'idle' | 'requesting' | 'active' | 'error';

// Opciones de frecuencia para tiempo real
const FPS_OPTIONS = [
  { label: '0.5 FPS (2s)', value: 2000 },
  { label: '1 FPS (1s)', value: 1000 },
  { label: '2 FPS (0.5s)', value: 500 },
] as const;

interface WebcamViewProps {
  onSnapshot?: (blob: Blob) => void;
  onEmotionDetected?: (emotion: EmotionFromFrameResponse) => void;
}

export function WebcamView({ onSnapshot, onEmotionDetected }: WebcamViewProps) {
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

      // Llamar callback para notificar la emoción detectada
      if (onEmotionDetected) {
        onEmotionDetected(result);
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
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Cámara Web</h2>

      {/* Estado: Solicitando acceso */}
      {status === 'requesting' && (
        <div className="flex items-center justify-center p-8 bg-blue-50 border border-blue-200 rounded">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-3"></div>
            <p className="text-blue-800 font-semibold">Solicitando acceso a la cámara…</p>
            <p className="text-sm text-blue-600 mt-1">Por favor, acepta los permisos en tu navegador</p>
          </div>
        </div>
      )}

      {/* Estado: Cámara activa */}
      {status === 'active' && (
        <div>
          <div className="mb-3 p-3 bg-green-50 border border-green-200 rounded">
            <p className="text-green-800 font-semibold">✓ Cámara activa</p>
            <p className="text-sm text-green-600">La cámara está transmitiendo correctamente</p>
          </div>
          
          <div className="relative bg-gray-900 rounded-lg overflow-hidden mb-4">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-auto min-h-[240px]"
              style={{ objectFit: 'cover' }}
              aria-label="Vista de la cámara web en tiempo real"
            />
          </div>

          {/* Canvas oculto para captura */}
          <canvas ref={canvasRef} className="hidden" />

          {/* Controles de modo tiempo real */}
          <div className="mb-4 p-4 bg-teal-50 border border-teal-200 rounded-lg">
            <h3 className="text-lg font-semibold text-teal-900 mb-3">Modo Tiempo Real</h3>
            
            <div className="space-y-3">
              {/* Estado actual */}
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Estado:</span>
                <span className={`text-sm font-bold ${realtimeState.isRunning ? 'text-teal-700' : 'text-gray-500'}`}>
                  {realtimeState.isRunning ? '🟢 Activado' : '⚪ Desactivado'}
                </span>
              </div>

              {/* Selector de frecuencia */}
              <div className="flex items-center justify-between">
                <label htmlFor="fps-selector" className="text-sm font-medium text-gray-700">
                  Frecuencia:
                </label>
                <select
                  id="fps-selector"
                  value={realtimeInterval}
                  onChange={(e) => setRealtimeInterval(Number(e.target.value))}
                  disabled={realtimeEnabled}
                  className="px-3 py-1 border border-gray-300 rounded bg-white text-sm disabled:bg-gray-100 disabled:cursor-not-allowed"
                >
                  {FPS_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Botones de control */}
              <div className="flex gap-2">
                {!realtimeEnabled ? (
                  <button
                    onClick={() => setRealtimeEnabled(true)}
                    className="flex-1 bg-teal-600 hover:bg-teal-700 text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
                    aria-label="Iniciar análisis en tiempo real"
                  >
                    Iniciar tiempo real
                  </button>
                ) : (
                  <button
                    onClick={() => setRealtimeEnabled(false)}
                    className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
                    aria-label="Detener análisis en tiempo real"
                  >
                    ⏹️ Detener tiempo real
                  </button>
                )}
              </div>

              {/* Indicador de análisis */}
              {/* {realtimeState.isAnalyzing && (
                <div className="flex items-center gap-2 text-sm text-teal-700">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-teal-600"></div>
                  <span>Analizando...</span>
                </div>
              )} */}
            </div>
          </div>

          {/* Resultados en tiempo real */}
          {realtimeState.isRunning && realtimeState.emotion && (
            <div className="mb-4 p-4 bg-teal-50 border border-teal-200 rounded">
              <h3 className="text-lg font-semibold text-teal-900 mb-3">📊 Emoción actual (tiempo real):</h3>
              
              {realtimeState.faceDetected ? (
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Emoción:</span>
                    <span className="text-xl font-bold text-teal-800 capitalize">
                      {realtimeState.emotion}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Valencia:</span>
                    <span className="text-lg font-semibold text-blue-700">
                      {realtimeState.valence?.toFixed(2)}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Activación:</span>
                    <span className="text-lg font-semibold text-orange-700">
                      {realtimeState.arousal?.toFixed(2)}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-yellow-700">
                  <p className="font-semibold">⚠️ No se detecta rostro</p>
                  <p className="text-sm mt-1">Asegúrate de estar frente a la cámara</p>
                </div>
              )}
            </div>
          )}

          {/* Error en tiempo real */}
          {realtimeState.error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded">
              <p className="text-red-800 font-semibold text-sm">Error en tiempo real:</p>
              <p className="text-sm text-red-600 mt-1">{realtimeState.error}</p>
            </div>
          )}

          <div className="my-4 border-t border-gray-300"></div>

          {/* Botones de captura manual */}
          <h3 className="text-lg font-semibold text-gray-800 mb-3">Captura Manual</h3>

          {/* Botones de captura */}
          <div className="flex gap-3 mb-4">
            <button
              onClick={captureSnapshot}
              className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
              aria-label="Capturar imagen de la cámara"
            >
              Capturar imagen
            </button>

            {snapshotBlob && (
              <>
                <button
                  onClick={analyzeImage}
                  disabled={analyzing}
                  className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
                  aria-label="Analizar emoción en la imagen"
                >
                  {analyzing ? 'Analizando...' : 'Analizar imagen'}
                </button>

                <button
                  onClick={clearSnapshot}
                  className="bg-gray-500 hover:bg-gray-600 text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
                  aria-label="Borrar imagen capturada"
                >
                  Borrar
                </button>
              </>
            )}
          </div>

          {/* Estado del snapshot */}
          {snapshotBlob ? (
            <div className="mb-3 p-3 bg-indigo-50 border border-indigo-200 rounded">
              <p className="text-indigo-800 font-semibold">✓ Imagen capturada</p>
              <p className="text-sm text-indigo-600">
                Tamaño: {(snapshotBlob.size / 1024).toFixed(2)} KB
              </p>
            </div>
          ) : (
            <div className="mb-3 p-3 bg-gray-50 border border-gray-200 rounded">
              <p className="text-gray-600 font-semibold">No hay imagen capturada</p>
              <p className="text-sm text-gray-500">Haz clic en "Capturar imagen" para tomar una foto</p>
            </div>
          )}

          {/* Resultados del análisis emocional */}
          {emotionResult && (
            <div className="mb-4 p-4 bg-purple-50 border border-purple-200 rounded">
              <h3 className="text-lg font-semibold text-purple-900 mb-3">Análisis emocional:</h3>
              
              {emotionResult.face_detected ? (
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Emoción:</span>
                    <span className="text-lg font-bold text-purple-800 capitalize">
                      {emotionResult.emotion}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Valencia:</span>
                    <span className="text-md font-semibold text-blue-700">
                      {emotionResult.valence.toFixed(2)}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Activación:</span>
                    <span className="text-md font-semibold text-orange-700">
                      {emotionResult.arousal.toFixed(2)}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-yellow-700">
                  <p className="font-semibold">No se detectó rostro</p>
                  <p className="text-sm mt-1">La imagen no contiene un rostro visible. Resultado neutral.</p>
                </div>
              )}
            </div>
          )}

          {/* Error de análisis */}
          {analysisError && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded">
              <p className="text-red-800 font-semibold">✗ Error al analizar</p>
              <p className="text-sm text-red-600 mt-1">{analysisError}</p>
            </div>
          )}

          {/* Preview de la imagen capturada */}
          {previewUrl && (
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Vista previa:</h3>
              <div className="relative bg-gray-100 rounded-lg overflow-hidden border border-gray-300">
                <img
                  src={previewUrl}
                  alt="Vista previa de la imagen capturada"
                  className="w-full h-auto"
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Estado: Error */}
      {status === 'error' && (
        <div className="p-4 bg-red-50 border border-red-200 rounded">
          <p className="text-red-800 font-semibold">✗ No se pudo acceder a la cámara</p>
          <p className="text-sm text-red-600 mt-1">{error}</p>
        </div>
      )}
    </div>
  );
}
