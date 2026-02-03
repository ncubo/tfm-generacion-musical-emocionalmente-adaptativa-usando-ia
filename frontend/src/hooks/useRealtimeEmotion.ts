import { useEffect, useRef, useState, useCallback } from 'react';
import { apiClient } from '../api/client';

interface UseRealtimeEmotionOptions {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  intervalMs: number; // Intervalo entre capturas (ms)
  enabled: boolean; // Si el modo tiempo real está activado
}

interface RealtimeEmotionState {
  emotion: string | null;
  valence: number | null;
  arousal: number | null;
  faceDetected: boolean;
  isAnalyzing: boolean;
  error: string | null;
  isRunning: boolean;
}

/**
 * Hook para análisis emocional en tiempo real desde webcam.
 * 
 * Implementa un sistema de polling que:
 * - Captura snapshots periódicos del video
 * - Los envía al backend para inferencia
 * - Evita concurrencia (no envía si hay request en vuelo)
 * - Gestiona errores de forma robusta
 * 
 * ¿Por qué polling y no streaming?
 * - Simplicidad: no requiere WebSockets ni infraestructura compleja
 * - Control: fácil ajustar frecuencia según rendimiento
 * - Compatibilidad: funciona con cualquier backend HTTP estándar
 * 
 * ¿Por qué evitar concurrencia?
 * - Previene saturación del backend con requests acumuladas
 * - Mantiene orden temporal de las emociones detectadas
 * - Evita condiciones de carrera en actualización de estado
 */
export function useRealtimeEmotion({
  videoRef,
  canvasRef,
  intervalMs,
  enabled,
}: UseRealtimeEmotionOptions): RealtimeEmotionState {
  const [state, setState] = useState<RealtimeEmotionState>({
    emotion: null,
    valence: null,
    arousal: null,
    faceDetected: false,
    isAnalyzing: false,
    error: null,
    isRunning: false,
  });

  // Ref para evitar requests concurrentes
  const isProcessingRef = useRef(false);
  const intervalIdRef = useRef<number | null>(null);

  /**
   * Captura un snapshot del video y lo convierte a Blob.
   * Reutiliza la misma lógica del componente WebcamView.
   */
  const captureSnapshotBlob = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas) {
        console.error('[Realtime] Video o canvas no disponibles');
        resolve(null);
        return;
      }

      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      if (videoWidth === 0 || videoHeight === 0) {
        console.error('[Realtime] Video no tiene dimensiones válidas');
        resolve(null);
        return;
      }

      // Configurar canvas
      canvas.width = videoWidth;
      canvas.height = videoHeight;

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        console.error('[Realtime] No se pudo obtener contexto 2D');
        resolve(null);
        return;
      }

      // Dibujar frame actual
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

      // Convertir a Blob
      canvas.toBlob(
        (blob) => {
          resolve(blob);
        },
        'image/jpeg',
        0.85 // Calidad ligeramente menor para tiempo real (menos peso)
      );
    });
  }, [videoRef, canvasRef]);

  /**
   * Procesa un frame: captura snapshot y lo envía al backend.
   * Solo ejecuta si no hay otra request en vuelo (evita concurrencia).
   */
  const processFrame = useCallback(async () => {
    // Evitar concurrencia: si ya hay una request en proceso, skip
    if (isProcessingRef.current) {
      console.log('[Realtime] Request en vuelo, saltando frame');
      return;
    }

    // Marcar como procesando
    isProcessingRef.current = true;
    setState(prev => ({ ...prev, isAnalyzing: true, error: null }));

    try {
      // Capturar snapshot
      const blob = await captureSnapshotBlob();

      if (!blob) {
        throw new Error('No se pudo capturar snapshot');
      }

      // Enviar al backend
      const result = await apiClient.analyzeImageEmotion(blob);

      // Actualizar estado con resultado
      setState(prev => ({
        ...prev,
        emotion: result.emotion,
        valence: result.valence,
        arousal: result.arousal,
        faceDetected: result.face_detected,
        isAnalyzing: false,
        error: null,
      }));

    } catch (err) {
      console.error('[Realtime] Error al procesar frame:', err);
      setState(prev => ({
        ...prev,
        isAnalyzing: false,
        error: err instanceof Error ? err.message : 'Error desconocido',
      }));
    } finally {
      // Liberar flag de procesamiento
      isProcessingRef.current = false;
    }
  }, [captureSnapshotBlob]);

  /**
   * Effect: gestiona el intervalo de captura periódica.
   * Se activa/desactiva según el flag 'enabled'.
   */
  useEffect(() => {
    if (!enabled) {
      // Limpiar intervalo si está desactivado
      if (intervalIdRef.current) {
        clearInterval(intervalIdRef.current);
        intervalIdRef.current = null;
      }
      setState(prev => ({ ...prev, isRunning: false }));
      isProcessingRef.current = false;
      return;
    }

    // Activar modo tiempo real
    setState(prev => ({ ...prev, isRunning: true }));

    // Procesar primer frame inmediatamente
    processFrame();

    // Configurar intervalo para frames subsecuentes
    intervalIdRef.current = setInterval(() => {
      processFrame();
    }, intervalMs);

    // Cleanup: detener intervalo al desmontar o cambiar dependencias
    return () => {
      if (intervalIdRef.current) {
        clearInterval(intervalIdRef.current);
        intervalIdRef.current = null;
      }
      isProcessingRef.current = false;
      setState(prev => ({ ...prev, isRunning: false }));
    };
  }, [enabled, intervalMs, processFrame]);

  return state;
}
