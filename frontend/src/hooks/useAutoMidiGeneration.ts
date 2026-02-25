import { useEffect, useRef, useCallback, useState } from 'react';
import { apiClient } from '../api/client';

interface AutoMidiGenerationOptions {
  enabled: boolean; // Si el modo auto está activado
  currentEmotion: {
    valence: number | null;
    arousal: number | null;
    emotion: string | null;
  };
  engine: 'baseline' | 'transformer_pretrained' | 'transformer_finetuned';
  lengthBars: number;
  deltaValenceThreshold: number; // Umbral de cambio en valence (default 0.2)
  deltaArousalThreshold: number; // Umbral de cambio en arousal (default 0.2)
  cooldownMs: number; // Tiempo mínimo entre generaciones (default 4000ms)
  onMidiGenerated: (result: {
    midiBlob: Blob;
    engine: string;
    seed: number;
    length_bars: number;
  }) => void;
  onError: (error: string) => void;
}

interface AutoMidiGenerationState {
  isGenerating: boolean;
  lastGeneratedValence: number | null;
  lastGeneratedArousal: number | null;
  lastGeneratedTimestamp: number | null;
  generationCount: number;
  timeSinceLastGeneration: number | null; // en segundos
}

/**
 * Hook para auto-generación de MIDI basada en cambios emocionales.
 *
 * Características:
 * - Detecta cambios significativos en valence/arousal usando thresholds configurables
 * - Implementa cooldown para evitar spam de generaciones
 * - Previene condiciones de carrera (solo una generación a la vez)
 * - Actualiza contador de tiempo desde última generación
 * - Soporte para AbortController para cancelación
 */
export function useAutoMidiGeneration({
  enabled,
  currentEmotion,
  engine,
  lengthBars,
  deltaValenceThreshold,
  deltaArousalThreshold,
  cooldownMs,
  onMidiGenerated,
  onError,
}: AutoMidiGenerationOptions): AutoMidiGenerationState {
  const [state, setState] = useState<AutoMidiGenerationState>({
    isGenerating: false,
    lastGeneratedValence: null,
    lastGeneratedArousal: null,
    lastGeneratedTimestamp: null,
    generationCount: 0,
    timeSinceLastGeneration: null,
  });

  // Ref para prevenir condiciones de carrera
  const isGeneratingRef = useRef(false);

  // Ref para AbortController
  const abortControllerRef = useRef<AbortController | null>(null);

  // Ref para intervalo de actualización de tiempo
  const intervalRef = useRef<number | null>(null);

  /**
   * Actualiza el tiempo transcurrido desde la última generación cada segundo
   */
  useEffect(() => {
    if (state.lastGeneratedTimestamp) {
      intervalRef.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - state.lastGeneratedTimestamp!) / 1000);
        setState(prev => ({ ...prev, timeSinceLastGeneration: elapsed }));
      }, 1000);

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      };
    }
  }, [state.lastGeneratedTimestamp]);

  /**
   * Verifica si se cumplen los criterios para disparar una nueva generación
   */
  const shouldTriggerGeneration = useCallback((): boolean => {
    // No generar si está desactivado o no hay datos emocionales
    if (!enabled || !currentEmotion.valence || !currentEmotion.arousal) {
      return false;
    }

    // No generar si ya hay una generación en curso
    if (isGeneratingRef.current) {
      return false;
    }

    // Si es la primera generación, disparar inmediatamente
    if (state.lastGeneratedValence === null || state.lastGeneratedArousal === null) {
      return true;
    }

    // Verificar cooldown
    if (state.lastGeneratedTimestamp) {
      const timeSinceLast = Date.now() - state.lastGeneratedTimestamp;
      if (timeSinceLast < cooldownMs) {
        return false;
      }
    }

    // Calcular deltas
    const deltaValence = Math.abs(currentEmotion.valence - state.lastGeneratedValence);
    const deltaArousal = Math.abs(currentEmotion.arousal - state.lastGeneratedArousal);

    // Disparar si alguno de los deltas supera el threshold
    return deltaValence >= deltaValenceThreshold || deltaArousal >= deltaArousalThreshold;
  }, [
    enabled,
    currentEmotion,
    state.lastGeneratedValence,
    state.lastGeneratedArousal,
    state.lastGeneratedTimestamp,
    deltaValenceThreshold,
    deltaArousalThreshold,
    cooldownMs,
  ]);

  /**
   * Ejecuta la generación de MIDI
   */
  const triggerGeneration = useCallback(async () => {
    if (!currentEmotion.valence || !currentEmotion.arousal || !currentEmotion.emotion) {
      return;
    }

    // Prevenir múltiples generaciones simultáneas
    if (isGeneratingRef.current) {
      console.log('[AutoMidi] Ya hay una generación en curso, saltando...');
      return;
    }

    isGeneratingRef.current = true;
    setState(prev => ({ ...prev, isGenerating: true }));

    // Crear AbortController para esta generación
    abortControllerRef.current = new AbortController();

    try {
      const seed = Math.floor(Math.random() * 10000);

      console.log(
        `[AutoMidi] Generando MIDI automático: engine=${engine}, V=${currentEmotion.valence.toFixed(2)}, A=${currentEmotion.arousal.toFixed(2)}`
      );

      const result = await apiClient.generateMidi({
        engine,
        seed,
        length_bars: lengthBars,
        valence: currentEmotion.valence,
        arousal: currentEmotion.arousal,
        emotion: currentEmotion.emotion,
      });

      // Actualizar estado con la emoción usada para esta generación
      setState(prev => ({
        ...prev,
        isGenerating: false,
        lastGeneratedValence: currentEmotion.valence!,
        lastGeneratedArousal: currentEmotion.arousal!,
        lastGeneratedTimestamp: Date.now(),
        generationCount: prev.generationCount + 1,
        timeSinceLastGeneration: 0,
      }));

      // Notificar al componente padre
      onMidiGenerated(result);
    } catch (err) {
      console.error('[AutoMidi] Error al generar MIDI:', err);

      setState(prev => ({ ...prev, isGenerating: false }));

      // Solo reportar error si no fue abortado manualmente
      if (err instanceof Error && err.name !== 'AbortError') {
        onError(err.message);
      }
    } finally {
      isGeneratingRef.current = false;
      abortControllerRef.current = null;
    }
  }, [currentEmotion, engine, lengthBars, onMidiGenerated, onError]);

  /**
   * Effect principal: verifica continuamente si se deben disparar generaciones
   */
  useEffect(() => {
    if (!enabled) {
      return;
    }

    if (shouldTriggerGeneration()) {
      triggerGeneration();
    }
  }, [enabled, currentEmotion, shouldTriggerGeneration, triggerGeneration]);

  /**
   * Cleanup: abortar generación en curso si se desmonta o se desactiva
   */
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      isGeneratingRef.current = false;
    };
  }, []);

  return state;
}
