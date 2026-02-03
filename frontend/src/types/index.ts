/**
 * Tipos compartidos para la aplicación
 */

// Tipos para las respuestas del backend
export interface HealthResponse {
  status: string;
}

export interface EmotionResponse {
  emotion: string;
  valence: number;
  arousal: number;
}

export interface EmotionFromFrameResponse {
  emotion: string;
  valence: number;
  arousal: number;
  face_detected: boolean;
}

export interface GenerateMidiRequest {
  emotion?: string;
  valence?: number;
  arousal?: number;
  params?: Record<string, unknown>;
}

export interface GenerateMidiResponse {
  emotion: string;
  valence: number;
  arousal: number;
  params: Record<string, unknown>;
  midi_path: string;
}

// Estados de UI comunes
export interface LoadingState {
  loading: boolean;
  error: string | null;
}

// Tipo para valores de emoción normalizados
export type EmotionValue = number; // Rango: -1 a 1
