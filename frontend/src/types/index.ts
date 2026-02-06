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

// Tipos para engines de generación MIDI
export type MusicEngine = 'baseline' | 'transformer_pretrained' | 'transformer_finetuned';

export interface EngineInfo {
  name: MusicEngine;
  description: string;
  available: boolean;
}

export interface EnginesListResponse {
  engines: EngineInfo[];
}

export interface GenerateMidiRequest {
  engine?: MusicEngine;
  emotion?: string;
  valence?: number;
  arousal?: number;
  seed?: number;
  params?: Record<string, unknown>;
}

export interface GenerateMidiResponse {
  engine: MusicEngine;
  emotion?: string;
  valence: number;
  arousal: number;
  generation_params: Record<string, unknown>;
  midi_path: string;
  midi_data: string; // MIDI file content in base64
}

// Estados de UI comunes
export interface LoadingState {
  loading: boolean;
  error: string | null;
}

// Tipo para valores de emoción normalizados
export type EmotionValue = number; // Rango: -1 a 1
