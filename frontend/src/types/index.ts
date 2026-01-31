/**
 * Tipos compartidos para la aplicación
 */

// Re-exportar tipos de API para uso en la aplicación
export type {
  HealthResponse,
  EmotionResponse,
  GenerateMidiRequest,
  GenerateMidiResponse,
} from '../api/client';

// Estados de UI comunes
export interface LoadingState {
  loading: boolean;
  error: string | null;
}

// Tipo para valores de emoción normalizados
export type EmotionValue = number; // Rango: -1 a 1
