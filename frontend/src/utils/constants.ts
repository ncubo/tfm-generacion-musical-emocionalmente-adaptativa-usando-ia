/**
 * Constantes de configuración de la aplicación
 */

// Configuración de UI
export const COPIED_MESSAGE_DURATION = 2000; // ms

// Rangos de valores de emoción (Valencia y Activación van de -1 a 1)
export const EMOTION_RANGE = {
  MIN: -1,
  MAX: 1,
} as const;

/**
 * Normaliza un valor de emoción (-1 a 1) a porcentaje (0-100%)
 */
export const normalizeEmotionToPercentage = (value: number): number => {
  return ((value - EMOTION_RANGE.MIN) / (EMOTION_RANGE.MAX - EMOTION_RANGE.MIN)) * 100;
};

// Colores de los componentes (para mantener consistencia)
export const COLORS = {
  status: {
    primary: 'blue',
    success: 'green',
    error: 'red',
  },
  emotion: {
    primary: 'purple',
    valence: 'blue',
    arousal: 'orange',
  },
  midi: {
    primary: 'green',
  },
} as const;
