/**
 * Mapeo de emociones de inglés a español
 * Basado en las emociones básicas de Ekman + neutral
 */
export const EMOTION_TRANSLATION: Record<string, string> = {
  happy: 'Feliz',
  sad: 'Triste',
  angry: 'Enfadado',
  fear: 'Miedo',
  surprise: 'Sorpresa',
  disgust: 'Disgusto',
  neutral: 'Neutral',
};

/**
 * Traduce una emoción del inglés al español
 * @param emotion - Emoción en inglés (ej: "happy")
 * @returns Emoción traducida al español (ej: "Feliz")
 */
export function translateEmotion(emotion: string): string {
  return EMOTION_TRANSLATION[emotion.toLowerCase()] || emotion;
}
