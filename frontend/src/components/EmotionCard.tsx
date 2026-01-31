import { useState } from 'react';
import { apiClient } from '../api/client';
import type { EmotionResponse } from '../api/client';
import { normalizeEmotionToPercentage } from '../utils/constants';

export function EmotionCard() {
  const [emotion, setEmotion] = useState<EmotionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGetEmotion = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.getEmotion();
      setEmotion(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al obtener la emoción');
      setEmotion(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Emoción Actual</h2>

      <button
        onClick={handleGetEmotion}
        disabled={loading}
        aria-label="Obtener emoción detectada"
        className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
      >
        {loading ? 'Obteniendo...' : 'Obtener Emoción'}
      </button>

      {emotion && (
        <div className="mt-4 space-y-3">
          <div className="p-4 bg-purple-50 border border-purple-200 rounded">
            <p className="text-sm text-gray-600">Emoción:</p>
            <p className="text-xl font-bold text-purple-800 capitalize">{emotion.emotion}</p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-blue-50 border border-blue-200 rounded">
              <p className="text-xs text-gray-600 mb-1">Valencia</p>
              <p className="text-lg font-semibold text-blue-800">{emotion.valence.toFixed(2)}</p>
              {/* Barra de progreso para valencia */}
              <div className="mt-2 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{
                    width: `${normalizeEmotionToPercentage(emotion.valence)}%`,
                  }}
                />
              </div>
            </div>

            <div className="p-3 bg-orange-50 border border-orange-200 rounded">
              <p className="text-xs text-gray-600 mb-1">Activación</p>
              <p className="text-lg font-semibold text-orange-800">{emotion.arousal.toFixed(2)}</p>
              {/* Barra de progreso para arousal */}
              <div className="mt-2 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-orange-600 h-2 rounded-full transition-all duration-300"
                  style={{
                    width: `${normalizeEmotionToPercentage(emotion.arousal)}%`,
                  }}
                />
              </div>
            </div>
          </div>

          {/* Visualización simple del espacio V-A */}
          <div className="p-4 bg-gray-50 border border-gray-200 rounded">
            <p className="text-xs text-gray-600 mb-2">Espacio Valencia-Activación:</p>
            <div className="relative w-full h-32 bg-gradient-to-br from-blue-100 via-gray-100 to-orange-100 rounded border border-gray-300">
              {/* Punto indicador */}
              <div
                className="absolute w-4 h-4 bg-purple-600 rounded-full border-2 border-white shadow-lg transform -translate-x-2 -translate-y-2"
                style={{
                  left: `${normalizeEmotionToPercentage(emotion.valence)}%`,
                  top: `${100 - normalizeEmotionToPercentage(emotion.arousal)}%`,
                }}
              />
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded">
          <p className="text-red-800 font-semibold">Error</p>
          <p className="text-sm text-red-600 mt-1">{error}</p>
        </div>
      )}
    </div>
  );
}
