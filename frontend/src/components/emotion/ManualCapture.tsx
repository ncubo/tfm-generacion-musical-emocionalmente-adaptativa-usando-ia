import type { EmotionFromFrameResponse } from '../../types';
import { translateEmotion } from '../../utils/emotionTranslation';

interface ManualCaptureProps {
  snapshotBlob: Blob | null;
  previewUrl: string | null;
  analyzing: boolean;
  emotionResult: EmotionFromFrameResponse | null;
  analysisError: string | null;
  onCapture: () => void;
  onAnalyze: () => void;
  onClear: () => void;
}

export function ManualCapture({
  snapshotBlob,
  previewUrl,
  analyzing,
  emotionResult,
  analysisError,
  onCapture,
  onAnalyze,
  onClear,
}: ManualCaptureProps) {
  return (
    <div>
      {/* Botones de captura */}
      <div className="flex gap-3 mb-4">
        <button
          onClick={onCapture}
          className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
          aria-label="Capturar imagen de la cámara"
        >
          Capturar imagen
        </button>

        {snapshotBlob && (
          <>
            <button
              onClick={onAnalyze}
              disabled={analyzing}
              className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
              aria-label="Analizar emoción en la imagen"
            >
              {analyzing ? 'Analizando...' : 'Analizar imagen'}
            </button>

            <button
              onClick={onClear}
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

      {/* Preview de la imagen capturada */}
      {previewUrl && (
        <div className="mb-4">
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

      {/* Resultados del análisis emocional */}
      {emotionResult && (
        <div className="mb-4 p-4 bg-purple-50 border border-purple-200 rounded">
          <h3 className="text-lg font-semibold text-purple-900 mb-3">Análisis emocional:</h3>

          {emotionResult.face_detected ? (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Emoción:</span>
                <span className="text-lg font-bold text-purple-800 capitalize">
                  {translateEmotion(emotionResult.emotion)}
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
              <p className="text-sm mt-1">
                La imagen no contiene un rostro visible. Resultado neutral.
              </p>
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
    </div>
  );
}
