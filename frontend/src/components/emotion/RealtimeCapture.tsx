interface RealtimeState {
  isRunning: boolean;
  isAnalyzing: boolean;
  emotion: string | null;
  valence: number | null;
  arousal: number | null;
  faceDetected: boolean;
  error: string | null;
}

interface RealtimeCaptureProps {
  realtimeState: RealtimeState;
  realtimeEnabled: boolean;
  realtimeInterval: number;
  onToggleRealtime: (enabled: boolean) => void;
  onIntervalChange: (interval: number) => void;
  fpsOptions: readonly { value: number; label: string }[];
}

export function RealtimeCapture({
  realtimeState,
  realtimeEnabled,
  realtimeInterval,
  onToggleRealtime,
  onIntervalChange,
  fpsOptions,
}: RealtimeCaptureProps) {
  return (
    <div>
      {/* Controles de modo tiempo real */}
      <div className="mb-4 p-4 bg-teal-50 border border-teal-200 rounded-lg">
        <h3 className="text-lg font-semibold text-teal-900 mb-3">Modo Tiempo Real</h3>

        <div className="space-y-3">
          {/* Estado actual */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">Estado:</span>
            <span
              className={`text-sm font-bold ${realtimeState.isRunning ? 'text-teal-700' : 'text-gray-500'}`}
            >
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
              onChange={(e) => onIntervalChange(Number(e.target.value))}
              disabled={realtimeEnabled}
              className="px-3 py-1 border border-gray-300 rounded bg-white text-sm disabled:bg-gray-100 disabled:cursor-not-allowed"
            >
              {fpsOptions.map((option) => (
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
                onClick={() => onToggleRealtime(true)}
                className="flex-1 bg-teal-600 hover:bg-teal-700 text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
                aria-label="Iniciar análisis en tiempo real"
              >
                Iniciar tiempo real
              </button>
            ) : (
              <button
                onClick={() => onToggleRealtime(false)}
                className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
                aria-label="Detener análisis en tiempo real"
              >
                ⏹️ Detener tiempo real
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Resultados en tiempo real */}
      {realtimeState.isRunning && realtimeState.emotion && (
        <div className="mb-4 p-4 bg-teal-50 border border-teal-200 rounded">
          <h3 className="text-lg font-semibold text-teal-900 mb-3">
            Emoción actual (tiempo real):
          </h3>

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
              <p className="font-semibold">No se detecta rostro</p>
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
    </div>
  );
}
