import { Video, Music, Play, Square, Loader2, Settings, Info, Circle } from 'lucide-react';
import { translateEmotion } from '../../utils/emotionTranslation';

interface RealtimeState {
  isRunning: boolean;
  isAnalyzing: boolean;
  emotion: string | null;
  valence: number | null;
  arousal: number | null;
  faceDetected: boolean;
  error: string | null;
}

interface AutoMidiState {
  enabled: boolean;
  isGenerating: boolean;
  generationCount: number;
  timeSinceLastGeneration: number | null;
  deltaValenceThreshold: number;
  deltaArousalThreshold: number;
  cooldownMs: number;
}

interface RealtimeCaptureProps {
  realtimeState: RealtimeState;
  realtimeEnabled: boolean;
  realtimeInterval: number;
  onToggleRealtime: (enabled: boolean) => void;
  onIntervalChange: (interval: number) => void;
  fpsOptions: readonly { value: number; label: string }[];

  // Props para modo Auto MIDI
  autoMidiState: AutoMidiState;
  onToggleAutoMidi: (enabled: boolean) => void;
  onThresholdChange: (type: 'valence' | 'arousal', value: number) => void;
  onCooldownChange: (cooldownMs: number) => void;
  selectedEngine: string;
}

export function RealtimeCapture({
  realtimeState,
  realtimeEnabled,
  realtimeInterval,
  onToggleRealtime,
  onIntervalChange,
  fpsOptions,
  autoMidiState,
  onToggleAutoMidi,
  onThresholdChange,
  onCooldownChange,
  selectedEngine,
}: RealtimeCaptureProps) {
  return (
    <div className="space-y-4">
      {/* Controles de modo tiempo real - Detección emocional */}
      <div className="p-4 bg-teal-50 border border-teal-200 rounded-lg">
        <div className="flex items-center gap-2 mb-3">
          <Video className="w-5 h-5 text-teal-700" />
          <h3 className="text-lg font-semibold text-teal-900">Detección Emocional (Tiempo Real)</h3>
        </div>

        <div className="space-y-3">
          {/* Estado actual */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">Estado:</span>
            <span className="flex items-center gap-1.5 text-sm font-bold">
              <Circle
                className={`w-3 h-3 ${realtimeState.isRunning ? 'fill-green-500 text-green-500' : 'fill-gray-400 text-gray-400'}`}
              />
              <span className={realtimeState.isRunning ? 'text-teal-700' : 'text-gray-500'}>
                {realtimeState.isRunning ? 'Activado' : 'Desactivado'}
              </span>
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
              onChange={e => onIntervalChange(Number(e.target.value))}
              disabled={realtimeEnabled}
              className="px-3 py-1 border border-gray-300 rounded bg-white text-sm disabled:bg-gray-100 disabled:cursor-not-allowed"
            >
              {fpsOptions.map(option => (
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
                className="flex-1 bg-teal-600 hover:bg-teal-700 text-white font-semibold py-2 px-4 rounded transition-colors duration-200 flex items-center justify-center gap-2"
                aria-label="Iniciar análisis en tiempo real"
              >
                <Play className="w-4 h-4" />
                Iniciar detección
              </button>
            ) : (
              <button
                onClick={() => onToggleRealtime(false)}
                className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded transition-colors duration-200 flex items-center justify-center gap-2"
                aria-label="Detener análisis en tiempo real"
              >
                <Square className="w-4 h-4" />
                Detener detección
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Controles de modo Auto MIDI */}
      <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
        <div className="flex items-center gap-2 mb-3">
          <Music className="w-5 h-5 text-purple-700" />
          <h3 className="text-lg font-semibold text-purple-900">Generación Automática de MIDI</h3>
        </div>

        <div className="space-y-3">
          {/* Estado del modo Auto */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">Modo Auto:</span>
            <span className="flex items-center gap-1.5 text-sm font-bold">
              <Circle
                className={`w-3 h-3 ${autoMidiState.enabled ? 'fill-green-500 text-green-500' : 'fill-gray-400 text-gray-400'}`}
              />
              <span className={autoMidiState.enabled ? 'text-purple-700' : 'text-gray-500'}>
                {autoMidiState.enabled ? 'Activado' : 'Desactivado'}
              </span>
            </span>
          </div>

          {/* Motor seleccionado */}
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">Motor actual:</span>
            <span className="text-sm font-semibold text-purple-800">
              {selectedEngine === 'baseline' && 'Baseline'}
              {selectedEngine === 'transformer_pretrained' && 'HF Maestro'}
              {selectedEngine === 'transformer_finetuned' && 'Fine-tuned'}
            </span>
          </div>

          {/* Estadísticas de generación */}
          {autoMidiState.generationCount > 0 && (
            <div className="bg-purple-100 rounded p-2 space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-700">Generaciones:</span>
                <span className="font-bold text-purple-900">{autoMidiState.generationCount}</span>
              </div>
              {autoMidiState.timeSinceLastGeneration !== null && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-700">Última generación:</span>
                  <span className="font-semibold text-purple-800">
                    hace {autoMidiState.timeSinceLastGeneration}s
                  </span>
                </div>
              )}
              {autoMidiState.isGenerating && (
                <div className="flex items-center justify-center gap-1.5 text-xs text-purple-700">
                  <Loader2 className="w-3 h-3 animate-spin" />
                  <span className="font-semibold">Generando...</span>
                </div>
              )}
            </div>
          )}

          {/* Configuración de thresholds (panel avanzado) */}
          {autoMidiState.enabled && (
            <details className="bg-purple-100 rounded p-2">
              <summary className="text-xs font-semibold text-purple-900 cursor-pointer hover:text-purple-700 flex items-center gap-1.5">
                <Settings className="w-3.5 h-3.5" />
                Configuración avanzada
              </summary>
              <div className="mt-2 space-y-2">
                {/* Threshold Valence */}
                <div>
                  <label
                    htmlFor="threshold-valence"
                    className="text-xs font-medium text-gray-700 block mb-1"
                  >
                    Umbral Valencia: {autoMidiState.deltaValenceThreshold.toFixed(2)}
                  </label>
                  <input
                    id="threshold-valence"
                    type="range"
                    min="0.05"
                    max="0.5"
                    step="0.05"
                    value={autoMidiState.deltaValenceThreshold}
                    onChange={e => onThresholdChange('valence', parseFloat(e.target.value))}
                    className="w-full h-1 bg-purple-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
                  />
                </div>

                {/* Threshold Arousal */}
                <div>
                  <label
                    htmlFor="threshold-arousal"
                    className="text-xs font-medium text-gray-700 block mb-1"
                  >
                    Umbral Activación: {autoMidiState.deltaArousalThreshold.toFixed(2)}
                  </label>
                  <input
                    id="threshold-arousal"
                    type="range"
                    min="0.05"
                    max="0.5"
                    step="0.05"
                    value={autoMidiState.deltaArousalThreshold}
                    onChange={e => onThresholdChange('arousal', parseFloat(e.target.value))}
                    className="w-full h-1 bg-purple-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
                  />
                </div>

                {/* Cooldown */}
                <div>
                  <label
                    htmlFor="cooldown-ms"
                    className="text-xs font-medium text-gray-700 block mb-1"
                  >
                    Cooldown: {(autoMidiState.cooldownMs / 1000).toFixed(0)}s
                  </label>
                  <input
                    id="cooldown-ms"
                    type="range"
                    min="2000"
                    max="10000"
                    step="1000"
                    value={autoMidiState.cooldownMs}
                    onChange={e => onCooldownChange(parseInt(e.target.value))}
                    className="w-full h-1 bg-purple-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
                  />
                </div>
              </div>
            </details>
          )}

          {/* Botón toggle modo Auto */}
          <div className="flex gap-2">
            {!autoMidiState.enabled ? (
              <button
                onClick={() => onToggleAutoMidi(true)}
                disabled={!realtimeEnabled || !realtimeState.isRunning}
                className="flex-1 bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-4 rounded transition-colors duration-200 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                aria-label="Activar generación automática de MIDI"
                title={
                  !realtimeEnabled
                    ? 'Primero debes activar la detección emocional'
                    : 'Activa la generación automática'
                }
              >
                <Music className="w-4 h-4" />
                Activar Modo Auto
              </button>
            ) : (
              <button
                onClick={() => onToggleAutoMidi(false)}
                className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded transition-colors duration-200 flex items-center justify-center gap-2"
                aria-label="Desactivar generación automática de MIDI"
              >
                <Square className="w-4 h-4" />
                Desactivar Auto
              </button>
            )}
          </div>

          {/* Ayuda/Info */}
          {!realtimeEnabled && (
            <div className="flex items-start gap-2 text-xs text-purple-700 bg-purple-100 p-2 rounded">
              <Info className="w-4 h-4 shrink-0 mt-0.5" />
              <p>
                Primero activa la <strong>detección emocional</strong> arriba antes de activar el
                Modo Auto.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Resultados en tiempo real */}
      {realtimeState.isRunning && realtimeState.emotion && (
        <div className="p-4 bg-teal-50 border border-teal-200 rounded">
          <h3 className="text-lg font-semibold text-teal-900 mb-3">
            Emoción actual (tiempo real):
          </h3>

          {realtimeState.faceDetected ? (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Emoción:</span>
                <span className="text-xl font-bold text-teal-800">
                  {translateEmotion(realtimeState.emotion)}
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
        <div className="p-3 bg-red-50 border border-red-200 rounded">
          <p className="text-red-800 font-semibold text-sm">Error en tiempo real:</p>
          <p className="text-sm text-red-600 mt-1">{realtimeState.error}</p>
        </div>
      )}
    </div>
  );
}
