import { useState } from 'react';
import { Music, Loader2, Download } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { EmotionFromFrameResponse } from '../../types';
import { MidiPlayer } from './MidiPlayer';
import { translateEmotion } from '../../utils/emotionTranslation';

interface MidiGeneratorProps {
  emotionData: EmotionFromFrameResponse | null;
  selectedEngine?: MusicEngine;
  onEngineChange?: (engine: MusicEngine) => void;
  lengthBars?: number;
  onLengthBarsChange?: (bars: number) => void;
  midiUrl?: string | null;
  generatedInfo?: {
    engine: string;
    seed: number;
    lengthBars: number;
  } | null;
  onMidiGenerated?: (result: {
    midiBlob: Blob;
    engine: string;
    seed: number;
    length_bars: number;
  }) => void;
  isAutoGenerating?: boolean;
  autoPlayMidi?: boolean; // Nueva prop para auto-reproducir
  autoMidiEnabled?: boolean; // Si el modo auto está activado
}

type MusicEngine = 'baseline' | 'transformer_pretrained' | 'transformer_finetuned';

const ENGINE_INFO = {
  baseline: {
    name: 'Baseline (Reglas)',
    description: 'Generador basado en reglas musicales deterministas',
  },
  transformer_pretrained: {
    name: 'HF Maestro-REMI',
    description: 'Modelo Transformer preentrenado (Hugging Face)',
  },
  transformer_finetuned: {
    name: 'Transformer (fine-tuned)',
    description: 'Modelo ajustado con Lakh piano-only + tokens VA',
  },
} as const;

export function MidiGenerator({
  emotionData,
  selectedEngine: externalEngine,
  // onEngineChange, // Not used currently
  lengthBars: externalLengthBars,
  onLengthBarsChange,
  midiUrl: externalMidiUrl,
  generatedInfo: externalGeneratedInfo,
  onMidiGenerated,
  isAutoGenerating = false,
  autoPlayMidi = false,
  autoMidiEnabled = false,
}: MidiGeneratorProps) {
  // Estados internos (usados solo si no hay props externas)
  const [internalLengthBars, setInternalLengthBars] = useState(4);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [internalMidiUrl, setInternalMidiUrl] = useState<string | null>(null);
  const [internalGeneratedInfo, setInternalGeneratedInfo] = useState<{
    engine: string;
    seed: number;
    lengthBars: number;
  } | null>(null);

  // Usar props externas si están disponibles, sino usar estado interno
  const selectedEngine: MusicEngine = externalEngine ?? 'transformer_finetuned';
  const lengthBars = externalLengthBars !== undefined ? externalLengthBars : internalLengthBars;
  const midiUrl = externalMidiUrl !== undefined ? externalMidiUrl : internalMidiUrl;
  const generatedInfo =
    externalGeneratedInfo !== undefined ? externalGeneratedInfo : internalGeneratedInfo;
  const isGenerating = isAutoGenerating || generating;

  const handleGenerate = async () => {
    if (!emotionData) {
      setError('No hay datos emocionales disponibles');
      return;
    }

    setGenerating(true);
    setError(null);

    // Limpiar MIDI anterior solo si usamos estado interno
    if (externalMidiUrl === undefined && internalMidiUrl) {
      URL.revokeObjectURL(internalMidiUrl);
      setInternalMidiUrl(null);
    }

    try {
      const seed = Math.floor(Math.random() * 10000);

      const result = await apiClient.generateMidi({
        engine: selectedEngine,
        seed,
        length_bars: lengthBars,
        valence: emotionData.valence,
        arousal: emotionData.arousal,
        emotion: emotionData.emotion,
      });

      // Si hay callback externo, usarlo
      if (onMidiGenerated) {
        onMidiGenerated(result);
      } else {
        // Si no, usar estado interno
        const url = URL.createObjectURL(result.midiBlob);
        setInternalMidiUrl(url);

        setInternalGeneratedInfo({
          engine: result.engine,
          seed: result.seed,
          lengthBars: result.length_bars,
        });
      }
    } catch (err) {
      console.error('Error al generar MIDI:', err);
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setGenerating(false);
    }
  };

  const handleLengthBarsChange = (bars: number) => {
    if (onLengthBarsChange) {
      onLengthBarsChange(bars);
    } else {
      setInternalLengthBars(bars);
    }
  };

  const handleDownload = () => {
    if (!midiUrl) return;

    const a = document.createElement('a');
    a.href = midiUrl;
    a.download = `emotion_${selectedEngine}_${Date.now()}.mid`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  if (!emotionData) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
        <p className="text-yellow-800">Captura y analiza una imagen primero para generar música</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Generador de Música MIDI</h2>
        <p className="text-gray-600">
          Emoción detectada:{' '}
          <span className="font-semibold">{translateEmotion(emotionData.emotion)}</span> (V:{' '}
          {emotionData.valence.toFixed(2)}, A: {emotionData.arousal.toFixed(2)})
        </p>
        <p className="text-sm text-gray-500 mt-1">
          Motor:{' '}
          <span className="font-semibold text-blue-600">
            {ENGINE_INFO[selectedEngine]?.name || selectedEngine}
          </span>
          {autoMidiEnabled && (
            <span className="ml-2 text-purple-600">• Modo Auto (4 compases)</span>
          )}
        </p>
      </div>

      {/* Duración - Solo mostrar si modo auto está desactivado */}
      {!autoMidiEnabled && (
        <div>
          <label htmlFor="length-bars" className="block text-sm font-medium text-gray-700 mb-2">
            Duración: {lengthBars} compases
          </label>
          <input
            id="length-bars"
            type="range"
            min="4"
            max="32"
            step="4"
            value={lengthBars}
            onChange={e => handleLengthBarsChange(parseInt(e.target.value))}
            disabled={isGenerating}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>4 compases</span>
            <span>32 compases</span>
          </div>
        </div>
      )}

      {/* Botón Generar - Solo mostrar si modo auto está desactivado */}
      {!autoMidiEnabled && (
        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className={`w-full py-3 px-4 rounded-lg font-semibold text-white transition-colors ${
            isGenerating
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
          }`}
        >
          {isGenerating ? (
            <span className="flex items-center justify-center">
              <Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5" />
              {isAutoGenerating ? 'Generando automático...' : 'Generando MIDI...'}
            </span>
          ) : (
            <span className="flex items-center justify-center">
              <Music className="-ml-1 mr-2 h-5 w-5" />
              Generar Música (Manual)
            </span>
          )}
        </button>
      )}

      {/* Mensaje cuando modo auto está activo */}
      {autoMidiEnabled && (
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <p className="text-sm text-purple-800 text-center">
            Modo automático activado. La música se generará automáticamente al detectar cambios
            emocionales.
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 text-sm">{error}</p>
        </div>
      )}

      {/* Reproductor MIDI */}
      {midiUrl && generatedInfo && (
        <div className="border-t pt-6">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-gray-900">MIDI Generado</h3>
              <p className="text-sm text-gray-600">
                Engine:{' '}
                {ENGINE_INFO[generatedInfo.engine as MusicEngine]?.name || generatedInfo.engine}
                {' • '}Seed: {generatedInfo.seed}
                {' • '}
                {generatedInfo.lengthBars} compases
              </p>
            </div>
            <button
              onClick={handleDownload}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors flex items-center space-x-2"
            >
              <Download className="w-4 h-4" />
              <span>Descargar</span>
            </button>
          </div>

          <MidiPlayer midiUrl={midiUrl} autoPlay={autoPlayMidi} />
        </div>
      )}
    </div>
  );
}
