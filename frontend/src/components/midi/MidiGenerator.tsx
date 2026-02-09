import { useState } from 'react';
import { Music, Bot, Loader2, Download, CheckCircle2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { EmotionFromFrameResponse } from '../../types';
import { MidiPlayer } from './MidiPlayer';

interface MidiGeneratorProps {
  emotionData: EmotionFromFrameResponse | null;
}

type MusicEngine = 'baseline' | 'transformer_pretrained';

const ENGINE_INFO = {
  baseline: {
    name: 'Baseline (Reglas)',
    description: 'Generador basado en reglas musicales deterministas',
    icon: Music,
  },
  transformer_pretrained: {
    name: 'HF Maestro-REMI',
    description: 'Modelo Transformer preentrenado (Hugging Face)',
    icon: Bot,
  },
} as const;

export function MidiGenerator({ emotionData }: MidiGeneratorProps) {
  const [selectedEngine, setSelectedEngine] = useState<MusicEngine>('baseline');
  const [lengthBars, setLengthBars] = useState(4);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [midiUrl, setMidiUrl] = useState<string | null>(null);
  const [generatedInfo, setGeneratedInfo] = useState<{
    engine: string;
    seed: number;
    lengthBars: number;
  } | null>(null);

  const handleGenerate = async () => {
    if (!emotionData) {
      setError('No hay datos emocionales disponibles');
      return;
    }

    setGenerating(true);
    setError(null);
    
    // Limpiar MIDI anterior
    if (midiUrl) {
      URL.revokeObjectURL(midiUrl);
      setMidiUrl(null);
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

      // Crear URL para el blob MIDI
      const url = URL.createObjectURL(result.midiBlob);
      setMidiUrl(url);
      
      setGeneratedInfo({
        engine: result.engine,
        seed: result.seed,
        lengthBars: result.length_bars,
      });

    } catch (err) {
      console.error('Error al generar MIDI:', err);
      setError(err instanceof Error ? err.message : 'Error desconocido');
    } finally {
      setGenerating(false);
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
        <p className="text-yellow-800">
          📸 Captura y analiza una imagen primero para generar música
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Generador de Música MIDI
        </h2>
        <p className="text-gray-600">
          Emoción detectada: <span className="font-semibold">{emotionData.emotion}</span> 
          {' '}(V: {emotionData.valence.toFixed(2)}, A: {emotionData.arousal.toFixed(2)})
        </p>
      </div>

      {/* Selección de Engine */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Motor de Generación
        </label>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {(Object.keys(ENGINE_INFO) as MusicEngine[]).map((engine) => {
            const info = ENGINE_INFO[engine];
            const IconComponent = info.icon;
            const isSelected = selectedEngine === engine;
            
            return (
              <button
                key={engine}
                onClick={() => setSelectedEngine(engine)}
                className={`p-4 rounded-lg border-2 transition-all text-left ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50 shadow-md'
                    : 'border-gray-200 bg-white hover:border-gray-300'
                }`}
              >
                <div className="flex items-start space-x-3">
                  <IconComponent className="w-8 h-8 text-blue-600" />
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900">{info.name}</h3>
                    <p className="text-sm text-gray-600 mt-1">{info.description}</p>
                  </div>
                  {isSelected && (
                    <CheckCircle2 className="w-6 h-6 text-blue-500" />
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Duración */}
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
          onChange={(e) => setLengthBars(parseInt(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>4 compases</span>
          <span>32 compases</span>
        </div>
      </div>

      {/* Botón Generar */}
      <button
        onClick={handleGenerate}
        disabled={generating}
        className={`w-full py-3 px-4 rounded-lg font-semibold text-white transition-colors ${
          generating
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
        }`}
      >
        {generating ? (
          <span className="flex items-center justify-center">
            <Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5" />
            Generando MIDI...
          </span>
        ) : (
          <span className="flex items-center justify-center">
            <Music className="-ml-1 mr-2 h-5 w-5" />
            Generar Música
          </span>
        )}
      </button>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 text-sm">❌ {error}</p>
        </div>
      )}

      {/* Reproductor MIDI */}
      {midiUrl && generatedInfo && (
        <div className="border-t pt-6">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-gray-900">MIDI Generado</h3>
              <p className="text-sm text-gray-600">
                Engine: {ENGINE_INFO[generatedInfo.engine as MusicEngine]?.name || generatedInfo.engine}
                {' • '}Seed: {generatedInfo.seed}
                {' • '}{generatedInfo.lengthBars} compases
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
          
          <MidiPlayer midiUrl={midiUrl} />
        </div>
      )}
    </div>
  );
}
