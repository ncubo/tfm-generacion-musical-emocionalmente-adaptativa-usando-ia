import { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import type { GenerateMidiResponse, MusicEngine, EngineInfo } from '../types';
import { COPIED_MESSAGE_DURATION } from '../utils/constants';
import { MidiPlayer } from './MidiPlayer';

// Nombres amigables para los engines (en español)
const ENGINE_LABELS: Record<MusicEngine, string> = {
  baseline: 'Baseline (reglas)',
  transformer_pretrained: 'Transformer (preentrenado)',
  transformer_finetuned: 'Transformer (fine-tuned)',
};

interface MidiCardProps {
  capturedEmotion?: { valence: number; arousal: number; emotion: string } | null;
}

export function MidiCard({ capturedEmotion }: MidiCardProps) {
  const [midiData, setMidiData] = useState<GenerateMidiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [playerError, setPlayerError] = useState<string | null>(null);
  
  // Estados para engines
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [selectedEngine, setSelectedEngine] = useState<MusicEngine>('baseline');
  const [loadingEngines, setLoadingEngines] = useState(true);

  // Cargar lista de engines al montar
  useEffect(() => {
    const loadEngines = async () => {
      try {
        const response = await apiClient.getEngines();
        setEngines(response.engines);
        
        // Seleccionar el primer engine disponible por defecto
        const availableEngine = response.engines.find(e => e.available);
        if (availableEngine) {
          setSelectedEngine(availableEngine.name);
        }
      } catch (err) {
        console.error('Error al cargar engines:', err);
        // Usar baseline como fallback
        setSelectedEngine('baseline');
      } finally {
        setLoadingEngines(false);
      }
    };

    loadEngines();
  }, []);

  const handleGenerateMidi = async () => {
    setLoading(true);
    setError(null);
    setPlayerError(null);
    setCopied(false);
    
    try {
      const result = await apiClient.generateMidi({
        engine: selectedEngine,
        // Usar emoción capturada si está disponible
        ...(capturedEmotion && {
          valence: capturedEmotion.valence,
          arousal: capturedEmotion.arousal,
        }),
      });
      setMidiData(result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Error al generar MIDI';
      setError(errorMessage);
      setMidiData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleCopyPath = () => {
    if (midiData?.midi_path) {
      navigator.clipboard.writeText(midiData.midi_path);
      setCopied(true);
      setTimeout(() => setCopied(false), COPIED_MESSAGE_DURATION);
    }
  };

  // Verificar si el engine seleccionado está disponible
  const isEngineAvailable = engines.find(e => e.name === selectedEngine)?.available ?? true;

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Generación MIDI</h2>

      {/* Selector de motor */}
      <div className="mb-4">
        <label htmlFor="engine-select" className="block text-sm font-semibold text-gray-700 mb-2">
          Motor de generación:
        </label>
        
        {loadingEngines ? (
          <div className="text-sm text-gray-500">Cargando motores...</div>
        ) : (
          <select
            id="engine-select"
            value={selectedEngine}
            onChange={(e) => setSelectedEngine(e.target.value as MusicEngine)}
            className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-green-500 focus:border-transparent"
          >
            {engines.map((engine) => (
              <option key={engine.name} value={engine.name} disabled={!engine.available}>
                {ENGINE_LABELS[engine.name]} {!engine.available && '(no disponible)'}
              </option>
            ))}
          </select>
        )}
        
        {/* Descripción del engine seleccionado */}
        {!loadingEngines && (
          <p className="mt-2 text-xs text-gray-500">
            {engines.find(e => e.name === selectedEngine)?.description}
          </p>
        )}

        {/* Advertencia si el engine no está disponible */}
        {!isEngineAvailable && (
          <div className="mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded">
            <p className="text-sm text-yellow-800 font-semibold">⚠️ Motor no disponible</p>
            <p className="text-xs text-yellow-700 mt-1">
              {selectedEngine === 'transformer_finetuned' 
                ? 'El motor fine-tuned aún no está implementado.'
                : 'Este motor requiere descargar el checkpoint. Ejecuta los scripts en backend/scripts/'}
            </p>
          </div>
        )}
      </div>

      {/* Indicador de emoción capturada */}
      {capturedEmotion && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded">
          <p className="text-sm text-blue-800 font-semibold">✓ Emoción capturada:</p>
          <p className="text-md font-bold text-blue-900 capitalize mt-1">{capturedEmotion.emotion}</p>
          <div className="mt-1 text-xs text-blue-700">
            Valencia: {capturedEmotion.valence.toFixed(2)} · Activación: {capturedEmotion.arousal.toFixed(2)}
          </div>
          <p className="text-xs text-blue-600 mt-1">
            Se usarán estos valores para generar el MIDI
          </p>
        </div>
      )}

      {/* Botón de generación */}
      <button
        onClick={handleGenerateMidi}
        disabled={loading || loadingEngines}
        aria-label="Generar archivo MIDI"
        className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold py-3 px-4 rounded transition-colors duration-200"
      >
        {loading ? 'Generando...' : `Generar MIDI con ${ENGINE_LABELS[selectedEngine]}`}
      </button>

      {/* Resultados */}
      {midiData && (
        <div className="mt-4 space-y-4">
          {/* Reproductor MIDI */}
          <MidiPlayer 
            midiBase64={midiData.midi_data} 
            onError={setPlayerError}
          />

          {/* Error del reproductor */}
          {playerError && (
            <div className="p-3 bg-red-50 border border-red-200 rounded">
              <p className="text-sm text-red-800">⚠️ {playerError}</p>
            </div>
          )}

        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded">
          <p className="text-red-800 font-semibold">❌ Error</p>
          <p className="text-sm text-red-600 mt-1">{error}</p>
          
          {/* Instrucciones específicas según el error */}
          {error.includes('checkpoint') && (
            <div className="mt-3 p-3 bg-red-100 rounded">
              <p className="text-xs font-semibold text-red-900">Instrucciones:</p>
              <p className="text-xs text-red-800 mt-1">
                Ejecuta en el backend:
              </p>
              <code className="block mt-1 text-xs bg-white p-2 rounded text-red-900">
                python scripts/download_transformer_pretrained.py
              </code>
              <code className="block mt-1 text-xs bg-white p-2 rounded text-red-900">
                python scripts/verify_transformer_pretrained.py
              </code>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
