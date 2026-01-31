import { useState } from 'react';
import { apiClient } from '../api/client';
import type { GenerateMidiResponse } from '../types';
import { COPIED_MESSAGE_DURATION } from '../utils/constants';

export function MidiCard() {
  const [midiData, setMidiData] = useState<GenerateMidiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleGenerateMidi = async () => {
    setLoading(true);
    setError(null);
    setCopied(false);
    try {
      const result = await apiClient.generateMidi();
      setMidiData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al generar MIDI');
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

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Generación MIDI</h2>

      <button
        onClick={handleGenerateMidi}
        disabled={loading}
        aria-label="Generar archivo MIDI"
        className="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
      >
        {loading ? 'Generando...' : 'Generar MIDI'}
      </button>

      {midiData && (
        <div className="mt-4 space-y-4">
          {/* Información de emoción */}
          <div className="p-4 bg-green-50 border border-green-200 rounded">
            <p className="text-sm text-gray-600">Emoción generada:</p>
            <p className="text-lg font-bold text-green-800 capitalize">{midiData.emotion}</p>
            <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
              <p className="text-gray-600">
                Valencia: <span className="font-semibold">{midiData.valence.toFixed(2)}</span>
              </p>
              <p className="text-gray-600">
                Activación: <span className="font-semibold">{midiData.arousal.toFixed(2)}</span>
              </p>
            </div>
          </div>

          {/* Ruta del archivo MIDI */}
          <div className="p-4 bg-gray-50 border border-gray-200 rounded">
            <p className="text-sm text-gray-600 mb-2">Ruta MIDI:</p>
            <div className="flex items-center gap-2">
              <code className="flex-1 text-sm bg-white p-2 rounded border border-gray-300 font-mono text-gray-800 break-all">
                {midiData.midi_path}
              </code>
              <button
                onClick={handleCopyPath}
                aria-label="Copiar ruta del archivo MIDI"
                className="bg-gray-600 hover:bg-gray-700 text-white px-3 py-2 rounded text-sm font-semibold transition-colors duration-200 whitespace-nowrap"
              >
                {copied ? '✓ Copiado' : 'Copiar'}
              </button>
            </div>
          </div>

          {/* Parámetros musicales */}
          <div className="p-4 bg-blue-50 border border-blue-200 rounded">
            <p className="text-sm font-semibold text-gray-700 mb-2">Parámetros musicales:</p>
            <div className="bg-white p-3 rounded border border-blue-200">
              <pre className="text-xs text-gray-800 overflow-x-auto">
                {JSON.stringify(midiData.params, null, 2)}
              </pre>
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
