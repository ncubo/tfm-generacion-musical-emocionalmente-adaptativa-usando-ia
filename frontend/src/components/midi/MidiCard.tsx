import { useState } from 'react';
import { apiClient } from '../../api/client';

interface MidiGenerationResult {
  midiBlob: Blob;
  engine: string;
  seed: number;
  length_bars: number;
}

export function MidiCard() {
  const [midiData, setMidiData] = useState<MidiGenerationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateMidi = async () => {
    setLoading(true);
    setError(null);
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

  const handleDownload = () => {
    if (midiData?.midiBlob) {
      const url = URL.createObjectURL(midiData.midiBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `music_${midiData.engine}_${midiData.seed}.mid`;
      a.click();
      URL.revokeObjectURL(url);
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
          {/* Información del archivo generado */}
          <div className="p-4 bg-green-50 border border-green-200 rounded">
            <p className="text-sm text-gray-600">Archivo generado:</p>
            <p className="text-lg font-bold text-green-800">
              {midiData.engine === 'baseline' ? '🎵 Baseline Rules' : '🤖 Transformer'}
            </p>
            <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
              <p className="text-gray-600">
                Seed: <span className="font-semibold">{midiData.seed}</span>
              </p>
              <p className="text-gray-600">
                Compases: <span className="font-semibold">{midiData.length_bars}</span>
              </p>
            </div>
          </div>

          {/* Botón de descarga */}
          <div className="p-4 bg-gray-50 border border-gray-200 rounded">
            <button
              onClick={handleDownload}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
            >
              📥 Descargar MIDI
            </button>
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
