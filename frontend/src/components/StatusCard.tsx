import { useState } from 'react';
import { apiClient } from '../api/client';
import type { HealthResponse } from '../api/client';

export function StatusCard() {
  const [status, setStatus] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCheckHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.checkHealth();
      setStatus(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al conectar con el servidor');
      setStatus(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Estado del Servidor</h2>

      <button
        onClick={handleCheckHealth}
        disabled={loading}
        aria-label="Comprobar estado del servidor"
        className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded transition-colors duration-200"
      >
        {loading ? 'Comprobando...' : 'Comprobar Estado'}
      </button>

      {status && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded">
          <p className="text-green-800 font-semibold">✓ Servidor OK</p>
          <p className="text-sm text-green-600 mt-1">Estado: {status.status}</p>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded">
          <p className="text-red-800 font-semibold">✗ Error al conectar</p>
          <p className="text-sm text-red-600 mt-1">{error}</p>
        </div>
      )}
    </div>
  );
}
