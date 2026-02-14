import type { RefObject } from 'react';

interface CameraViewProps {
  status: 'idle' | 'requesting' | 'active' | 'error';
  error: string | null;
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
}

export function CameraView({ status, error, videoRef, canvasRef }: CameraViewProps) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Cámara</h2>

      {/* Estado: Cargando */}
      {status === 'requesting' && (
        <div className="flex items-center justify-center p-8 bg-blue-50 border border-blue-200 rounded">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-3"></div>
            <p className="text-blue-800 font-semibold">Iniciando cámara...</p>
          </div>
        </div>
      )}

      {/* Estado: Activa */}
      {status === 'active' && (
        <div>
          <div className="relative bg-gray-900 rounded-lg overflow-hidden mb-4">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full min-h-60 object-cover"
              style={{ objectFit: 'cover' }}
            />
          </div>

          {/* Canvas oculto para captura */}
          <canvas ref={canvasRef} className="hidden" />
        </div>
      )}

      {/* Estado: Error */}
      {status === 'error' && (
        <div className="p-4 bg-red-50 border border-red-200 rounded">
          <p className="text-red-800 font-semibold">✗ No se pudo acceder a la cámara</p>
          <p className="text-sm text-red-600 mt-1">{error}</p>
        </div>
      )}
    </div>
  );
}
