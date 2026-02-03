import { useEffect, useRef, useState } from 'react';

type WebcamStatus = 'idle' | 'requesting' | 'active' | 'error';

export function WebcamView() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  const [status, setStatus] = useState<WebcamStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Solicitar acceso a la cámara al montar el componente
    const startWebcam = async () => {
      setStatus('requesting');
      setError(null);

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 640 },
            height: { ideal: 480 }
          } 
        });

        // Guardar referencia del stream para limpieza posterior
        streamRef.current = stream;

        // Conectar stream al elemento video
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        setStatus('active');
      } catch (err) {
        console.error('Error al acceder a la cámara:', err);
        
        // Mensajes de error específicos según el tipo
        if (err instanceof Error) {
          if (err.name === 'NotAllowedError') {
            setError('Permiso denegado. Por favor, permite el acceso a la cámara.');
          } else if (err.name === 'NotFoundError') {
            setError('No se encontró ninguna cámara en este dispositivo.');
          } else {
            setError(`Error al acceder a la cámara: ${err.message}`);
          }
        } else {
          setError('Error desconocido al acceder a la cámara.');
        }
        
        setStatus('error');
      }
    };

    startWebcam();

    // Cleanup: liberar recursos al desmontar el componente
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    };
  }, []); // Solo ejecutar una vez al montar

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Cámara Web</h2>

      {/* Estado: Solicitando acceso */}
      {status === 'requesting' && (
        <div className="flex items-center justify-center p-8 bg-blue-50 border border-blue-200 rounded">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-3"></div>
            <p className="text-blue-800 font-semibold">Solicitando acceso a la cámara…</p>
            <p className="text-sm text-blue-600 mt-1">Por favor, acepta los permisos en tu navegador</p>
          </div>
        </div>
      )}

      {/* Estado: Cámara activa */}
      {status === 'active' && (
        <div>
          <div className="mb-3 p-3 bg-green-50 border border-green-200 rounded">
            <p className="text-green-800 font-semibold">✓ Cámara activa</p>
            <p className="text-sm text-green-600">La cámara está transmitiendo correctamente</p>
          </div>
          
          <div className="relative bg-gray-900 rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-auto"
              aria-label="Vista de la cámara web en tiempo real"
            />
          </div>
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
