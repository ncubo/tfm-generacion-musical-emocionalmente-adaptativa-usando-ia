import { StatusCard } from '../components/StatusCard';
import { EmotionCard } from '../components/EmotionCard';
import { MidiCard } from '../components/MidiCard';

export function LiveDemo() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            Generación Musical Emocionalmente Adaptativa
          </h1>
          <p className="mt-2 text-sm text-gray-600">Demo en vivo - Interacción con backend Flask</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Columna izquierda */}
          <div className="space-y-6">
            <StatusCard />
            <EmotionCard />
          </div>

          {/* Columna derecha */}
          <div className="space-y-6">
            <MidiCard />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 bg-white border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <p className="text-center text-sm text-gray-500">
            TFM - Generación Musical con IA © 2026
          </p>
        </div>
      </footer>
    </div>
  );
}
