import { useState } from 'react';
import { WebcamView } from '../components/WebcamView';
import { MidiCard } from '../components/MidiCard';
import type { EmotionFromFrameResponse } from '../types';

export function LiveDemo() {
  const [capturedEmotion, setCapturedEmotion] = useState<EmotionFromFrameResponse | null>(null);

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-100 to-gray-200">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            Generación Musical Emocionalmente Adaptativa
          </h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Webcam para detección emocional */}
          <WebcamView onEmotionDetected={setCapturedEmotion} />
          
          {/* Generación MIDI con selector de motor */}
          <MidiCard capturedEmotion={capturedEmotion} />
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
