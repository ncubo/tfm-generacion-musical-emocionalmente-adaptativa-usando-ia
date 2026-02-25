import { ENV } from '../config/env';
import type { HealthResponse, EmotionResponse, EmotionFromFrameResponse } from '../types';

// Configuración base para el cliente API
const API_BASE_URL = ENV.API_BASE_URL;

// Cliente API usando fetch
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  /**
   * Método genérico para realizar peticiones HTTP
   */
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    // Configurar timeout de 30 segundos
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        signal: controller.signal,
        ...options,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Error HTTP: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error) {
        // Mejorar mensajes de error específicos
        if (error.name === 'AbortError') {
          throw new Error('La petición tardó demasiado (timeout)');
        }
        if (error.message.includes('Failed to fetch')) {
          throw new Error('No se pudo conectar con el servidor. Verifica que esté ejecutándose.');
        }
        throw error;
      }
      throw new Error('Error desconocido al realizar la petición');
    }
  }

  /**
   * GET /health - Comprueba el estado del servidor
   */
  async checkHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  /**
   * POST /emotion - Obtiene la emoción actual
   */
  async getEmotion(): Promise<EmotionResponse> {
    return this.request<EmotionResponse>('/emotion', {
      method: 'POST',
    });
  }

  /**
   * POST /emotion-from-frame - Analiza emoción desde una imagen enviada
   */
  async analyzeImageEmotion(imageBlob: Blob): Promise<EmotionFromFrameResponse> {
    const formData = new FormData();
    formData.append('image', imageBlob, 'snapshot.jpg');

    const url = `${this.baseUrl}/emotion-from-frame`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        // No establecer Content-Type manualmente, fetch lo hará automáticamente con boundary
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Error HTTP: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('La petición tardó demasiado (timeout)');
        }
        if (error.message.includes('Failed to fetch')) {
          throw new Error('No se pudo conectar con el servidor. Verifica que esté ejecutándose.');
        }
        throw error;
      }
      throw new Error('Error desconocido al realizar la petición');
    }
  }

  /**
   * POST /generate-midi - Genera un archivo MIDI basado en emoción
   * Acepta AbortController opcional para cancelación manual
   */
  async generateMidi(
    params: {
      engine?: 'baseline' | 'transformer_pretrained' | 'transformer_finetuned';
      seed?: number;
      length_bars?: number;
      valence?: number;
      arousal?: number;
      emotion?: string;
      signal?: AbortSignal; // Opcional: permite cancelación manual
    } = {}
  ): Promise<{
    midiBlob: Blob;
    engine: string;
    seed: number;
    length_bars: number;
  }> {
    const queryParams = new URLSearchParams();
    if (params.engine) queryParams.append('engine', params.engine);
    if (params.seed !== undefined) queryParams.append('seed', params.seed.toString());
    if (params.length_bars !== undefined)
      queryParams.append('length_bars', params.length_bars.toString());

    const url = `${this.baseUrl}/generate-midi?${queryParams.toString()}`;

    // Usar AbortController externo si se proporciona, sino crear uno interno para timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout para HF model

    // Si hay un signal externo, abortar el controller interno también
    const externalAbortHandler = () => controller.abort();
    if (params.signal) {
      params.signal.addEventListener('abort', externalAbortHandler);
    }

    try {
      // Preparar body con datos de emoción si están presentes
      const body =
        params.valence !== undefined && params.arousal !== undefined
          ? JSON.stringify({
              valence: params.valence,
              arousal: params.arousal,
              emotion: params.emotion || 'unknown',
            })
          : undefined;

      const response = await fetch(url, {
        method: 'POST',
        headers: body ? { 'Content-Type': 'application/json' } : {},
        body,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      if (params.signal) {
        params.signal.removeEventListener('abort', externalAbortHandler);
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Error HTTP: ${response.status}`);
      }

      // Extraer metadata de headers
      const engine = response.headers.get('X-Engine') || params.engine || 'baseline';
      const seed = parseInt(response.headers.get('X-Seed') || '0');
      const length_bars = parseInt(response.headers.get('X-Length-Bars') || '8');

      // Obtener blob MIDI
      const midiBlob = await response.blob();

      return {
        midiBlob,
        engine,
        seed,
        length_bars,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      if (params.signal) {
        params.signal.removeEventListener('abort', externalAbortHandler);
      }

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('La generación fue cancelada o tardó demasiado (timeout).');
        }
        if (error.message.includes('Failed to fetch')) {
          throw new Error('No se pudo conectar con el servidor. Verifica que esté ejecutándose.');
        }
        throw error;
      }
      throw new Error('Error desconocido al generar MIDI');
    }
  }
}

// Instancia única del cliente API
export const apiClient = new ApiClient(API_BASE_URL);
