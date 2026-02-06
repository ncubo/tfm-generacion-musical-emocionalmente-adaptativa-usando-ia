import { ENV } from '../config/env';
import type {
  HealthResponse,
  EmotionResponse,
  EmotionFromFrameResponse,
  GenerateMidiRequest,
  GenerateMidiResponse,
  EnginesListResponse,
} from '../types';

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
   */
  async generateMidi(data: GenerateMidiRequest = {}): Promise<GenerateMidiResponse> {
    return this.request<GenerateMidiResponse>('/generate-midi', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /**
   * GET /engines - Lista los engines de generación disponibles
   */
  async getEngines(): Promise<EnginesListResponse> {
    return this.request<EnginesListResponse>('/engines');
  }
}

// Instancia única del cliente API
export const apiClient = new ApiClient(API_BASE_URL);
