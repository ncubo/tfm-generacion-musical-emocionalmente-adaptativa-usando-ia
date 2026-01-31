import { ENV } from '../config/env';

// Configuración base para el cliente API
const API_BASE_URL = ENV.API_BASE_URL;

// Tipos para las respuestas del backend
export interface HealthResponse {
  status: string;
}

export interface EmotionResponse {
  emotion: string;
  valence: number;
  arousal: number;
}

export interface GenerateMidiRequest {
  emotion?: string;
  valence?: number;
  arousal?: number;
  params?: Record<string, unknown>;
}

export interface GenerateMidiResponse {
  emotion: string;
  valence: number;
  arousal: number;
  params: Record<string, unknown>;
  midi_path: string;
}

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
   * POST /generate-midi - Genera un archivo MIDI basado en emoción
   */
  async generateMidi(data: GenerateMidiRequest = {}): Promise<GenerateMidiResponse> {
    return this.request<GenerateMidiResponse>('/generate-midi', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
}

// Instancia única del cliente API
export const apiClient = new ApiClient(API_BASE_URL);
