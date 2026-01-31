/**
 * Validación y configuración de variables de entorno
 */

function getEnvVar(key: string, defaultValue?: string): string {
  const value = import.meta.env[key];

  if (!value && !defaultValue) {
    console.warn(`Variable de entorno ${key} no está definida. Usando valor por defecto.`);
  }

  return (value || defaultValue) as string;
}

// Variables de entorno validadas
export const ENV = {
  API_BASE_URL: getEnvVar('VITE_API_BASE_URL', 'http://localhost:5000'),
  DEV: import.meta.env.DEV,
  PROD: import.meta.env.PROD,
} as const;
