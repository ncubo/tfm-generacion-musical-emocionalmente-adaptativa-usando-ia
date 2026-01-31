# Frontend - Generación Musical Emocionalmente Adaptativa

Frontend desarrollado con React, TypeScript, Vite y Tailwind CSS para interactuar con el backend Flask de generación musical basada en emociones.

## Tecnologías

- **React 19** - Librería de UI
- **TypeScript** - Tipado estático
- **Vite** - Build tool y dev server
- **Tailwind CSS** - Framework de estilos
- **ESLint + Prettier** - Calidad de código
- **Husky + lint-staged** - Git hooks

## Instalación

```bash
# Instalar dependencias
npm install

# Copiar archivo de configuración de entorno
cp .env.example .env
```

## Configuración

Edita el archivo `.env` y configura la URL del backend:

```
VITE_API_BASE_URL=http://localhost:5000
```

## Scripts Disponibles

```bash
# Iniciar servidor de desarrollo
npm run dev

# Build de producción
npm run build

# Vista previa del build
npm run preview

# Ejecutar linter
npm run lint

# Formatear código
npm run format
```

## Estructura del Proyecto

```
src/
├── api/
│   └── client.ts          # Cliente API con fetch
├── components/
│   ├── StatusCard.tsx     # Tarjeta de estado del servidor
│   ├── EmotionCard.tsx    # Tarjeta de emoción actual
│   └── MidiCard.tsx       # Tarjeta de generación MIDI
├── pages/
│   └── LiveDemo.tsx       # Página principal
├── App.tsx                # Componente raíz
├── main.tsx               # Punto de entrada
└── index.css              # Estilos globales (Tailwind)
```

## Funcionalidades

### 1. Estado del Servidor
- Comprueba la conexión con el backend
- Muestra el estado de salud del servidor

### 2. Emoción Actual
- Obtiene la emoción detectada desde el backend
- Visualiza valores de valencia y activación
- Muestra representación gráfica del espacio V-A

### 3. Generación MIDI
- Genera archivos MIDI basados en emoción
- Muestra la ruta del archivo generado
- Permite copiar la ruta al portapapeles
- Visualiza parámetros musicales utilizados

## Calidad de Código

### ESLint
Configurado con reglas para React, TypeScript y Prettier.

### Prettier
Formateo automático con configuración personalizada.

### Husky + lint-staged
Pre-commit hook que ejecuta:
- ESLint con auto-fix en archivos `.ts` y `.tsx`
- Prettier en archivos modificados

## Notas de Desarrollo

- Todo el código (variables, funciones, componentes) está en **inglés**
- Todo el texto visible en la UI está en **español**
- Los comentarios están en español solo donde aportan valor
- No se usan librerías de UI pesadas (solo React + Tailwind)
- Manejo robusto de estados de carga y errores

## Próximos Pasos

- Implementar captura de webcam en el frontend
- Añadir visualización en tiempo real
- Integrar reproductor MIDI
- Añadir tests unitarios y de integración

