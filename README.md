# TFM - Generación Musical Emocionalmente Adaptativa usando IA

Sistema de generación musical basado en detección de emociones mediante visión por computadora.

## Estructura del Proyecto

```
.
├── backend/          # API Flask + ML pipelines
└── frontend/         # UI React + TypeScript
```

## Inicio Rápido

### Backend (Flask)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.app
```

El backend estará disponible en `http://localhost:5000`

### Frontend (React)

```bash
cd frontend
npm install
cp .env.example .env  # Configurar VITE_API_BASE_URL
npm run dev
```

El frontend estará disponible en `http://localhost:5173`

## Documentación

### General
- [Backend README](backend/README.md)
- [Frontend README](frontend/README.md)
- [Documentación API](backend/DOCUMENTACION_API.md)

### Dataset y Fine-tuning
- [Preparación de Dataset](backend/PREPARACION_DATASET.md)
- [Dataset Fine-tuning](backend/DATASET_FINETUNING.md)

### Evaluación y Benchmarks
- [Benchmark de Modelos](backend/BENCHMARK.md)
- [Comparación Modelos](backend/COMPARACION_MODELOS.md)
- [Evaluación de Rendimiento](backend/EVALUACION_RENDIMIENTO.md)

## Tecnologías Principales

### Backend
- Python 3.x
- Flask (API REST)
- DeepFace (detección de emociones)
- mido (generación MIDI)
- OpenCV (captura de cámara)

### Frontend
- React 19
- TypeScript
- Vite
- Tailwind CSS
- ESLint + Prettier

## Desarrollo

Consulta los README individuales de cada módulo para instrucciones detalladas de desarrollo.

