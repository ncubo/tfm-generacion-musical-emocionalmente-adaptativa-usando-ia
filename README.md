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
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/app.py
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

- [Backend README](backend/README.md)
- [Frontend README](frontend/README.md)
- [API Documentation](backend/API_DOCUMENTATION.md)

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

