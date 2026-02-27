# TFM - Generación Musical Emocionalmente Adaptativa usando IA

Sistema que detecta la emoción del usuario en tiempo real mediante visión por computadora y genera música MIDI adaptada a su estado emocional. Implementa tres motores de generación: un motor heurístico basado en reglas, un Transformer preentrenado sobre MAESTRO y un Transformer con fine-tuning sobre coordenadas Valence--Arousal.

## Estructura del Proyecto

```
.
├── backend/            # API Flask + motores de generación musical
│   ├── src/            # Código fuente principal
│   ├── scripts/        # Benchmarks, demos y análisis
│   ├── data/           # Datasets (Lakh piano, fine-tuning)
│   ├── models/         # Modelo fine-tuned local (si aplica)
│   ├── results/        # Outputs de benchmarks
│   └── docs/           # Documentación técnica del backend
├── frontend/           # UI React + TypeScript
└── finetune_bundle/    # Entorno autónomo de fine-tuning (Google Colab)
```

## Inicio Rápido

### Backend (Flask)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # Revisar configuración si es necesario
python -m src.app
```

El backend estará disponible en `http://localhost:5000`

### Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

El frontend estará disponible en `http://localhost:5173`

## Motores de Generación

| Motor | Descripción | Latencia |
|---|---|---|
| `baseline` | Reglas heurísticas, mapeo VA determinista | ~1 ms |
| `transformer_pretrained` | `Natooz/Maestro-REMI-bpe20k` | ~4 s (CPU) |
| `transformer_finetuned` | `mmayorga/maestro-remi-finetuned-va` | ~4 s (CPU) |

## Documentación

### General
- [Backend README](backend/README.md)
- [Frontend README](frontend/README.md)
- [Fine-tuning Bundle README](finetune_bundle/README.md)
- [Documentación API](backend/docs/DOCUMENTACION_API.md)

### Dataset y Fine-tuning
- [Preparación de Dataset](backend/docs/PREPARACION_DATASET.md)
- [Dataset Fine-tuning](backend/docs/DATASET_FINETUNING.md)

### Evaluación y Benchmarks
- [Métricas Implementadas](backend/docs/METRICAS_IMPLEMENTADAS.md)
- [Benchmark de Modelos](backend/docs/BENCHMARK.md)
- [Benchmark Final](backend/docs/BENCHMARK_FINAL.md)
- [Comparación Modelos](backend/docs/COMPARACION_MODELOS.md)
- [Evaluación de Rendimiento](backend/docs/EVALUACION_RENDIMIENTO.md)

## Tecnologías Principales

### Backend
- Python 3.10+
- Flask (API REST, patrón Application Factory)
- DeepFace (detección de emociones facial)
- OpenCV (captura de cámara)
- Hugging Face Transformers + PyTorch (modelos Transformer)
- MidiTok / MidiTok REMI (tokenización musical)
- symusic (procesamiento de MIDI)
- mido (generación MIDI baseline)

### Frontend
- React 19
- TypeScript
- Vite
- Tailwind CSS
- @tonejs/midi + soundfont-player (reproducción MIDI en navegador)

### Fine-tuning
- Hugging Face Trainer
- Google Colab (GPU Tesla T4)
- Modelo publicado: `mmayorga/maestro-remi-finetuned-va`

## Desarrollo

Consulta los README individuales de cada módulo para instrucciones detalladas.

