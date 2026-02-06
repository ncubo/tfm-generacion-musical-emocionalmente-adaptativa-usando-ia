# Arquitectura de Engines para Generación MIDI

Este documento describe la implementación de la arquitectura modular de engines para generación musical emocionalmente adaptativa.

## 🎯 Objetivo

Implementar una arquitectura extensible con múltiples motores (engines) de generación MIDI:
- **baseline**: Generación basada en reglas deterministas
- **transformer_pretrained**: Modelo SkyTNT/midi-model preentrenado desde Hugging Face
- **transformer_finetuned**: Placeholder para futuro fine-tuning (retorna 501)

## 📁 Estructura Creada

```
backend/
├── models/
│   └── transformer_pretrained/
│       ├── README.md
│       └── model.ckpt (descargado vía script)
├── scripts/
│   ├── download_transformer_pretrained.py
│   └── verify_transformer_pretrained.py
└── src/
    └── core/
        └── music/
            └── engines/
                ├── __init__.py
                ├── base.py                          # Interfaz MusicGenerationEngine
                ├── registry.py                      # Registry de engines
                ├── baseline_engine.py               # Engine de reglas
                ├── transformer_pretrained_engine.py # Engine SkyTNT
                └── transformer_finetuned_engine.py  # Placeholder

frontend/
└── src/
    ├── types/index.ts          # Tipos actualizados (MusicEngine, EngineInfo)
    ├── api/client.ts           # Método getEngines() añadido
    ├── components/
    │   └── MidiCard.tsx        # Actualizado con selector de motor
    └── pages/
        └── LiveDemo.tsx        # MidiCard integrado
```

## 🚀 Instrucciones de Uso

### 1. Configurar Backend

#### Instalar dependencias (si no están instaladas):

```bash
cd backend
pip install -r requirements.txt
```

Esto instalará:
- `torch` (PyTorch)
- `huggingface_hub` (para descargar modelos)
- Dependencias existentes (flask, mido, deepface, etc.)

#### Descargar checkpoint del transformer preentrenado:

```bash
python scripts/download_transformer_pretrained.py
```

Esto descarga `model.ckpt` desde `skytnt/midi-model` en Hugging Face a `models/transformer_pretrained/`.

#### Verificar instalación:

```bash
python scripts/verify_transformer_pretrained.py
```

Deberías ver:
```
✅ OK: Loaded pretrained weights
```

### 2. Ejecutar Backend

```bash
cd backend
source .venv/bin/activate  # o tu entorno virtual
python src/app.py
```

El servidor estará en `http://localhost:5000`

### 3. Ejecutar Frontend

```bash
cd frontend
npm install  # si no lo has hecho
npm run dev
```

El frontend estará en `http://localhost:5173`

## 🎮 Uso de la Interfaz

1. **Detección emocional**:
   - Permite acceso a la cámara
   - Opcionalmente captura frames en tiempo real o manual
   - Se detecta emoción → coordenadas V/A

2. **Generación MIDI**:
   - Selecciona motor en el dropdown:
     - **Baseline (reglas)**: Rápido, predecible
     - **Transformer (preentrenado)**: Usa modelo SkyTNT (requiere checkpoint)
     - **Transformer (fine-tuned)**: No disponible (501)
   
3. **Clic en "Generar MIDI"**:
   - Se genera el archivo MIDI
   - Se muestra ruta, parámetros y coordenadas V/A

## 🔧 API

### Endpoint Principal: `POST /generate-midi`

```json
{
  "engine": "baseline" | "transformer_pretrained" | "transformer_finetuned",
  "valence": 0.7,     // Opcional si se usa webcam
  "arousal": 0.6,     // Opcional si se usa webcam
  "seed": 42          // Opcional (reproducibilidad)
}
```

**Respuesta exitosa (200)**:
```json
{
  "engine": "transformer_pretrained",
  "valence": 0.7,
  "arousal": 0.6,
  "generation_params": {
    "temperature": 1.06,
    "top_k": 84,
    "top_p": 0.91,
    ...
  },
  "midi_path": "/path/to/transformer_pretrained_20260205_123045.mid"
}
```

**Errores**:
- **400**: Engine inválido
- **500**: Checkpoint faltante (transformer_pretrained sin descargar)
- **501**: Engine no disponible (transformer_finetuned)

### Endpoint Auxiliar: `GET /engines`

Lista engines disponibles:

```json
{
  "engines": [
    {
      "name": "baseline",
      "description": "Generación basada en reglas...",
      "available": true
    },
    {
      "name": "transformer_pretrained",
      "description": "Transformer preentrenado (SkyTNT)...",
      "available": true
    },
    {
      "name": "transformer_finetuned",
      "description": "Transformer fine-tuned...",
      "available": false
    }
  ]
}
```

## 🧠 Condicionamiento Emocional

### Baseline
- **Directo**: V/A → parámetros musicales explícitos (tempo, modo, densidad)
- Mapeo determinista definido en `mapping.py`

### Transformer Pretrained (SkyTNT)
- **Indirecto**: V/A → sampling parameters (temperature, top_k, top_p)
- El modelo **no** recibe V/A directamente (generación incondicional)
- Control emocional a través de:
  - **Arousal alto** → temperature alta, top_k grande (música variada/energética)
  - **Arousal bajo** → temperature baja, top_k pequeño (música calmada/predecible)

### Transformer Finetuned (Futuro)
- **Directo**: V/A embeddings como entrada al modelo
- Requiere entrenamiento con datos emocionales anotados

## ⚠️ Notas Importantes

1. **Checkpoints NO están en git**: 
   - Los archivos `.ckpt` están en `.gitignore`
   - Cada desarrollador debe descargarlos con los scripts

2. **Conversión tokens → MIDI**:
   - Actualmente es un **placeholder**
   - El transformer genera tokens pero la conversión final a MIDI es simplificada
   - Para producción, implementar tokenizer REMI real (miditok)

3. **Compatibilidad**:
   - El endpoint legacy `/generate-midi-legacy` mantiene compatibilidad

## 🧪 Testing

### Probar baseline:
```bash
curl -X POST http://localhost:5000/generate-midi \
  -H "Content-Type: application/json" \
  -d '{"engine":"baseline","valence":0.7,"arousal":0.6}'
```

### Probar transformer_pretrained (requiere checkpoint):
```bash
curl -X POST http://localhost:5000/generate-midi \
  -H "Content-Type: application/json" \
  -d '{"engine":"transformer_pretrained","valence":0.7,"arousal":0.6,"seed":42}'
```

### Probar transformer_finetuned (esperar 501):
```bash
curl -X POST http://localhost:5000/generate-midi \
  -H "Content-Type: application/json" \
  -d '{"engine":"transformer_finetuned","valence":0.7,"arousal":0.6}'
```

## 📚 Referencias

- [SkyTNT/midi-model](https://huggingface.co/skytnt/midi-model)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub)
