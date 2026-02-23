# API REST - TFM Generación Musical Emocional

API Flask para el sistema de generación musical emocionalmente adaptativa.

## Inicio Rápido

### Instalación de dependencias

```bash
cd backend
pip install -r requirements.txt
```

### Ejecutar el servidor

```bash
cd backend
python -m src.app
```

El servidor se iniciará en `http://localhost:5000`

## Endpoints

### 1. Health Check

Verifica que el servicio está activo.

**Endpoint:** `GET /health`

**Respuesta:**
```json
{
  "status": "ok"
}
```

**Ejemplo curl:**
```bash
curl http://localhost:5000/health
```

---

### 2. Detectar Emoción

Captura un frame de la webcam y detecta la emoción facial actual.

**Endpoint:** `POST /emotion`

**Respuesta:**
```json
{
  "emotion": "happy",
  "valence": 0.68,
  "arousal": 0.58
}
```

**Campos:**
- `emotion` (string): Emoción normalizada detectada (neutral, happy, sad, angry, fear, calm, excited)
- `valence` (float): Valencia en rango [-1, 1] (-1=muy negativo, +1=muy positivo)
- `arousal` (float): Activación en rango [-1, 1] (-1=muy calmado, +1=muy activado)

**Ejemplo curl:**
```bash
curl -X POST http://localhost:5000/emotion
```

**Ejemplo Python:**
```python
import requests

response = requests.post('http://localhost:5000/emotion')
data = response.json()
print(f"Emoción: {data['emotion']}")
print(f"Valence: {data['valence']}, Arousal: {data['arousal']}")
```

---

### 3. Generar MIDI

Genera un archivo MIDI baseline basado en el estado emocional actual del usuario.

**Endpoint:** `POST /generate-midi`

**Respuesta:**
```json
{
  "emotion": "happy",
  "valence": 0.68,
  "arousal": 0.58,
  "params": {
    "tempo_bpm": 132,
    "mode": "major",
    "density": 0.74,
    "pitch_low": 64,
    "pitch_high": 76,
    "rhythm_complexity": 0.74,
    "velocity_mean": 92,
    "velocity_spread": 22
  },
  "midi_path": "/path/to/output/emotion_20260130_123045.mid"
}
```

**Campos:**
- `emotion` (string): Emoción detectada
- `valence` (float): Valencia utilizada
- `arousal` (float): Activación utilizada
- `params` (object): Parámetros musicales generados
  - `tempo_bpm` (int): Tempo en beats por minuto [60-180]
  - `mode` (string): Modo musical ("major" o "minor")
  - `density` (float): Densidad de notas [0.2-1.0]
  - `pitch_low` (int): Nota MIDI más grave [48-60]
  - `pitch_high` (int): Nota MIDI más aguda [72-84]
  - `rhythm_complexity` (float): Complejidad rítmica [0.0-1.0]
  - `velocity_mean` (int): Intensidad promedio de notas [40-120]
  - `velocity_spread` (int): Variación de intensidad [0-30]
- `midi_path` (string): Ruta absoluta al archivo MIDI generado

**Ejemplo curl:**
```bash
curl -X POST http://localhost:5000/generate-midi
```

**Ejemplo Python:**
```python
import requests
import json

response = requests.post('http://localhost:5000/generate-midi')
data = response.json()

print(f"Emoción: {data['emotion']}")
print(f"Parámetros musicales:")
print(json.dumps(data['params'], indent=2))
print(f"Archivo MIDI: {data['midi_path']}")
```

---

## Workflow Típico

1. **Verificar estado del servicio:**
   ```bash
   curl http://localhost:5000/health
   ```

2. **Detectar emoción actual:**
   ```bash
   curl -X POST http://localhost:5000/emotion
   ```

3. **Generar música basada en la emoción:**
   ```bash
   curl -X POST http://localhost:5000/generate-midi
   ```

4. **Reproducir el archivo MIDI generado** con tu reproductor preferido

---

## Configuración

La aplicación se puede configurar pasando un diccionario de configuración:

```python
from src.app import create_app

config = {
    'OUTPUT_DIR': '/custom/output/path',
    'DEBUG': True,
    'PORT': 8080
}

app = create_app(config=config)
app.run()
```

---

## Testing con Postman

1. Importar los siguientes endpoints como colección
2. Configurar base URL: `http://localhost:5000`
3. Ejecutar requests en secuencia para probar el flujo completo

---

## Manejo de Errores

Todos los endpoints pueden retornar errores HTTP 500 en caso de fallo:

```json
{
  "error": "Error al detectar emoción",
  "message": "Detalles del error..."
}
```

---

## Arquitectura

```
backend/src/
├── app.py                    # Punto de entrada Flask
├── routes/                   # Blueprints de la API
│   ├── __init__.py
│   ├── health.py            # GET /health
│   ├── emotion.py           # POST /emotion
│   └── music.py             # POST /generate-midi
└── core/                     # Lógica de dominio
    ├── camera/              # Captura de webcam
    ├── emotion/             # Detección emocional
    ├── pipeline/            # Pipeline integrado
    ├── va/                  # Mapeo Valencia-Activación
    └── music/               # Generación MIDI
```

---

## Notas de Implementación

- **Webcam:** Se inicializa una única vez al arrancar el servidor, no en cada request
- **Pipeline emocional:** Mantiene estado con suavizado temporal (ventana de 5 frames)
- **CORS:** Habilitado para permitir requests desde cualquier origen (frontend)
- **Sincronía:** API completamente síncrona (no usa async/await todavía)
- **Archivos MIDI:** Se guardan en `backend/output/` con timestamp único
