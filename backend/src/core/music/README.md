# Generador MIDI Baseline

Este módulo implementa un generador MIDI simple basado en reglas que convierte parámetros musicales (derivados de emociones) en archivos MIDI reproducibles.

## Características

- **Basado en reglas**: Sin modelos de ML, solo lógica composicional explícita
- **Determinista**: Reproducible con semilla aleatoria
- **Configurable**: Controla tempo, modo, densidad, rango tonal, ritmo y dinámica
- **Escalas musicales**: Soporte para modos major y minor
- **Patrones rítmicos**: Complejidad adaptativa (simple, moderado, complejo)

## Uso Básico

### Generar MIDI desde webcam (captura emocional)

```bash
# Captura emoción durante 10 segundos y genera 8 compases
python scripts/generate_baseline_from_webcam.py

# Opciones personalizadas
python scripts/generate_baseline_from_webcam.py --duration 15 --bars 16 --output mi_emocion.mid
```

### Test rápido sin webcam

```bash
# Genera archivos MIDI para diferentes emociones
python scripts/test_midi_generation.py
```

### Uso programático

```python
from core.music import va_to_music_params, generate_midi_baseline

# Coordenadas emocionales (happy)
valence = 0.70
arousal = 0.60

# Convertir a parámetros musicales
params = va_to_music_params(valence, arousal)

# Generar MIDI
output_path = generate_midi_baseline(
    params=params,
    out_path='output.mid',
    length_bars=8,
    seed=42  # Para reproducibilidad
)

print(f"MIDI generado: {output_path}")
```

## Parámetros Musicales

El sistema convierte coordenadas Valence-Arousal en estos parámetros:

| Parámetro | Descripción | Rango |
|-----------|-------------|-------|
| `tempo_bpm` | Velocidad del tempo | 60-180 BPM |
| `mode` | Modo musical | 'major' o 'minor' |
| `density` | Densidad de notas | 0.0-1.0 |
| `pitch_low` | Nota más grave | 48-60 MIDI |
| `pitch_high` | Nota más aguda | 72-84 MIDI |
| `rhythm_complexity` | Complejidad rítmica | 0.0-1.0 |
| `velocity_mean` | Intensidad media | 40-120 |
| `velocity_spread` | Variación de intensidad | 0-30 |

## Mapeo Emoción → Música

- **Arousal (activación)** controla:
  - Tempo (mayor arousal = más rápido)
  - Densidad de notas (más arousal = más notas)
  - Complejidad rítmica (más arousal = más complejo)
  - Intensidad (velocity)

- **Valence (valencia)** controla:
  - Modo musical (positivo = major, negativo = minor)
  - Registro tonal (positivo = más agudo)

## Estructura del Código

```
backend/src/core/music/
├── engines/
│   ├── __init__.py
│   ├── baseline.py         # Generador MIDI baseline
│   └── hf_maestro_remi.py  # Generador con HF Transformers
├── mapping.py              # Mapeo VA → parámetros musicales
└── __init__.py

backend/scripts/
├── generate_baseline_from_webcam.py  # Demo con webcam
└── test_midi_generation.py           # Test sin webcam
```

### Modelos Usados

- **Baseline**: Generador basado en reglas musicales deterministas
- **HF Maestro-REMI**: Modelo transformer preentrenado
  - Checkpoint: [`NathanFradet/Maestro-REMI-bpe20k`](https://huggingface.co/NathanFradet/Maestro-REMI-bpe20k)
  - Entrenado en el dataset MAESTRO con tokenización REMI + BPE
  - 20k vocabulario, arquitectura GPT-2

## Ejemplos de Salida

### Happy (V=+0.70, A=+0.60)
- Tempo: ~132 BPM (rápido)
- Modo: Major (alegre)
- Densidad: Alta (~0.8)
- Registro: Medio-agudo

### Sad (V=-0.70, A=-0.40)
- Tempo: ~78 BPM (lento)
- Modo: Minor (melancólico)
- Densidad: Baja (~0.4)
- Registro: Grave

### Angry (V=-0.60, A=+0.70)
- Tempo: ~136 BPM (muy rápido)
- Modo: Minor (oscuro)
- Densidad: Muy alta (~0.85)
- Registro: Medio

## Limitaciones del Baseline

Este es un generador **baseline** con las siguientes limitaciones intencionales:

1. **Melodía monofónica**: Solo una voz, sin armonía ni acompañamiento
2. **Random walk simple**: La melodía se genera con saltos aleatorios ponderados
3. **Sin estructura formal**: No hay repetición de temas ni desarrollo musical
4. **Escalas fijas**: Solo usa escalas mayor y menor natural
5. **Tónica fija**: Siempre en Do (C)

Estas limitaciones son por diseño para establecer un baseline simple y reproducible que pueda servir como referencia para comparar con generadores más sofisticados (ej. modelos neuronales).

## Instalación de Dependencias

```bash
pip install mido>=1.3.0
```

O instalar todas las dependencias del proyecto:

```bash
pip install -r requirements.txt
```
