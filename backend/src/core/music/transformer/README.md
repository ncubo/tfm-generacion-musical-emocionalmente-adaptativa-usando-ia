# Generación Musical con Music Transformer (PyTorch)

Generación de música simbólica usando un Music Transformer preentrenado, con condicionamiento indirecto mediante parámetros de sampling derivados de Valence-Arousal.

## Arquitectura

```
backend/src/core/music/transformer/
├── __init__.py                # Exports públicos
├── transformer_infer.py       # Generador principal (TransformerMusicGenerator)
├── sampling.py                # Estrategias de muestreo (temperature, top-k, top-p)
└── io.py                      # Conversión tokens ↔ MIDI (REMI encoding)
```

## Modelo Implementado

### Music Transformer
- **Arquitectura**: Transformer decoder autoregresivo
- **Tokenización**: REMI (REpresentation for Music with Instruments)
- **Framework**: PyTorch
- **Parámetros**: 
  - Vocab size: 512 tokens
  - d_model: 256
  - Heads: 8
  - Layers: 6
  - Context: 2048 tokens

### Tokenización REMI

REMI representa música como secuencia de eventos:
- **Bar**: Marca de compás
- **Position**: Posición dentro del compás
- **Tempo**: Cambios de tempo
- **Chord**: Acordes detectados
- **Pitch**: Nota MIDI
- **Velocity**: Intensidad de la nota
- **Duration**: Duración de la nota

Librería: **miditok >= 3.0** (https://github.com/Natooz/MidiTok)
- Usa backend `symusic` en lugar de `mido` para mayor eficiencia
- API actualizada: `encode()` / `decode()` en lugar de `midi_to_tokens()` / `tokens_to_midi()`

## Condicionamiento Emocional (Sin Fine-Tuning)

**IMPORTANTE**: Este módulo NO entrena ni fine-tunea el modelo. El "condicionamiento emocional" se realiza **indirectamente** mediante:

### 1. Temperature
Control de la aleatoriedad en el sampling:

```python
temperature = 0.6 + (arousal + 1.0) / 2.0 * (1.4 - 0.6)
```

- `A = -1.0` → `temperature = 0.6` (conservador, predecible)
- `A = 0.0` → `temperature = 1.0` (balanceado)
- `A = 1.0` → `temperature = 1.4` (creativo, impredecible)

### 2. Top-p (Nucleus Sampling)
Control del conjunto de tokens considerados:

```python
top_p = 0.85 + (arousal + 1.0) / 2.0 * (0.98 - 0.85)
```

- `A = -1.0` → `top_p = 0.85` (restrictivo)
- `A = 1.0` → `top_p = 0.98` (permisivo)

### 3. Top-k
Número de top tokens considerados:

```python
top_k = 20 + (arousal + 1.0) / 2.0 * (120 - 20)
```

- `A = -1.0` → `top_k = 20` (muy selectivo)
- `A = 1.0` → `top_k = 120` (diverso)

### 4. Primer (Opcional)
Melodía inicial generada desde parámetros musicales:
- **Modo** (major/minor) desde Valence
- **Registro tonal** (pitch_low/pitch_high) desde Valence
- **Tempo** desde Arousal

## Instalación

### 1. Instalar dependencias

```bash
cd backend
pip install -r requirements.txt
```

Dependencias clave:
- `torch>=2.0.0` - PyTorch para el modelo
- `miditok>=3.0.0` - Tokenización MIDI (requiere v3.0+ para symusic backend)
- `mido>=1.3.0` - Manejo de archivos MIDI (usado en módulos complementarios)

### 2. Checkpoint del modelo (Opcional)

Por defecto, el modelo usa **pesos aleatorios** (solo para demostración).

Para producción, se recomienda:

**Opción A: Entrenar desde cero**
```bash
# Entrenar en dataset MAESTRO, Lakh MIDI, etc.
# (No implementado en este módulo - enfoque: inferencia)
```

**Opción B: Usar checkpoint preentrenado**

Fuentes recomendadas:
- **Hugging Face**: https://huggingface.co/models?filter=music
- **Music Transformer PyTorch**: https://github.com/jason9693/MusicTransformer-pytorch
- **Pop Music Transformer**: https://github.com/YatingMusic/remi

Guardar en: `backend/data/checkpoints/music_transformer.pt`

## Uso

### Script Standalone

Genera MIDI desde coordenadas V/A manuales:

```bash
# Happy (V=0.7, A=0.6)
python scripts/generate_transformer_from_va.py \
  --v 0.7 --a 0.6 \
  --out data/outputs/midis/happy_transformer.mid \
  --seed 42

# Sad (V=-0.7, A=-0.4)
python scripts/generate_transformer_from_va.py \
  --v -0.7 --a -0.4 \
  --out data/outputs/midis/sad_transformer.mid

# Excited (V=0.5, A=0.9) con primer
python scripts/generate_transformer_from_va.py \
  --v 0.5 --a 0.9 \
  --out data/outputs/midis/excited_transformer.mid \
  --use-primer \
  --max-length 1024
```

**Nota**: Ejecutar desde el directorio `backend/`. Usar rutas relativas sin `../` para guardar dentro de `backend/data/`.

**Opciones:**
- `--v VALENCE`: Valence en [-1, 1]
- `--a AROUSAL`: Arousal en [-1, 1]
- `--out PATH`: Path del MIDI generado
- `--checkpoint PATH`: Path al checkpoint (None = pesos aleatorios)
- `--max-length INT`: Longitud máxima en tokens (default: 512)
- `--seed INT`: Semilla para reproducibilidad
- `--use-primer`: Activar primer melody

### Programático (Python)

```python
from core.music.transformer.transformer_infer import TransformerMusicGenerator
from core.music.mapping import va_to_music_params

# 1. Inicializar generador (una sola vez)
generator = TransformerMusicGenerator(
    checkpoint_path=None,  # o "data/checkpoints/music_transformer.pt"
    device='cpu'  # o 'cuda'
)

# 2. Mapear emoción a parámetros musicales
v, a = 0.7, 0.6  # Happy
music_params = va_to_music_params(v, a)

# 3. Generar MIDI
result = generator.generate(
    v=v,
    a=a,
    out_path="happy.mid",
    max_length=512,
    seed=42,
    use_primer=True,
    music_params=music_params
)

print(f"Generado: {result['midi_path']}")
print(f"Tokens: {result['num_tokens']}")
print(f"Temperature: {result['generation_params']['temperature']}")
```

## Integración con API (Futuro)

El generador está listo para integración en el endpoint unificado `/generate-midi`:

```python
# En backend/src/routes/music.py
from core.music.transformer.transformer_infer import TransformerMusicGenerator

@music_bp.route('/generate-midi', methods=['POST'])
def generate_midi():
    engine = request.json.get('engine', 'baseline')  # 'baseline' | 'transformer'
    
    if engine == 'transformer':
        generator = _get_or_create_transformer_generator()
        result = generator.generate(v=valence, a=arousal, ...)
    else:
        # Baseline por reglas
        result = generate_midi_baseline(...)
    
    return jsonify(result)
```

## Comparación: Baseline vs Transformer

| Característica | Baseline (Reglas) | Transformer (PyTorch) |
|----------------|-------------------|------------------------|
| **Método** | Algoritmos deterministas | Modelo generativo (Transformer) |
| **Condicionamiento** | Directo (V,A → params) | Indirecto (V,A → sampling) |
| **Coherencia** | ⭐⭐ (reglas simples) | ⭐⭐⭐⭐ (aprendida, si hay checkpoint) |
| **Diversidad** | ⭐⭐ (limitada) | ⭐⭐⭐⭐⭐ (alta) |
| **Velocidad** | ⚡ Instantáneo | 🐌 ~1-3s (GPU), ~5-10s (CPU) |
| **Dependencias** | Mínimas (mido) | Medias (PyTorch, miditok) |
| **Tamaño** | ~1 KB | ~10-50 MB (modelo) |
| **Setup** | ✅ Listo | ⚙️ Requiere PyTorch |
| **Uso** | Prototipado, baseline | Producción (con checkpoint) |

## Troubleshooting

### Error: "miditok no está instalado" o problemas de importación
```bash
# Asegurarse de instalar miditok 3.0+
pip install "miditok>=3.0.0"
```

Si ya tienes una versión antigua:
```bash
pip install --upgrade miditok
```

### Error: "'TrackTick' object is not iterable" o similar

Esto indica que estás usando miditok < 3.0 con código para miditok 3.0+. La solución:

```bash
pip install --upgrade "miditok>=3.0.0"
```

Cambios en miditok 3.0:
- Backend cambió de `mido` a `symusic` (más eficiente)
- API: `tokens_to_midi()` → `decode()`
- API: `midi_to_tokens()` → `encode()`
- Configuración: Se usa `TokenizerConfig` en lugar de parámetros directos

### Error: "torch no disponible"
```bash
# CPU
pip install torch

# GPU (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Salida MIDI vacía o inválida

El modelo con **pesos aleatorios** puede generar secuencias sin sentido musical. Soluciones:

1. **Usar checkpoint preentrenado** (recomendado)
2. **Entrenar el modelo** en dataset musical (MAESTRO, Lakh MIDI)
3. **Ajustar parámetros de sampling**:
   - Reducir temperature (más conservador)
   - Reducir top-k (más restrictivo)
   - Aumentar max_length (secuencias más largas)

### Generación lenta

- Primera generación es lenta (carga del modelo)
- Usar GPU acelera significativamente: `device='cuda'`
- Reducir `max_length` para secuencias más cortas

## Referencias

- **Music Transformer Paper**: Huang et al. (2018) "Music Transformer"
- **REMI Tokenization**: https://github.com/YatingMusic/remi
- **miditok Library**: https://github.com/Natooz/MidiTok
- **PyTorch**: https://pytorch.org/
