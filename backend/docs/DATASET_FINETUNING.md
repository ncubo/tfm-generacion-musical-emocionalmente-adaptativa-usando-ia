# Dataset Fine-tuning Maestro-REMI

Pipeline de construcción de dataset para fine-tuning del modelo Maestro-REMI-bpe20k con conditioning emocional Valencia-Activación mediante tokens discretos.

## Estructura

```
backend/
├── src/core/music/tokenization/remi.py
├── scripts/
│   ├── generate_va_labels.py
│   ├── build_finetune_dataset.py
│   ├── verify_finetune_dataset.py
│   └── build_full_dataset.sh
└── data/
    ├── lakh_piano_metadata/lakh_piano_subset_labeled.csv
    └── finetune_dataset/
        ├── train/
        ├── val/
        └── dataset_info.json
```

## Pipeline

### 1. Generación de labels VA

```bash
python scripts/generate_va_labels.py
```

Computa valence/arousal heurístico desde características musicales (tempo, velocity, pitch). Salida: CSV con columnas `filename`, `valence`, `arousal`.

### 2. Construcción del dataset

```bash
python scripts/build_finetune_dataset.py
```

**Parámetros:** `--window_size 1024 --stride 512 --bins 9 --train_split 0.9`

**Proceso:**
1. Tokeniza MIDI → REMI IDs
2. Discretiza VA → bins [-4..+4]
3. Windowing con overlap
4. Prepend conditioning tokens: `[VAL=X] [ARO=Y] <música>`
5. Split por archivo (evita leakage)

**Salida:** `data/finetune_dataset/{train,val}` + metadata JSON

### 3. Verificación

```bash
python scripts/verify_finetune_dataset.py --num_examples 3
```

Muestra estadísticas, decode de ejemplos, y valida ausencia de IDs negativos.

## Conditioning Tokens

**Bins:** 9 por dimensión (-4, -3, -2, -1, 0, +1, +2, +3, +4)  
**Formato:** `[VAL=-2]`, `[ARO=+3]`  
**Total:** 18 tokens (9 valence + 9 arousal)  
**IDs:** 20000-20017 (vocab extendido desde 20000)

Los tokens se añaden al tokenizador con `add_to_vocab()` DURANTE la construcción del dataset. Los IDs finales están ya presentes en el dataset guardado (no son placeholders).

## Estadísticas (16/02/2026)

- **Archivos:** 1851/1851 procesados (100%)
- **Ejemplos:** 6489 total (5893 train, 596 val)
- **Window:** 1024 tokens, stride 512
- **Vocab:** 20018 (20000 base + 18 conditioning)
- **Media longitud:** ~945 tokens

**Distribución VA:** Sesgo hacia valence negativo (música contemplativa), arousal uniforme.

## Fine-tuning

```python
# Cargar tokenizador con conditioning tokens (mismo orden que build)
tokenizer = load_remi_tokenizer("Natooz/Maestro-REMI-bpe20k")
tokenizer.add_to_vocab(conditioning_tokens)  # vocab 20000 → 20018

# Redimensionar embeddings del modelo y entrenar
model = AutoModelForCausalLM.from_pretrained("Natooz/Maestro-REMI-bpe20k")
model.resize_token_embeddings(len(tokenizer))
# Trainer con causal LM objective
```

## Notas Técnicas

**Normalización pitches:** Clamp 21-108 antes de tokenizar (evita Pitch_109 error)  
**Data leakage:** Split por archivo (todas las ventanas de un MIDI van al mismo split)  
**Reproducibilidad:** Seeds para shuffle y heurísticas VA  
**Compatibilidad:** Mismo tokenizador que `hf_maestro_remi.py`
