# Preparación Dataset Lakh MIDI - Piano Only

Preparar subset piano-only de ~2000 archivos MIDI del Lakh Dataset para fine-tuning de Maestro-REMI.

## 1. Descargar Dataset

```bash
cd data/lakh_raw
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
tar -xzf lmd_matched.tar.gz
cd ../..
```

Fuente: https://colinraffel.com/projects/lmd/ (45,129 archivos MIDI, ~45 GB)

## 2. Preparar Subset Piano-Only

```bash
python scripts/prepare_lakh_piano_subset.py \
    --input_dir data/lakh_raw/lmd_matched \
    --output_dir data/lakh_piano_clean \
    --max_files 2000 \
    --min_notes 100 \
    --seed 42
```

Procesamiento:
- Detecta pistas piano (program 0-7 o nombre contiene "piano")
- Mergea todas en un solo track
- Descarta otros instrumentos
- Filtra: duración 10-300s, notas ≥100, rango ≥12 semitonos

## 3. Analizar Estadísticas

```bash
python scripts/analyze_lakh_subset.py
```

Muestra distribución de duración, notas, rango tonal, etc.

## Salidas

```
data/lakh_piano_clean/        # ~2000 MIDIs piano-only
data/lakh_piano_metadata/
  └── lakh_piano_subset_metadata.csv  # metadata completa
```

## Próximos Pasos

1. Etiquetado heurístico V/A
2. Tokenización REMI
3. Fine-tuning Maestro-REMI-bpe20k
