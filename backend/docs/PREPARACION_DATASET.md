# Subset Piano-Only del Lakh MIDI Dataset

Extracción de ~5000 archivos MIDI piano-only del Lakh Dataset para fine-tuning.

## 1. Descarga

```bash
cd data/lakh_raw
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
tar -xzf lmd_matched.tar.gz
```

Fuente: https://colinraffel.com/projects/lmd/ (45129 archivos, ~45 GB)

## 2. Extracción Piano

```bash
python scripts/prepare_lakh_piano_subset.py \
    --input_dir data/lakh_raw/lmd_matched \
    --output_dir data/lakh_piano_clean \
    --max_files 5000 --seed 42
```

Detección: Program 0-7 o "piano" en nombre. Merge a mono-track. Filtros: 10-300s, ≥50 notas, rango ≥12 semitonos.

## 3. Análisis

```bash
python scripts/analyze_lakh_subset.py
```

## Salidas

```
data/lakh_piano_clean/  # ~5000 MIDIs
data/lakh_piano_metadata/lakh_piano_subset_metadata.csv
```
