#!/bin/bash
# Script para construir el dataset completo de fine-tuning
# Usa todos los archivos disponibles (2000 MIDIs)

set -e  # Salir si hay error

echo "=========================================="
echo "CONSTRUCCIÓN DATASET COMPLETO FINE-TUNING"
echo "=========================================="
echo ""

# Activar entorno virtual
source .venv/bin/activate

# Verificar que existan las labels
if [ ! -f "data/lakh_piano_metadata/lakh_piano_subset_labeled.csv" ]; then
    echo "Labels VA no encontradas. Generando..."
    python scripts/generate_va_labels.py
    echo ""
fi

# Construir dataset completo
echo "Construyendo dataset con TODOS los archivos..."
echo "Configuración:"
echo "  - Window size: 1024 tokens"
echo "  - Stride: 512 tokens"
echo "  - Bins VA: 9 (rango -4..+4)"
echo "  - Train/val split: 90/10"
echo ""

python scripts/build_finetune_dataset.py \
    --window_size 1024 \
    --stride 512 \
    --min_window_tokens 256 \
    --bins 9 \
    --train_split 0.9 \
    --seed 42

echo ""
echo "=========================================="
echo "Dataset construido. Verificando..."
echo "=========================================="
echo ""

# Verificar dataset
python scripts/verify_finetune_dataset.py \
    --num_examples 3 \
    --split both

echo ""
echo "=========================================="
echo "PROCESO COMPLETADO"
echo "=========================================="
echo ""
echo "Dataset listo para fine-tuning en:"
echo "  backend/data/finetune_dataset/"
echo ""
echo "Próximo paso: entrenar el modelo"
echo ""
