# Fine-tuning Maestro-REMI

Fine-tuning de Maestro-REMI-bpe20k con conditioning tokens de Valencia-Activación para Google Colab.

**Opción recomendada:** Usar el notebook `finetune_maestro.ipynb` en Colab (incluye todos los pasos pre-configurados).

**Opción manual:** Seguir los comandos de este README.

## 1. Generar dataset (local)

```bash
# Desde backend/
python scripts/generate_va_labels.py
python scripts/build_finetune_dataset.py
cp -r data/finetune_dataset ../finetune_bundle/data/
tar -czf finetune_bundle.tar.gz finetune_bundle
```

Genera ~6500 ejemplos MIDI con etiquetas VA (train/val/dataset_info.json).

## 2. Colab: Setup

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!tar -xzf "/content/drive/MyDrive/TFM/finetune_bundle.tar.gz"
%cd finetune_bundle
!pip install -r requirements.txt
```

## 3. Entrenar

**Test rápido (50 steps):**
```python
!python train_maestro_finetune.py \
  --dataset_dir data/finetune_dataset \
  --output_dir models/maestro_finetuned \
  --fp16 \
  --quick_test
```

**Entrenamiento completo (500 steps, ~60-90 min):**
```python
!python train_maestro_finetune.py \
  --dataset_dir data/finetune_dataset \
  --output_dir models/maestro_finetuned \
  --fp16 \
  --max_steps 500 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4
```

## 4. Descargar modelo

```python
!tar -czf maestro_finetuned.tar.gz models/maestro_finetuned
!cp maestro_finetuned.tar.gz "/content/drive/MyDrive/"
```

**En local, extraer al backend:**
```bash
# Desde raíz del proyecto
tar -xzf maestro_finetuned.tar.gz -C backend/
# El modelo queda en backend/models/maestro_finetuned/final/
```

## Troubleshooting

**OOM:** `--per_device_train_batch_size 1 --gradient_accumulation_steps 8`  
**Dataset no encontrado:** Copiar `backend/data/finetune_dataset` al bundle antes de crear `.tar.gz`


