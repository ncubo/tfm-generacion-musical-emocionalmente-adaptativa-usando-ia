#!/usr/bin/env python3
"""
Fine-tuning de Maestro-REMI-bpe20k con conditioning tokens VA.
Optimizado para Colab con GPU (T4/L4) o CPU fallback.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import miditok
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class TrainSummary:
    model_name: str
    device: str
    used_fp16: bool
    used_bf16: bool
    train_examples: int
    val_examples: int
    num_train_epochs: float
    max_steps: int
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    train_runtime_sec: float
    train_loss: Optional[float]
    eval_loss: Optional[float]
    perplexity: Optional[float]
    output_dir: str


def load_conditioning_tokens(dataset_dir: str) -> list:
    """Carga conditioning tokens desde dataset_info.json."""
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"dataset_info.json no encontrado en {dataset_dir}")
    
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    conditioning_tokens = info.get('conditioning_tokens', [])
    
    if len(conditioning_tokens) != 18:
        raise ValueError(f"Se esperaban 18 conditioning tokens, se encontraron {len(conditioning_tokens)}")
    
    print(f"Conditioning tokens cargados: {len(conditioning_tokens)} tokens")
    print(f"  Ejemplos: {conditioning_tokens[:2]} ... {conditioning_tokens[-2:]}")
    
    return conditioning_tokens


def setup_tokenizer(model_name: str, conditioning_tokens: list):
    """Configura tokenizer REMI con conditioning tokens usando MidiTok API."""
    print(f"Cargando tokenizer REMI desde {model_name}")
    
    # Cargar tokenizer REMI (MidiTok, NO transformers tokenizer)
    tokenizer = miditok.REMI.from_pretrained(model_name)
    original_vocab_size = len(tokenizer)
    
    print(f"Vocab size original: {original_vocab_size}")
    
    # Añadir conditioning tokens usando API de MidiTok
    # IMPORTANTE: deben añadirse en el mismo orden que en build_finetune_dataset.py
    print(f"Añadiendo {len(conditioning_tokens)} conditioning tokens...")
    for token in conditioning_tokens:
        tokenizer.add_to_vocab([token])
    
    new_vocab_size = len(tokenizer)
    num_added = new_vocab_size - original_vocab_size
    
    print(f"Tokens añadidos: {num_added}")
    print(f"Nuevo vocab size: {new_vocab_size}")
    
    if new_vocab_size != 20018:
        raise ValueError(f"Vocab size esperado: 20018, obtenido: {new_vocab_size}")
    
    print("Conditioning tokens añadidos exitosamente al tokenizer REMI")
    
    return tokenizer


def setup_model(model_name: str, tokenizer, use_fp16: bool, use_bf16: bool):
    """Configura modelo y redimensiona embeddings."""
    print(f"Cargando modelo desde {model_name}")
    
    # Cargar modelo en float32 - el Trainer se encarga de fp16/bf16
    # Si cargamos directamente en fp16 causa conflicto con el gradient scaler
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    
    # Redimensionar embeddings para incluir conditioning tokens
    old_embeddings_size = model.get_input_embeddings().weight.shape[0]
    print(f"Tamaño embeddings original: {old_embeddings_size}")
    
    model.resize_token_embeddings(len(tokenizer))
    
    new_embeddings_size = model.get_input_embeddings().weight.shape[0]
    print(f"Tamaño embeddings nuevo: {new_embeddings_size}")
    
    return model


def simple_collator(features):
    """
    Collator con padding para batching.
    Los datos están preprocesados pero pueden tener diferentes longitudes.
    """
    # Encontrar la longitud máxima en el batch
    max_length = max(len(f['input_ids']) for f in features)
    
    # Pad token (usamos 0 como en la mayoría de tokenizers)
    pad_token_id = 0
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    
    for f in features:
        input_ids = f['input_ids']
        attention_mask = f['attention_mask']
        labels = f['labels']
        
        # Calcular padding necesario
        padding_length = max_length - len(input_ids)
        
        # Aplicar padding a la derecha
        padded_input_ids = input_ids + [pad_token_id] * padding_length
        padded_attention_mask = attention_mask + [0] * padding_length
        # Para labels, usar -100 (ignorado por la loss)
        padded_labels = labels + [-100] * padding_length
        
        batch_input_ids.append(torch.tensor(padded_input_ids))
        batch_attention_mask.append(torch.tensor(padded_attention_mask))
        batch_labels.append(torch.tensor(padded_labels))
    
    batch = {
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'labels': torch.stack(batch_labels)
    }
    
    return batch


def compute_metrics(eval_pred):
    """Calcula métricas de evaluación."""
    return {}


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tuning de Maestro-REMI-bpe20k para Colab (GPU/CPU)'
    )
    
    # Rutas
    parser.add_argument('--model_name', default='Natooz/Maestro-REMI-bpe20k', help='Modelo HuggingFace')
    parser.add_argument('--dataset_dir', default='data/finetune_dataset', help='Directorio del dataset')
    parser.add_argument('--output_dir', default='models/maestro_finetuned', help='Directorio de salida')
    
    # Hiperparámetros
    parser.add_argument('--num_train_epochs', type=float, default=5.0)
    parser.add_argument('--max_steps', type=int, default=-1, help='Limitar steps (-1 = sin límite)')
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # Logging y guardado
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=200)
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--save_total_limit', type=int, default=2)
    
    # Precision (GPU)
    parser.add_argument('--fp16', action='store_true', help='Usar fp16 en GPU (T4/L4)')
    parser.add_argument('--bf16', action='store_true', help='Usar bf16 en GPU (si disponible)')
    
    # Memory
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    parser.add_argument('--disable_gradient_checkpointing', action='store_true')
    
    # Testing
    parser.add_argument('--quick_test', action='store_true', help='Test rápido con pocos steps')
    
    # Otros
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path al checkpoint para reanudar (ej: models/maestro_finetuned/checkpoint-6000)')
    
    args = parser.parse_args()
    
    if args.disable_gradient_checkpointing:
        args.gradient_checkpointing = False
    
    # Detectar device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_gpu = device == "cuda"
    
    # Quick test mode
    if args.quick_test:
        args.max_steps = 50
        args.save_steps = 25
        args.eval_steps = 25
        args.logging_steps = 10
        args.per_device_train_batch_size = 1
        args.gradient_accumulation_steps = 4
    
    # Precision solo en GPU
    use_fp16 = bool(args.fp16 and is_gpu)
    use_bf16 = bool(args.bf16 and is_gpu)
    
    print("=" * 80)
    print("FINE-TUNING MAESTRO-REMI (CPU/GPU)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Precision: fp16={use_fp16}, bf16={use_bf16}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print("=" * 80)
    
    # Set seed
    set_seed(args.seed)
    
    # Crear output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Cargar conditioning tokens
    print("\n[1/6] Cargando conditioning tokens...")
    conditioning_tokens = load_conditioning_tokens(args.dataset_dir)
    
    # 2. Cargar datasets
    print("\n[2/6] Cargando datasets...")
    train_path = os.path.join(args.dataset_dir, "train")
    val_path = os.path.join(args.dataset_dir, "val")
    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path)
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")
    
    # Verificar estructura
    example = train_dataset[0]
    print(f"Columnas: {list(example.keys())}")
    print(f"Longitud ejemplo: {len(example['input_ids'])} tokens")
    
    # 3. Setup tokenizer
    print("\n[3/6] Configurando tokenizer...")
    tokenizer = setup_tokenizer(args.model_name, conditioning_tokens)
    
    # 4. Setup model
    print("\n[4/6] Configurando modelo...")
    model = setup_model(args.model_name, tokenizer, use_fp16, use_bf16)
    
    # Gradient checkpointing
    if args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing activado")
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    
    # 5. Data collator
    print("\n[5/6] Configurando data collator...")
    data_collator = simple_collator
    
    # 6. Training arguments
    print("\n[6/6] Configurando training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        
        # Epochs y steps
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        
        # Batch size
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Optimización
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        
        # Precision
        fp16=use_fp16,
        bf16=use_bf16,
        
        # Dataloader
        dataloader_num_workers=0 if not is_gpu else 2,
        dataloader_pin_memory=is_gpu,
        
        # Logging
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        prediction_loss_only=True,  # No guardar logits, solo loss (ahorra memoria)
        
        # Guardado
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Otros
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Trainer
    print("\nInicializando Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Entrenar
    print("\n" + "=" * 80)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 80)
    
    start = time.time()
    
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        train_runtime = time.time() - start
        
        print("\n" + "=" * 80)
        print("ENTRENAMIENTO COMPLETADO")
        print("=" * 80)
        
        # Evaluar
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss", None)
        
        # Perplexity
        perplexity = None
        if eval_loss is not None and eval_loss < 20:
            perplexity = float(math.exp(eval_loss))
        
        # Guardar modelo final
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        print(f"\nGuardando modelo final en {final_dir}")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        # Train loss
        train_loss = None
        if train_result and train_result.metrics:
            train_loss = train_result.metrics.get("train_loss", None)
        
        # Summary
        summary = TrainSummary(
            model_name=args.model_name,
            device=device,
            used_fp16=use_fp16,
            used_bf16=use_bf16,
            train_examples=len(train_dataset),
            val_examples=len(val_dataset),
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            train_runtime_sec=float(train_runtime),
            train_loss=float(train_loss) if train_loss is not None else None,
            eval_loss=float(eval_loss) if eval_loss is not None else None,
            perplexity=float(perplexity) if perplexity is not None else None,
            output_dir=args.output_dir,
        )
        
        summary_path = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETADO")
        print("=" * 80)
        print(f"Modelo final: {final_dir}")
        print(f"Summary: {summary_path}")
        print(f"eval_loss: {summary.eval_loss}")
        print(f"perplexity: {summary.perplexity}")
        print(f"Duration: {train_runtime/60:.2f} min")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por usuario")
        
    except Exception as e:
        print(f"\nError durante entrenamiento: {e}")
        raise


if __name__ == "__main__":
    main()

