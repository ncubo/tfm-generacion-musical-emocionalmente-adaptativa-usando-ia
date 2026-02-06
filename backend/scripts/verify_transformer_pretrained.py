#!/usr/bin/env python3
"""
Script para verificar que el checkpoint del modelo SkyTNT/midi-model
se descargó correctamente y puede cargarse.

Este script valida:
1. Existencia de archivos requeridos
2. Carga exitosa del modelo con los pesos preentrenados

Uso:
    python scripts/verify_transformer_pretrained.py

Exit codes:
    0: Verificación exitosa
    1: Error de verificación
"""

import sys
from pathlib import Path


def verify_model():
    """
    Verifica que el modelo preentrenado esté disponible y se pueda cargar.
    
    Returns:
        bool: True si la verificación es exitosa, False en caso contrario
    """
    model_dir = Path(__file__).parent.parent / "models" / "transformer_pretrained"
    
    print("Verificando modelo SkyTNT/midi-model...")
    print(f"Directorio: {model_dir}")
    print()
    
    # 1. Verificar existencia del directorio
    if not model_dir.exists():
        print("FAIL: El directorio del modelo no existe")
        print(f"Ejecuta: python scripts/download_transformer_pretrained.py")
        return False
    
    # 2. Verificar existencia de archivos requeridos
    required_files = [
        "model.safetensors",
        "pytorch_model.bin"
    ]
    
    found_files = []
    for filename in required_files:
        file_path = model_dir / filename
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} existe ({file_size_mb:.2f} MB)")
            found_files.append(filename)
    
    if not found_files:
        print()
        print(f"FAIL: No se encontró ningún archivo de modelo")
        print(f"Ejecuta: python scripts/download_transformer_pretrained.py")
        return False
    
    # Usar el primer archivo encontrado para verificación
    checkpoint_file = found_files[0]
    
    print()
    
    # 3. Intentar cargar el modelo
    print("⏳ Intentando cargar el modelo con PyTorch...")
    
    try:
        import torch
    except ImportError:
        print("FAIL: PyTorch no está instalado")
        print("Ejecuta: pip install torch")
        return False
    
    try:
        checkpoint_path = model_dir / checkpoint_file
        print(f"Usando archivo: {checkpoint_file}")
        
        # Cargar el checkpoint (solo verificar que se puede leer, no instanciar modelo completo)
        if checkpoint_file.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                checkpoint = load_file(str(checkpoint_path))
                print(f"Checkpoint cargado con safetensors")
            except ImportError:
                print("safetensors no disponible, instalando...")
                import subprocess
                subprocess.check_call(['pip', 'install', 'safetensors'])
                from safetensors.torch import load_file
                checkpoint = load_file(str(checkpoint_path))
        else:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        
        # Verificar que contiene lo esperado
        if isinstance(checkpoint, dict):
            # Usualmente los checkpoints tienen keys como 'model', 'state_dict', etc.
            print(f"Checkpoint cargado correctamente")
            print(f"Keys en checkpoint: {list(checkpoint.keys())[:5]}...")  # Mostrar primeras 5 keys
            
            # Contar parámetros aproximadamente
            if 'state_dict' in checkpoint:
                num_params = sum(p.numel() for p in checkpoint['state_dict'].values() if isinstance(p, torch.Tensor))
                print(f"  Parámetros aprox: {num_params:,}")
            elif isinstance(checkpoint, dict):
                # El checkpoint podría ser directamente el state_dict
                num_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
                if num_params > 0:
                    print(f"  Parámetros aprox: {num_params:,}")
        else:
            print(f"Checkpoint cargado (tipo: {type(checkpoint).__name__})")
        
    except Exception as e:
        print(f"FAIL: Error al cargar el checkpoint")
        print(f"   Error: {str(e)}")
        return False
    
    print()
    print("=" * 60)
    print("OK: Loaded pretrained weights")
    print("=" * 60)
    print()
    print("El modelo está listo para usar con el engine transformer_pretrained")
    
    return True


def main():
    """Punto de entrada principal."""
    success = verify_model()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
