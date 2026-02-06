#!/usr/bin/env python3
"""
Script para descargar el checkpoint del modelo SkyTNT/midi-model desde Hugging Face.

Este script descarga el modelo preentrenado y lo guarda en la carpeta
models/transformer_pretrained/ para su uso posterior.

Uso:
    python scripts/download_transformer_pretrained.py [--force]

Opciones:
    --force: Forzar descarga incluso si los archivos ya existen
"""

import argparse
import sys
from pathlib import Path

def download_model(force: bool = False):
    """
    Descarga el modelo preentrenado desde Hugging Face.
    
    Args:
        force: Si True, descarga incluso si los archivos ya existen
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("❌ Error: huggingface_hub no está instalado")
        print("   Ejecuta: pip install huggingface_hub")
        sys.exit(1)
    
    # Configuración
    repo_id = "skytnt/midi-model"
    model_dir = Path(__file__).parent.parent / "models" / "transformer_pretrained"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Descargando modelo desde: {repo_id}")
    print(f"📂 Destino: {model_dir}")
    print()
    
    # Listar archivos disponibles en el repositorio
    print("🔍 Listando archivos disponibles en el repositorio...")
    try:
        files_in_repo = list_repo_files(repo_id)
        print(f"✓ Encontrados {len(files_in_repo)} archivos")
        
        # Filtrar archivos de modelo (pth, pt, ckpt, bin, safetensors)
        model_files = [f for f in files_in_repo if f.endswith(('.pth', '.pt', '.ckpt', '.bin', '.safetensors'))]
        
        if not model_files:
            print("❌ No se encontraron archivos de modelo en el repositorio")
            print(f"   Archivos disponibles: {files_in_repo[:10]}")
            sys.exit(1)
        
        print(f"📦 Archivos de modelo encontrados:")
        for f in model_files:
            print(f"   - {f}")
        print()
        
        files_to_download = model_files
        
    except Exception as e:
        print(f"⚠️  No se pudo listar archivos: {str(e)}")
        print("   Intentando con nombres comunes...")
        files_to_download = ["model.pth"]
    
    downloaded_files = []
    
    for filename in files_to_download:
        target_path = model_dir / filename
        
        # Verificar si ya existe
        if target_path.exists() and not force:
            print(f"✓ {filename} ya existe (usa --force para reemplazar)")
            continue
        
        try:
            print(f"⬇️  Descargando {filename}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=None,  # Usar cache por defecto de HF
                local_dir=str(model_dir),
                local_dir_use_symlinks=False  # Copiar archivos, no symlinks
            )
            print(f"✅ {filename} descargado correctamente")
            downloaded_files.append(filename)
            
        except Exception as e:
            print(f"❌ Error al descargar {filename}: {str(e)}")
            sys.exit(1)
    
    print()
    print("=" * 60)
    if downloaded_files:
        print(f"✅ Descarga completada. {len(downloaded_files)} archivo(s) descargado(s):")
        for f in downloaded_files:
            print(f"   - {f}")
    else:
        print("✅ Todos los archivos ya estaban presentes")
    
    print()
    print("📋 Siguiente paso: Verificar el modelo")
    print("   python scripts/verify_transformer_pretrained.py")


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="Descargar modelo SkyTNT/midi-model desde Hugging Face"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar descarga incluso si los archivos ya existen"
    )
    
    args = parser.parse_args()
    download_model(force=args.force)


if __name__ == "__main__":
    main()
