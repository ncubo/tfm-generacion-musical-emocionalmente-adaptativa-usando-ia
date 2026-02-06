# Transformer Pretrained Model (SkyTNT/midi-model)

Este directorio contiene el modelo preentrenado para generación MIDI usando la arquitectura Transformer de SkyTNT.

## Descarga del Modelo

Para descargar el checkpoint preentrenado desde Hugging Face:

```bash
cd backend
python scripts/download_transformer_pretrained.py
```

Opciones:
- `--force`: Forzar descarga incluso si los archivos ya existen

## Verificación

Para verificar que el modelo se descargó correctamente y puede cargarse:

```bash
cd backend
python scripts/verify_transformer_pretrained.py
```

Si la verificación es exitosa, verás: `OK: Loaded pretrained weights`

Si falla, verás un mensaje de error detallado y deberás ejecutar el script de descarga.

## Estructura Esperada

Después de la descarga, esta carpeta debería contener:
- `model.pth` o similar (checkpoint del modelo)
- Archivos de configuración según el modelo de Hugging Face

## IMPORTANTE

⚠️ **NO COMMITEAR LOS CHECKPOINTS AL REPOSITORIO**

Los archivos de modelo están excluidos del control de versiones (`.gitignore`).
Cada desarrollador debe descargarlos localmente usando los scripts proporcionados.

## Referencia

- Modelo: [SkyTNT/midi-model](https://huggingface.co/skytnt/midi-model)
- Autor: SkyTNT
- Tipo: Transformer para generación MIDI incondicional
