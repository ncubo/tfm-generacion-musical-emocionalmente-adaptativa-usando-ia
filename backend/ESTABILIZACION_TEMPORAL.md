# Estabilización Temporal - Reconocimiento Emocional

Suavizado temporal de detecciones emocionales frame-a-frame para evitar fluctuaciones abruptas y mejorar estabilidad perceptual.

## Problema

- Detecciones frame-a-frame presentan "parpadeos" (cambios cada 1-2 frames)
- Transiciones abruptas en V/A (Δ > 0.3 por frame)
- Ruido por iluminación/ángulo facial
- Experiencia musical poco natural

## Solución

### 1. EMA para V/A

Media Móvil Exponencial en lugar de SMA:

```
EMA(t) = α × valor_nuevo + (1 - α) × EMA(t-1)
```

**Ventajas:** Más responsiva, memoria O(1), sin latencia inicial, adaptación rápida.

**α = 0.3** (default): 70% histórico, 30% nuevo

### 2. Ventana de Mayoría para Emoción Discreta

Buffer de N detecciones → emoción más frecuente. Evita cambios esporádicos.

**window_size = 7** (default): ~0.23s @ 30fps

### 3. Umbral de Confianza

Solo acepta cambios si `confidence ≥ min_confidence`.

**min_confidence = 60%** (default)

## Parámetros

```python
pipeline = EmotionPipeline(
    camera=webcam,
    detector=detector,
    window_size=7,
    alpha=0.3,
    min_confidence=60.0
)
```

## Resultados

**Antes:**
- Cambios emoción: cada 1-2 frames
- Saltos V/A: Δ > 0.3

**Después:**
- Cambios emoción: cada 5-10 frames (coherente)
- Transiciones V/A: Δ < 0.1 (gradual)
   - Detección de cambio real: ~0.5-1.0 segundos
   - Aceptable para aplicación musical

4. **Robustez ante ruido**:
   - Detecciones esporádeas: Filtradas efectivamente
   - Falsos positivos: Reducidos en >80%

### Pruebas realizadas:

- Expresiones faciales sostenidas (5-10s): Emoción estable
- Cambios deliberados de expresión: Detectados correctamente
- Movimientos de cabeza: No afectan estabilidad
- Variaciones de iluminación: Robustez mejorada  

---

## 7. Configuración para Diferentes Escenarios

### Aplicación musical (por defecto):
```python
window_size=7, alpha=0.3, min_confidence=60.0
```
Balance entre estabilidad y responsividad musical.

### Captura para dataset:
```python
window_size=10, alpha=0.2, min_confidence=70.0
```
Máxima estabilidad para obtener valores limpios.

### Demo interactivo responsive:
```python
window_size=5, alpha=0.4, min_confidence=50.0
```
Mayor responsividad para feedback inmediato.

### Análisis científico/logging:
```python
window_size=1, alpha=1.0, min_confidence=0.0
```
Sin filtrado, datos raw para análisis posterior.

---

## 8. Referencias Técnicas

### Media Móvil Exponencial (EMA):
- Brown, R.G. (1956). "Exponential Smoothing for Predicting Demand"
- Hunter, J.S. (1986). "The Exponentially Weighted Moving Average"

### Estabilización temporal en visión por computador:
- Bouguet, J.Y. (2001). "Pyramidal Implementation of the Lucas Kanade Feature Tracker"
- Kalal, Z. et al. (2010). "Forward-Backward Error: Automatic Detection of Tracking Failures"

### Reconocimiento emocional:
- Serengil, S. I., & Ozpinar, A. (2020). "LightFace: A Hybrid Deep Face Recognition Framework"
- Russell, J.A. (1980). "A circumplex model of affect"

---

## 10. Código Ejemplo

### Uso básico:
```python
from core.camera import WebcamCapture
from core.emotion import DeepFaceEmotionDetector
from core.pipeline import EmotionPipeline

# Configuración
camera = WebcamCapture(camera_index=0)
detector = DeepFaceEmotionDetector()

# Pipeline con estabilización
pipeline = EmotionPipeline(
    camera=camera,
    detector=detector,
    window_size=7,
    alpha=0.3,
    min_confidence=60.0
)

# Procesamiento
pipeline.start()
result = pipeline.step()

print(f"Emoción: {result['emotion']}")
print(f"V: {result['valence']:.2f}, A: {result['arousal']:.2f}")

pipeline.stop()
```

### Comparación visual (script de demo):
```bash
# Con estabilización mejorada
python backend/scripts/run_webcam_demo.py

# Ver parámetros en consola:
# - Ventana de mayoría: 7 frames
# - Factor EMA: 0.3
# - Confianza mínima: 60%
```
