"""
Blueprint para endpoints relacionados con generación musical.

Proporciona endpoints para generar música MIDI basada en el estado
emocional del usuario usando diferentes engines de generación.
"""

import base64
from flask import Blueprint, jsonify, current_app, request
from pathlib import Path
from datetime import datetime
from ..core.music.mapping import va_to_music_params
from ..core.music.baseline_rules import generate_midi_baseline
from ..core.music.engines import get_engine, list_engines
from ..core.utils.metrics import get_metrics

# Importar función de lazy initialization del blueprint de emotion
from .emotion import _get_or_create_pipeline

music_bp = Blueprint('music', __name__)


@music_bp.route('/generate-midi', methods=['POST'])
def generate_midi():
    """
    Genera un archivo MIDI usando el engine seleccionado.
    
    Este endpoint unificado soporta múltiples engines de generación:
    - baseline: Reglas deterministas (rápido, predecible)
    - transformer_pretrained: Modelo SkyTNT preentrenado (requiere checkpoint)
    - transformer_finetuned: Modelo fine-tuned (no disponible aún, retorna 501)
    
    Body (JSON):
        {
            "engine": "baseline" | "transformer_pretrained" | "transformer_finetuned",
            "valence": float (opcional si se usa webcam),
            "arousal": float (opcional si se usa webcam),
            "seed": int (opcional, para reproducibilidad)
        }
    
    Si no se proveen valence/arousal, se capturan desde la webcam del servidor.
    
    Returns:
        JSON con información de la generación:
        {
            "engine": str,
            "valence": float,
            "arousal": float,
            "generation_params": dict,
            "midi_path": str
        }
    
    Error cases:
        - 400: Engine inválido o parámetros faltantes
        - 500: Error al generar o webcam no disponible
        - 501: Engine no disponible (transformer_finetuned)
    """
    try:
        # Obtener métricas
        metrics = get_metrics()
        
        # Parsear body (JSON opcional)
        data = request.get_json() or {}
        
        # Obtener engine (default: baseline)
        engine_name = data.get('engine', 'baseline')
        
        # Validar engine
        try:
            engine = get_engine(engine_name)
        except ValueError as e:
            return jsonify({
                'error': 'Engine inválido',
                'message': str(e)
            }), 400
        
        # Verificar si el engine es transformer_finetuned (no disponible aún)
        if engine_name == 'transformer_finetuned':
            return jsonify({
                'error': 'Motor aún no disponible',
                'message': 'El engine transformer_finetuned aún no está implementado. Usa "baseline" o "transformer_pretrained".'
            }), 501
        
        # Obtener valence/arousal
        valence = data.get('valence')
        arousal = data.get('arousal')
        
        # Si no se proveen V/A, capturar desde webcam
        if valence is None or arousal is None:
            current_app.logger.info("V/A no provistos, capturando desde webcam...")
            
            try:
                pipeline = _get_or_create_pipeline()
            except RuntimeError as e:
                return jsonify({
                    'error': 'No se pudo acceder a la webcam del servidor',
                    'message': str(e)
                }), 500
            
            with metrics.measure('emotion_detection_for_midi'):
                emotion_result = pipeline.step()
            
            emotion = emotion_result['emotion']
            valence = emotion_result['valence']
            arousal = emotion_result['arousal']
            
            current_app.logger.info(f"Emoción detectada: {emotion} (V={valence:.2f}, A={arousal:.2f})")
        else:
            emotion = None  # No detectada desde webcam
        
        # Validar V/A
        if valence is None or arousal is None:
            return jsonify({
                'error': 'Parámetros faltantes',
                'message': 'Se requieren "valence" y "arousal" en el body o detección desde webcam'
            }), 400
        
        # Obtener seed (opcional)
        seed = data.get('seed')
        
        # Construir path de salida con timestamp y engine
        output_dir = Path(current_app.config['OUTPUT_DIR'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        midi_filename = f"{engine_name}_{timestamp}.mid"
        midi_path = output_dir / midi_filename
        
        # Medir tiempo de generación
        with metrics.measure('midi_generation', metadata={
            'engine': engine_name,
            'valence': valence,
            'arousal': arousal
        }) as timing:
            try:
                # Generar MIDI usando el engine seleccionado
                result = engine.generate(
                    valence=valence,
                    arousal=arousal,
                    out_path=str(midi_path),
                    seed=seed
                )
            except FileNotFoundError as e:
                # Checkpoint faltante (transformer_pretrained)
                current_app.logger.error(f"Checkpoint faltante: {str(e)}")
                return jsonify({
                    'error': 'Checkpoint del modelo no encontrado',
                    'message': str(e),
                    'instructions': 'Ejecuta: python backend/scripts/download_transformer_pretrained.py && python backend/scripts/verify_transformer_pretrained.py'
                }), 500
            except Exception as e:
                current_app.logger.error(f"Error en generación: {str(e)}", exc_info=True)
                raise
        
        # Leer el archivo MIDI y codificarlo en base64
        with open(midi_path, 'rb') as f:
            midi_bytes = f.read()
            midi_base64 = base64.b64encode(midi_bytes).decode('utf-8')
        
        # Preparar respuesta
        response = {
            'engine': engine_name,
            'valence': round(valence, 2),
            'arousal': round(arousal, 2),
            'generation_params': result.get('generation_params', {}),
            'midi_path': result['midi_path'],
            'midi_data': midi_base64  # Contenido MIDI en base64 para reproducción
        }
        
        # Incluir emoción si fue detectada
        if emotion:
            response['emotion'] = emotion
        
        # Opcional: incluir tiempo de procesamiento
        if current_app.config.get('INCLUDE_METRICS', False):
            response['processing_time_ms'] = round(timing['duration'] * 1000, 2)
        
        return jsonify(response), 200
        
    except Exception as e:
        # Log del error para debugging
        current_app.logger.error(f"Error en /generate-midi: {str(e)}", exc_info=True)
        
        # En producción, no exponer detalles internos
        error_message = str(e) if current_app.debug else 'Error interno del servidor'
        return jsonify({
            'error': 'Error al generar MIDI',
            'message': error_message
        }), 500


@music_bp.route('/engines', methods=['GET'])
def get_engines():
    """
    Lista todos los engines de generación disponibles.
    
    Returns:
        JSON con lista de engines:
        {
            "engines": [
                {
                    "name": "baseline",
                    "description": "...",
                    "available": true
                },
                ...
            ]
        }
    """
    try:
        engines = list_engines()
        return jsonify({'engines': engines}), 200
    except Exception as e:
        current_app.logger.error(f"Error en /engines: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Error al listar engines',
            'message': str(e)
        }), 500


# Mantener endpoint legacy para compatibilidad (deprecado)
@music_bp.route('/generate-midi-legacy', methods=['POST'])
def generate_midi_legacy():
    """
    Genera un archivo MIDI baseline basado en el estado emocional actual.
    
    NOTA: Este endpoint usa lazy initialization. La webcam solo se activa
    la primera vez que se necesita detectar emoción desde la webcam del servidor.
    
    Workflow:
    1. Captura el estado emocional actual (webcam servidor)
    2. Mapea emoción a coordenadas Valence-Arousal
    3. Convierte coordenadas VA a parámetros musicales
    4. Genera archivo MIDI usando reglas baseline
    5. Retorna metadata y path del archivo generado
    
    Returns:
        JSON con información de la generación y path al archivo MIDI
    
    Example:
        POST /generate-midi
        
        Response:
        {
            "emotion": "happy",
            "valence": 0.68,
            "arousal": 0.58,
            "params": {
                "tempo_bpm": 132,
                "mode": "major",
                "density": 0.74,
                "pitch_low": 64,
                "pitch_high": 76,
                "rhythm_complexity": 0.74,
                "velocity_mean": 92,
                "velocity_spread": 22
            },
            "midi_path": "/path/to/output/emotion_20260130_123045.mid"
        }
    
    Error cases:
        - 500: Error al generar MIDI o webcam no disponible
    """
    try:
        # Obtener métricas
        metrics = get_metrics()
        
        # Lazy initialization: obtener o crear pipeline
        try:
            pipeline = _get_or_create_pipeline()
        except RuntimeError as e:
            # Error al inicializar webcam
            return jsonify({
                'error': 'No se pudo acceder a la webcam del servidor',
                'message': str(e)
            }), 500
        
        # Medir tiempo total del pipeline completo
        with metrics.measure('total_pipeline', metadata={'endpoint': '/generate-midi'}):
            # Medir tiempo de detección emocional
            with metrics.measure('emotion_detection_for_midi'):
                # Capturar estado emocional actual
                emotion_result = pipeline.step()
            
            emotion = emotion_result['emotion']
            valence = emotion_result['valence']
            arousal = emotion_result['arousal']
            
            # Medir tiempo de mapeo VA a parámetros musicales
            with metrics.measure('va_to_music_mapping'):
                # Mapear coordenadas VA a parámetros musicales
                music_params = va_to_music_params(valence, arousal)
            
            # Construir path de salida con timestamp
            output_dir = Path(current_app.config['OUTPUT_DIR'])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            midi_filename = f"emotion_{timestamp}.mid"
            midi_path = output_dir / midi_filename
            
            # Medir tiempo de generación MIDI
            with metrics.measure('midi_generation', metadata={'emotion': emotion, 'valence': valence, 'arousal': arousal}) as timing:
                # Generar archivo MIDI
                length_bars = current_app.config.get('MIDI_LENGTH_BARS', 8)
                generated_path = generate_midi_baseline(
                    params=music_params,
                    out_path=str(midi_path),
                    length_bars=length_bars,
                    seed=None  # Generación aleatoria
                )
        
        # Preparar respuesta
        response = {
            'emotion': emotion,
            'valence': round(valence, 2),
            'arousal': round(arousal, 2),
            'params': music_params,
            'midi_path': generated_path
        }
        
        # Opcional: incluir tiempo de procesamiento en la respuesta
        if current_app.config.get('INCLUDE_METRICS', False):
            response['processing_time_ms'] = round(timing['duration'] * 1000, 2)
        
        return jsonify(response), 200
        
    except Exception as e:
        # Log del error para debugging
        current_app.logger.error(f"Error en /generate-midi: {str(e)}", exc_info=True)
        
        # En producción, no exponer detalles internos
        error_message = str(e) if current_app.debug else 'Error interno del servidor'
        return jsonify({
            'error': 'Error al generar MIDI',
            'message': error_message
        }), 500
