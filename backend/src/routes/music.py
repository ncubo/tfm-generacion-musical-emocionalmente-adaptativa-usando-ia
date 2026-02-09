"""
Blueprint para endpoints relacionados con generación musical.

Proporciona endpoints para generar música MIDI basada en el estado
emocional del usuario.
"""

from flask import Blueprint, jsonify, request, current_app
from pathlib import Path
from datetime import datetime
import time
import os
from ..core.music.mapping import va_to_music_params
from ..core.music.engines.baseline import generate_midi_baseline
from ..core.music.engines.hf_maestro_remi import generate_midi_hf_maestro_remi
from ..core.utils.metrics import get_metrics

# Importar función de lazy initialization del blueprint de emotion
from .emotion import _get_or_create_pipeline

music_bp = Blueprint('music', __name__)


def cleanup_old_midi_files(output_dir: Path, max_age_minutes: int = 20):
    """
    Elimina archivos MIDI generados que tengan más de max_age_minutes minutos.
    
    Solo limpia archivos que coincidan con el patrón de generación automática:
    - emotion_*.mid
    
    Args:
        output_dir: Directorio donde están los archivos generados
        max_age_minutes: Antigüedad máxima en minutos (default: 20)
    """
    try:
        current_time = time.time()
        max_age_seconds = max_age_minutes * 60
        deleted_count = 0
        
        # Buscar archivos MIDI generados automáticamente
        for midi_file in output_dir.glob('emotion_*.mid'):
            file_age = current_time - os.path.getmtime(midi_file)
            
            if file_age > max_age_seconds:
                try:
                    midi_file.unlink()
                    deleted_count += 1
                    current_app.logger.debug(f"Eliminado archivo antiguo: {midi_file.name} (edad: {file_age/60:.1f} min)")
                except Exception as e:
                    current_app.logger.warning(f"Error al eliminar {midi_file.name}: {e}")
        
        if deleted_count > 0:
            current_app.logger.info(f"Limpieza automática: {deleted_count} archivo(s) MIDI eliminado(s)")
    
    except Exception as e:
        current_app.logger.error(f"Error en limpieza automática de MIDI: {e}")


@music_bp.route('/generate-midi', methods=['POST'])
def generate_midi():
    """
    Genera un archivo MIDI basado en el estado emocional actual.
    
    NOTA: Este endpoint usa lazy initialization. La webcam solo se activa
    la primera vez que se necesita detectar emoción desde la webcam del servidor.
    
    Query Parameters (opcionales):
        engine (str): Motor de generación - "baseline" (default) o "transformer_pretrained"
        seed (int): Semilla aleatoria para reproducibilidad
        length_bars (int): Número de compases a generar (default: MIDI_LENGTH_BARS config)
    
    Workflow:
    1. Captura el estado emocional actual (webcam servidor)
    2. Mapea emoción a coordenadas Valence-Arousal
    3. Convierte coordenadas VA a parámetros musicales
    4. Genera archivo MIDI usando el engine seleccionado:
       - baseline: Reglas heurísticas deterministas
       - transformer_pretrained: Modelo HF Maestro-REMI con condicionamiento indirecto
    5. Retorna metadata y path del archivo generado
    
    Returns:
        JSON con información de la generación y path al archivo MIDI
    
    Example:
        POST /generate-midi
        POST /generate-midi?engine=transformer_pretrained&seed=42&length_bars=16
        
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
            "midi_path": "/path/to/output/emotion_baseline_20260130_123045.mid",
            "engine": "baseline",
            "length_bars": 8,
            "seed": null
        }
    
    Error cases:
        - 500: Error al generar MIDI, webcam no disponible, o engine no disponible
    """
    try:
        # Obtener métricas
        metrics = get_metrics()
        
        # Parsear query parameters (con validación de tipo)
        engine = request.args.get('engine', 'baseline')
        
        # Validar seed
        try:
            seed_str = request.args.get('seed')
            seed = int(seed_str) if seed_str is not None else None
        except ValueError:
            return jsonify({
                'error': 'seed debe ser un entero',
                'message': f'seed inválido: {seed_str}'
            }), 400
        
        # Validar length_bars
        try:
            length_bars_str = request.args.get('length_bars')
            if length_bars_str is not None:
                length_bars = int(length_bars_str)
            else:
                length_bars = current_app.config.get('MIDI_LENGTH_BARS', 8)
        except ValueError:
            return jsonify({
                'error': 'length_bars debe ser un entero',
                'message': f'length_bars inválido: {length_bars_str}'
            }), 400
        
        # Validar length_bars (límite razonable)
        MAX_LENGTH_BARS = 64
        if length_bars < 1 or length_bars > MAX_LENGTH_BARS:
            return jsonify({
                'error': 'length_bars fuera de rango',
                'message': f'length_bars debe estar entre 1 y {MAX_LENGTH_BARS}, recibido: {length_bars}'
            }), 400
        
        # Validar engine
        valid_engines = ['baseline', 'transformer_pretrained']
        if engine not in valid_engines:
            return jsonify({
                'error': 'Engine no válido',
                'message': f'engine debe ser uno de: {valid_engines}'
            }), 400
        
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
        with metrics.measure('total_pipeline', metadata={
            'endpoint': '/generate-midi',
            'engine': engine,
            'length_bars': length_bars,
            'seed': seed
        }):
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
            
            # Construir path de salida con timestamp y engine
            output_dir = Path(current_app.config['OUTPUT_DIR'])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            midi_filename = f"emotion_{engine}_{timestamp}.mid"
            midi_path = output_dir / midi_filename
            
            # Limpieza automática de archivos antiguos (más de 20 minutos)
            cleanup_old_midi_files(output_dir, max_age_minutes=20)
            
            # Medir tiempo de generación MIDI según engine
            with metrics.measure('midi_generation', metadata={
                'emotion': emotion,
                'valence': valence,
                'arousal': arousal,
                'engine': engine,
                'length_bars': length_bars,
                'seed': seed
            }) as timing:
                # Seleccionar generador según engine
                if engine == 'baseline':
                    generated_path = generate_midi_baseline(
                        params=music_params,
                        out_path=str(midi_path),
                        length_bars=length_bars,
                        seed=seed
                    )
                
                elif engine == 'transformer_pretrained':
                    try:
                        generated_path = generate_midi_hf_maestro_remi(
                            params=music_params,
                            out_path=str(midi_path),
                            length_bars=length_bars,
                            seed=seed
                        )
                    except (RuntimeError, NotImplementedError) as e:
                        # Si falla el modelo HF, fallback a baseline con warning
                        current_app.logger.warning(
                            f"Engine transformer_pretrained falló: {e}. "
                            f"Fallback a baseline."
                        )
                        # Regenerar con baseline
                        midi_filename_fallback = f"emotion_baseline_fallback_{timestamp}.mid"
                        midi_path_fallback = output_dir / midi_filename_fallback
                        generated_path = generate_midi_baseline(
                            params=music_params,
                            out_path=str(midi_path_fallback),
                            length_bars=length_bars,
                            seed=seed
                        )
                        engine = 'baseline'  # Actualizar engine en respuesta
                
                else:
                    # No debería llegar aquí por validación previa
                    raise ValueError(f"Engine no implementado: {engine}")
        
        # Leer archivo MIDI generado
        with open(generated_path, 'rb') as f:
            midi_data = f.read()
        
        # Crear respuesta con el archivo MIDI
        from flask import make_response
        response = make_response(midi_data)
        response.headers['Content-Type'] = 'audio/midi'
        response.headers['Content-Disposition'] = f'attachment; filename="{Path(generated_path).name}"'
        
        # Agregar metadata como headers personalizados
        response.headers['X-Engine'] = engine
        response.headers['X-Seed'] = str(seed) if seed is not None else '0'
        response.headers['X-Length-Bars'] = str(length_bars)
        response.headers['X-Emotion'] = emotion
        response.headers['X-Valence'] = str(round(valence, 2))
        response.headers['X-Arousal'] = str(round(arousal, 2))
        
        # CORS headers para permitir acceso desde frontend
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Expose-Headers'] = 'X-Engine,X-Seed,X-Length-Bars,X-Emotion,X-Valence,X-Arousal'
        
        return response, 200
        
    except Exception as e:
        # Log del error para debugging
        current_app.logger.error(f"Error en /generate-midi: {str(e)}", exc_info=True)
        
        # En producción, no exponer detalles internos
        error_message = str(e) if current_app.debug else 'Error interno del servidor'
        return jsonify({
            'error': 'Error al generar MIDI',
            'message': error_message
        }), 500
