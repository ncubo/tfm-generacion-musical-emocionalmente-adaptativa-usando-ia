"""
Blueprint para endpoints relacionados con generación musical.

Proporciona endpoints para generar música MIDI basada en el estado
emocional del usuario.
"""

from flask import Blueprint, jsonify, current_app
from pathlib import Path
from datetime import datetime
from ..core.music.mapping import va_to_music_params
from ..core.music.baseline_rules import generate_midi_baseline

music_bp = Blueprint('music', __name__)


@music_bp.route('/generate-midi', methods=['POST'])
def generate_midi():
    """
    Genera un archivo MIDI baseline basado en el estado emocional actual.
    
    Workflow:
    1. Captura el estado emocional actual (webcam)
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
        - 500: Error al generar MIDI
    """
    try:
        # Obtener el pipeline del contexto de la aplicación
        pipeline = current_app.config['EMOTION_PIPELINE']
        
        # Capturar estado emocional actual
        emotion_result = pipeline.step()
        
        emotion = emotion_result['emotion']
        valence = emotion_result['valence']
        arousal = emotion_result['arousal']
        
        # Mapear coordenadas VA a parámetros musicales
        music_params = va_to_music_params(valence, arousal)
        
        # Construir path de salida con timestamp
        output_dir = Path(current_app.config['OUTPUT_DIR'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        midi_filename = f"emotion_{timestamp}.mid"
        midi_path = output_dir / midi_filename
        
        # Generar archivo MIDI
        generated_path = generate_midi_baseline(
            params=music_params,
            out_path=str(midi_path),
            length_bars=8,  # 8 compases por defecto
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
        
        return jsonify(response), 200
        
    except Exception as e:
        # Manejo de errores sin crashear el servidor
        return jsonify({
            'error': 'Error al generar MIDI',
            'message': str(e)
        }), 500
