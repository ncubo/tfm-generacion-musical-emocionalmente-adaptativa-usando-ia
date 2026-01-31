"""
Blueprint para endpoints relacionados con detección emocional.

Proporciona endpoints para capturar y analizar el estado emocional
del usuario a través de la webcam.
"""

from flask import Blueprint, jsonify, current_app

emotion_bp = Blueprint('emotion', __name__)


@emotion_bp.route('/emotion', methods=['POST'])
def detect_emotion():
    """
    Captura el estado emocional actual desde la webcam.
    
    Utiliza el pipeline emocional para:
    1. Capturar un frame de la webcam
    2. Detectar la emoción facial
    3. Mapear a coordenadas Valence-Arousal
    4. Aplicar suavizado temporal
    
    Returns:
        JSON con la emoción detectada y coordenadas VA
    
    Example:
        POST /emotion
        
        Response:
        {
            "emotion": "happy",
            "valence": 0.68,
            "arousal": 0.58
        }
    
    Error cases:
        - 500: Error interno del servidor
    """
    try:
        # Obtener el pipeline del contexto de la aplicación
        pipeline = current_app.config['EMOTION_PIPELINE']
        
        # Ejecutar un paso del pipeline
        result = pipeline.step()
        
        # Extraer solo los campos necesarios para la respuesta
        response = {
            'emotion': result['emotion'],
            'valence': round(result['valence'], 2),
            'arousal': round(result['arousal'], 2)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        # En caso de error, retornar un mensaje descriptivo
        # pero sin exponer detalles internos en producción
        return jsonify({
            'error': 'Error al detectar emoción',
            'message': str(e)
        }), 500
