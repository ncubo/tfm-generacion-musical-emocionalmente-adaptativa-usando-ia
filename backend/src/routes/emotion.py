"""
Blueprint para endpoints relacionados con detección emocional.

Proporciona endpoints para capturar y analizar el estado emocional
del usuario a través de la webcam.
"""

from flask import Blueprint, jsonify, current_app
from ..core.utils.metrics import get_metrics

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
        # Obtener métricas
        metrics = get_metrics()
        
        # Obtener el pipeline del contexto de la aplicación
        pipeline = current_app.config['EMOTION_PIPELINE']
        
        # Medir tiempo de ejecución del pipeline emocional
        with metrics.measure('emotion_detection', metadata={'endpoint': '/emotion'}):
            # Ejecutar un paso del pipeline
            result = pipeline.step()
        
        # Extraer solo los campos necesarios para la respuesta
        response = {
            'emotion': result['emotion'],
            'valence': round(result['valence'], 2),
            'arousal': round(result['arousal'], 2)
        }
        
        # Opcional: incluir tiempo de procesamiento en la respuesta (útil para debugging)
        # Nota: requiere acceso a las métricas almacenadas
        if current_app.config.get('INCLUDE_METRICS', False):
            # Obtener la última medición de emotion_detection
            if metrics.measurements.get('emotion_detection'):
                last_duration = metrics.measurements['emotion_detection'][-1]
                response['processing_time_ms'] = round(last_duration * 1000, 2)
        
        return jsonify(response), 200
        
    except Exception as e:
        # Log del error para debugging
        current_app.logger.error(f"Error en /emotion: {str(e)}", exc_info=True)
        
        # En producción, no exponer detalles internos
        error_message = str(e) if current_app.debug else 'Error interno del servidor'
        return jsonify({
            'error': 'Error al detectar emoción',
            'message': error_message
        }), 500
