"""
Blueprint para endpoints de salud y monitoreo de la API.

Proporciona endpoints para verificar el estado del servicio.
"""

from flask import Blueprint, jsonify

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de verificación de estado del servicio.
    
    Permite a clientes y sistemas de monitoreo verificar que
    la API está activa y respondiendo correctamente.
    
    Returns:
        JSON con status "ok" y código HTTP 200
    
    Example:
        GET /health
        
        Response:
        {
            "status": "ok"
        }
    """
    return jsonify({'status': 'ok'}), 200
