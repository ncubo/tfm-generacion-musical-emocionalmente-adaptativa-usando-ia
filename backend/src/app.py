"""
Aplicación principal del backend - TFM Generación Musical Emocional.

Este módulo implementa la API REST Flask que expone el pipeline de
reconocimiento emocional y generación musical MIDI baseline.

La API proporciona endpoints para:
- Detección de emociones desde webcam (lazy initialization)
- Detección de emociones desde imagen enviada
- Generación de música MIDI basada en emociones
- Monitoreo de salud del servicio

IMPORTANTE: La webcam del servidor NO se inicializa al arrancar.
Solo se activa cuando se necesita (/emotion o /generate-midi).
"""

import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from pathlib import Path
import logging

# Importar blueprints
from .routes import health_bp, emotion_bp, music_bp

__version__ = "0.3.0"  # Lazy initialization implementada

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config=None):
    """
    Factory function para crear y configurar la aplicación Flask.
    
    Este patrón Application Factory permite:
    - Crear múltiples instancias de la app (útil para testing)
    - Configurar la app de forma flexible
    - Inicializar recursos de forma controlada
    
    Args:
        config (dict, optional): Diccionario de configuración custom.
                                Si None, usa configuración por defecto.
    
    Returns:
        Flask: Aplicación Flask configurada y lista para usar
    
    Example:
        >>> app = create_app()
        >>> app.run(debug=True, port=5000)
    """
    # Crear instancia de Flask
    # app = Flask(__name__)
    app = Flask(__name__, static_folder='../../frontend/dist')
    
    # Configuración por defecto
    app.config['OUTPUT_DIR'] = Path(__file__).parent.parent / 'output'
    app.config['DEBUG'] = False
    app.config['HOST'] = '0.0.0.0'
    app.config['PORT'] = 5000
    
    # Aplicar configuración custom si se proporciona
    if config:
        app.config.update(config)
    
    # Asegurar que el directorio de salida existe
    Path(app.config['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
    
    # Habilitar CORS para permitir requests desde el frontend
    CORS(app)
    
    # Lazy initialization: NO inicializar webcam ni pipeline al arrancar
    # El pipeline se creará la primera vez que se llame al endpoint /emotion
    logger.info("Sistema iniciado (lazy initialization activa)")
    logger.info("La webcam se activará solo cuando se use /emotion")
    
    # Guardar None como placeholder para lazy initialization
    app.config['EMOTION_PIPELINE'] = None
    app.config['PIPELINE_LOCK'] = None  # Para thread-safety
    
    # Registrar blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(emotion_bp)
    app.register_blueprint(music_bp)
    
    logger.info("Blueprints registrados")
    
    # Hook para liberar recursos al cerrar la app
    @app.teardown_appcontext
    def cleanup(error=None):
        """Libera recursos cuando la aplicación se cierra."""
        pipeline = app.config.get('EMOTION_PIPELINE')
        if pipeline:
            try:
                pipeline.stop()
                logger.info("Pipeline emocional detenido")
            except Exception as e:
                logger.error(f"Error al detener pipeline: {e}")

    # Esto es para servir el frontend en render.com y usar solo un servicio
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(app.static_folder + '/' + path):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')
    
    return app


def main():
    """
    Función principal para ejecutar el servidor de desarrollo.
    
    Crea la aplicación Flask y la ejecuta en modo desarrollo.
    Para producción, usar un servidor WSGI como Gunicorn.
    
    Example:
        $ python backend/src/app.py
    """
    print("=" * 70)
    print(f"Backend TFM - Generación Musical Emocional v{__version__}")
    print("=" * 70)
    
    # Crear aplicación
    app = create_app()
    
    # Información sobre endpoints disponibles
    print("\nEndpoints disponibles:")
    print("  GET  /health                - Verificación de estado")
    print("  POST /emotion               - Detectar emoción actual (webcam servidor)")
    print("  POST /emotion-from-frame    - Detectar emoción desde imagen enviada")
    print("  POST /generate-midi         - Generar MIDI emocional")
    print("\n" + "=" * 70)
    print(f"Servidor iniciando en http://{app.config['HOST']}:{app.config['PORT']}")
    print("NOTA: Webcam en lazy mode - se activará solo al usar /emotion")
    print("=" * 70 + "\n")
    
    # Ejecutar servidor de desarrollo
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )


if __name__ == "__main__":
    main()
