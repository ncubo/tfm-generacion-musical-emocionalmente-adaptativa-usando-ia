"""
Blueprint para endpoints relacionados con detección emocional.

Proporciona endpoints para capturar y analizar el estado emocional
del usuario a través de la webcam o desde una imagen enviada.
"""

from flask import Blueprint, jsonify, current_app, request
import numpy as np
import cv2
import threading
from ..core.utils.metrics import get_metrics
from ..core.emotion.deepface_detector import DeepFaceEmotionDetector
from ..core.va.mapper import emotion_to_va
from ..core.camera.webcam import WebcamCapture
from ..core.pipeline.emotion_pipeline import EmotionPipeline

emotion_bp = Blueprint('emotion', __name__)

# Lock global para thread-safety en lazy initialization
_pipeline_lock = threading.Lock()


def _get_or_create_pipeline():
    """
    Obtiene el pipeline emocional existente o lo crea (lazy initialization).
    
    Esta función garantiza que:
    - El pipeline solo se crea cuando realmente se necesita
    - No hay condiciones de carrera (thread-safe)
    - La webcam solo se activa la primera vez que se usa /emotion
    
    Returns:
        EmotionPipeline: Instancia del pipeline emocional
        
    Raises:
        RuntimeError: Si no se puede inicializar la webcam
    """
    # Si el pipeline ya existe, devolverlo
    pipeline = current_app.config.get('EMOTION_PIPELINE')
    if pipeline is not None:
        return pipeline
    
    # Thread-safe lazy initialization
    with _pipeline_lock:
        # Double-check: otro thread pudo haberlo creado mientras esperábamos
        pipeline = current_app.config.get('EMOTION_PIPELINE')
        if pipeline is not None:
            return pipeline
        
        # Crear pipeline por primera vez
        current_app.logger.info("[LAZY INIT] Inicializando webcam y pipeline...")
        
        try:
            # Inicializar webcam
            camera = WebcamCapture(camera_index=0)
            current_app.logger.info("[LAZY INIT] Webcam inicializada")
            
            # Inicializar detector de emociones
            detector = DeepFaceEmotionDetector()
            current_app.logger.info("[LAZY INIT] Detector de emociones inicializado")
            
            # Crear pipeline con estabilización temporal
            pipeline = EmotionPipeline(
                camera=camera,
                detector=detector,
                window_size=7,
                alpha=0.3,
                min_confidence=60.0
            )
            
            # Iniciar pipeline (activa la cámara)
            pipeline.start()
            current_app.logger.info("[LAZY INIT] Pipeline emocional iniciado")
            
            # Guardar en configuración de la app
            current_app.config['EMOTION_PIPELINE'] = pipeline
            
            return pipeline
            
        except Exception as e:
            current_app.logger.error(f"[LAZY INIT] Error al inicializar pipeline: {e}")
            raise RuntimeError(f"No se pudo inicializar la webcam: {e}")


@emotion_bp.route('/emotion', methods=['POST'])
def detect_emotion():
    """
    Captura el estado emocional actual desde la webcam del servidor.
    
    NOTA: Este endpoint usa lazy initialization. La webcam solo se activa
    la primera vez que se llama a este endpoint.
    
    Utiliza el pipeline emocional para:
    1. Capturar un frame de la webcam (servidor)
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
        - 500: Error interno del servidor o webcam no disponible
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
        if current_app.config.get('INCLUDE_METRICS', False):
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


@emotion_bp.route('/emotion-from-frame', methods=['POST'])
def detect_emotion_from_frame():
    """
    Detecta emoción facial desde una imagen enviada por el cliente.
    
    A diferencia de /emotion que usa la webcam del servidor, este endpoint
    recibe una imagen desde el frontend (capturada en el navegador) y la analiza.
    
    Request:
        - Content-Type: multipart/form-data
        - Campo: "image" (archivo jpeg/png)
    
    Returns:
        JSON con la emoción detectada y coordenadas VA
    
    Example:
        POST /emotion-from-frame
        Content-Type: multipart/form-data
        
        image: [archivo binario]
        
        Response:
        {
            "emotion": "happy",
            "valence": 0.70,
            "arousal": 0.60
        }
    
    Error cases:
        - 400: Falta el campo "image" o formato inválido
        - 500: Error interno del servidor
    """
    try:
        # Verificar que se envió un archivo
        if 'image' not in request.files:
            return jsonify({
                'error': 'Falta el campo "image"',
                'message': 'Debes enviar una imagen en el campo "image"'
            }), 400
        
        file = request.files['image']
        
        # Verificar que el archivo tiene nombre (no está vacío)
        if file.filename == '':
            return jsonify({
                'error': 'Archivo vacío',
                'message': 'El campo "image" no contiene un archivo válido'
            }), 400
        
        # Leer bytes del archivo
        file_bytes = file.read()
        
        if not file_bytes:
            return jsonify({
                'error': 'Archivo vacío',
                'message': 'El archivo enviado no contiene datos'
            }), 400
        
        # Convertir bytes a array numpy
        nparr = np.frombuffer(file_bytes, np.uint8)
        
        # Decodificar imagen con OpenCV (BGR)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                'error': 'Formato de imagen inválido',
                'message': 'No se pudo decodificar la imagen. Usa formato JPEG o PNG'
            }), 400
        
        # Obtener métricas
        metrics = get_metrics()
        
        # Crear detector (sin enforce_detection para manejar caso sin rostro)
        detector = DeepFaceEmotionDetector(enforce_detection=False)
        
        # Medir tiempo de detección
        with metrics.measure('emotion_from_frame', metadata={'endpoint': '/emotion-from-frame'}):
            # Detectar emoción en el frame
            emotion_result = detector.predict(frame)
        
        # Extraer emoción detectada
        emotion = emotion_result['emotion']
        face_detected = emotion_result.get('face_detected', False)
        
        # Si no se detectó rostro, devolver neutral (0, 0)
        if not face_detected:
            current_app.logger.info("No se detectó rostro en la imagen, devolviendo neutral")
            return jsonify({
                'emotion': 'neutral',
                'valence': 0.0,
                'arousal': 0.0,
                'face_detected': False
            }), 200
        
        # Mapear emoción a coordenadas VA
        valence, arousal = emotion_to_va(emotion)
        
        # Construir respuesta
        response = {
            'emotion': emotion,
            'valence': round(valence, 2),
            'arousal': round(arousal, 2),
            'face_detected': True
        }
        
        # Opcional: incluir tiempo de procesamiento
        if current_app.config.get('INCLUDE_METRICS', False):
            if metrics.measurements.get('emotion_from_frame'):
                last_duration = metrics.measurements['emotion_from_frame'][-1]
                response['processing_time_ms'] = round(last_duration * 1000, 2)
        
        return jsonify(response), 200
        
    except Exception as e:
        # Log del error para debugging
        current_app.logger.error(f"Error en /emotion-from-frame: {str(e)}", exc_info=True)
        
        # En producción, no exponer detalles internos
        error_message = str(e) if current_app.debug else 'Error interno del servidor'
        return jsonify({
            'error': 'Error al procesar la imagen',
            'message': error_message
        }), 500
