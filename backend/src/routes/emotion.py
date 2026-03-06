"""
Blueprint para endpoints relacionados con detección emocional.

Proporciona endpoints para capturar y analizar el estado emocional
del usuario a través de la webcam o desde una imagen enviada.
"""

from flask import Blueprint, jsonify, current_app, request
import numpy as np
import cv2
import threading
import time
import hashlib
from ..core.utils.metrics import get_metrics
from ..core.emotion.deepface_detector import DeepFaceEmotionDetector
from ..core.va.mapper import emotion_to_va
from ..core.camera.webcam import WebcamCapture
from ..core.pipeline.emotion_pipeline import EmotionPipeline

emotion_bp = Blueprint('emotion', __name__)

# Lock global para thread-safety en lazy initialization (webcam pipeline)
_pipeline_lock = threading.Lock()

# Lock para el detector compartido de /emotion-from-frame
_frame_detector_lock = threading.Lock()

# Lock para el dict de pipelines por sesión
_frame_sessions_lock = threading.Lock()

# Tiempo máximo de inactividad de una sesión de frames (segundos)
_FRAME_SESSION_TIMEOUT = 60.0


def _get_session_key() -> str:
    """Genera una clave de sesión única basada en IP y User-Agent del cliente."""
    remote_addr = request.remote_addr or ''
    user_agent = request.headers.get('User-Agent', '')
    raw = f"{remote_addr}:{user_agent}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_shared_frame_detector() -> DeepFaceEmotionDetector:
    """
    Obtiene el detector compartido para /emotion-from-frame (lazy init).

    Reutiliza una única instancia por proceso en lugar de crear una nueva
    por request, evitando la sobrecarga de re-inicialización de modelos.
    """
    detector = current_app.config.get('FRAME_DETECTOR')
    if detector is not None:
        return detector
    with _frame_detector_lock:
        detector = current_app.config.get('FRAME_DETECTOR')
        if detector is not None:
            return detector
        detector = DeepFaceEmotionDetector(enforce_detection=False)
        current_app.config['FRAME_DETECTOR'] = detector
        current_app.logger.info("[LAZY INIT] Detector de frames inicializado")
        return detector


def _get_or_create_frame_pipeline(session_key: str) -> EmotionPipeline:
    """
    Obtiene o crea un pipeline de estabilización por sesión.

    Mantiene un dict en app.config['FRAME_PIPELINES'] con entradas:
        { session_key: {'pipeline': EmotionPipeline, 'last_access': float} }

    Las sesiones inactivas más de _FRAME_SESSION_TIMEOUT segundos se eliminan
    automáticamente en cada llamada para evitar fuga de memoria.
    """
    now = time.monotonic()

    with _frame_sessions_lock:
        sessions = current_app.config.setdefault('FRAME_PIPELINES', {})

        # Limpiar sesiones expiradas
        expired_keys = [
            k for k, v in sessions.items()
            if now - v['last_access'] > _FRAME_SESSION_TIMEOUT
        ]
        for k in expired_keys:
            del sessions[k]

        # Crear pipeline nuevo si no existe para esta sesión
        if session_key not in sessions:
            sessions[session_key] = {
                'pipeline': EmotionPipeline(window_size=7, alpha=0.3, min_confidence=60.0),
                'last_access': now
            }
        else:
            sessions[session_key]['last_access'] = now

        return sessions[session_key]['pipeline']


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
    3. Mapear a coordenadas Valencia-Activación
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

        # Obtener detector compartido (evita re-instanciación por request)
        detector = _get_shared_frame_detector()

        # Medir tiempo de detección
        with metrics.measure('emotion_from_frame', metadata={'endpoint': '/emotion-from-frame'}):
            emotion_result = detector.predict(frame)

        # Obtener/crear pipeline de estabilización para esta sesión
        session_key = _get_session_key()
        pipeline = _get_or_create_frame_pipeline(session_key)

        # Aplicar estabilización temporal (EMA + ventana de mayoría)
        stabilized = pipeline.process_detection(emotion_result)

        face_detected = emotion_result.get('face_detected', False)

        # Si no se detectó rostro, devolver neutral sin alterar los buffers
        if not face_detected:
            current_app.logger.info("No se detectó rostro en la imagen, devolviendo neutral")
            return jsonify({
                'emotion': 'neutral',
                'valence': 0.0,
                'arousal': 0.0,
                'face_detected': False
            }), 200

        # Construir respuesta con valores estabilizados
        response = {
            'emotion': stabilized['emotion'],
            'valence': round(stabilized['valence'], 2),
            'arousal': round(stabilized['arousal'], 2),
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
