#!/usr/bin/env python3
"""
VisionFlow v2 Web Server
Flask + SocketIO server for real-time detection streaming to frontend
"""

# Gevent is monkey-patched in main_with_web.py before any imports
import logging
import time
from collections import deque
import numpy as np
import cv2
import base64
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from threading import Lock

from src.app.core.events_manager.events_poster import EventPoster

# Detect if gevent was monkey-patched
try:
    import gevent
    SOCKETIO_ASYNC_MODE = 'gevent'
except ImportError:
    SOCKETIO_ASYNC_MODE = 'threading'

logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'visionflow-v2-secret-key'
CORS(app, resources={r"/*": {"origins": "*"}})

# SocketIO initialization with CORS
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode=SOCKETIO_ASYNC_MODE,
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

if SOCKETIO_ASYNC_MODE != 'gevent':
    logger.warning(
        "Gevent not installed; falling back to threading mode. "
        "WebSocket transport will be limited to long-polling."
    )

# Shared state for current frame and detections
current_frame = None
current_detections = None
frame_lock = Lock()

# Connected clients tracking
connected_clients = set()
clients_lock = Lock()

# Stats tracking
stats = {
    'total_frames_processed': 0,
    'total_detections': 0,
    'connected_clients': 0,
    'server_start_time': time.time(),
    'frames_received_from_frontend': 0
}
stats_lock = Lock()

# Frame processor reference (will be set by main app)
frame_processor = None
processor_lock = Lock()

# Detection filter state (for dynamic class filtering)
active_class_filter = None  # None = all classes, or list like ['person', 'cell phone']
filter_lock = Lock()

# Frontend clip recorder state
frontend_clip_recorder = None
processing_frame_flag = {'processing': False}  # Dict to allow modification in nested scope
processing_frame_lock = Lock()


def set_frame_processor(processor):
    """
    Set the object detector processor for processing frames from frontend

    Args:
        processor: ObjectDetector instance
    """
    global frame_processor, active_class_filter
    with processor_lock:
        frame_processor = processor

    # Sync active_class_filter with processor's initial filter
    with filter_lock:
        if hasattr(processor, 'filter_classes'):
            active_class_filter = processor.filter_classes
            logger.info(f"Synced initial class filter: {active_class_filter}")

    logger.info("Frame processor registered for frontend frame processing")


class FrontendClipRecorder:
    """Collect frames from the frontend stream and generate clips when detections arrive."""

    def __init__(self, clip_config: dict):
        self.config = clip_config or {}
        # Emit SocketIO event when a clip is ready
        def emit_clip_created(payload):
            try:
                # Emit in a background task to avoid context issues
                socketio.start_background_task(socketio.emit, 'clip_created', payload)
                socketio.start_background_task(socketio.emit, 'clip_event', payload)  # alias event name
                logger.info(f"Emitted clip_created event: {payload.get('frame_id')} ({payload.get('status')})")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to emit clip_created: {exc}")

        self.poster = EventPoster(self.config, on_clip_created=emit_clip_created)
        clip_duration = float(self.config.get('clip_duration', 30))
        fps = float(self.config.get('fps', 25.0))
        self.buffer_max_size = int(clip_duration * fps) or 1
        self.buffer = deque(maxlen=self.buffer_max_size)
        self.min_clip_cooldown = float(self.config.get('min_clip_cooldown', clip_duration))
        self.last_clip_time = 0.0
        self.detection_triggered = False
        self.frames_after_detection = 0
        self.frames_to_collect_after = self.buffer_max_size // 2  # Collect half the buffer after detection

    def handle_frame(self, frame: np.ndarray, detections):
        """
        Store frame in buffer; if detections are present, start collecting frames.
        Generate clip only after collecting enough frames after the detection.
        """
        if frame is None:
            return

        # Keep a copy to avoid mutation
        self.buffer.append(frame.copy())

        # If we haven't detected an event yet, check for detections
        if not self.detection_triggered and detections:
            now = time.time()
            # Check cooldown
            if (now - self.last_clip_time) >= self.min_clip_cooldown:
                self.detection_triggered = True
                self.frames_after_detection = 0
                logger.info(f"Detection triggered! Will collect {self.frames_to_collect_after} more frames...")
                return

        # If we're collecting frames after detection
        if self.detection_triggered:
            self.frames_after_detection += 1

            # Once we've collected enough frames after detection, generate the clip
            if self.frames_after_detection >= self.frames_to_collect_after:
                logger.info(f"Collected {self.frames_after_detection} frames, generating clip with {len(self.buffer)} total frames")
                self.last_clip_time = time.time()
                self.detection_triggered = False
                self.frames_after_detection = 0

                try:
                    # Use the last detection data stored
                    self.poster.write_clip_from_numpy_frames(list(self.buffer), detections, source="frontend")
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Failed to create frontend clip: {exc}")


def decode_base64_image(base64_string):
    """
    Decode base64 image string to numpy array

    Args:
        base64_string: Base64 encoded image (data:image/jpeg;base64,...)

    Returns:
        numpy array of image in BGR format
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_string)

        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return img
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None


def update_frame_and_detections(frame_data, metadata):
    """
    Called by VisionFlow pipeline to update latest frame and detections

    Args:
        frame_data: numpy array of the frame
        metadata: dict containing processor results
    """
    global current_frame, current_detections

    # Draw detections on a copy so MJPEG stream shows bounding boxes
    processor_data = metadata.get('processor', {})
    frame_to_store = frame_data
    try:
        tags = processor_data.get('tags', [])
        if tags:
            frame_to_store = frame_data.copy()
            for tag in tags:
                bbox = tag.get('bbox', {})
                x, y = int(bbox.get('x', 0)), int(bbox.get('y', 0))
                w, h = int(bbox.get('width', 0)), int(bbox.get('height', 0))
                cv2.rectangle(frame_to_store, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{tag.get('class_name', 'obj')} {tag.get('confidence', 0):.2f}"
                cv2.putText(
                    frame_to_store,
                    label,
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
    except Exception:
        frame_to_store = frame_data

    with frame_lock:
        current_frame = frame_to_store
        current_detections = metadata

    # Extract detections from metadata
    processor_data = metadata.get('processor', {})

    if processor_data.get('event', False):
        detections_list = convert_to_frontend_format(processor_data)

        # Update stats
        with stats_lock:
            stats['total_frames_processed'] += 1
            stats['total_detections'] += len(detections_list)

        # Broadcast to all connected clients
        if detections_list:
            socketio.emit('detections', detections_list)


def convert_to_frontend_format(processor_data):
    """
    Convert VisionFlow detection format to frontend Detection[] format

    Frontend expects:
    {
        id: string,
        class: string,
        confidence: number,
        bbox: [x, y, width, height],
        timestamp: number
    }
    """
    detections = []
    tags = processor_data.get('tags', [])

    for tag in tags:
        detection = {
            'id': f"{int(time.time() * 1000)}-{tag.get('class_name', 'unknown')}",
            'class': tag.get('class_name', 'unknown'),
            'confidence': tag.get('confidence', 0.0),
            'bbox': [
                tag['bbox']['x'],
                tag['bbox']['y'],
                tag['bbox']['width'],
                tag['bbox']['height']
            ],
            'timestamp': int(time.time() * 1000)
        }
        detections.append(detection)

    return detections


# SocketIO Event Handlers
@socketio.on('connect')
def handle_connect():
    """Handle new client connection"""
    with clients_lock:
        connected_clients.add(request.sid)
        with stats_lock:
            stats['connected_clients'] = len(connected_clients)

    logger.info(f"Client connected. Total clients: {len(connected_clients)}")
    emit('connection_status', {
        'status': 'connected',
        'message': 'Connected to VisionFlow backend',
        'timestamp': int(time.time() * 1000)
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    with clients_lock:
        connected_clients.discard(request.sid)
        with stats_lock:
            stats['connected_clients'] = len(connected_clients)

    logger.info(f"Client disconnected. Total clients: {len(connected_clients)}")


@socketio.on('ping')
def handle_ping():
    """Handle ping from client"""
    emit('pong', {'timestamp': int(time.time() * 1000)})


@socketio.on('frame')
def handle_frame_websocket(data):
    """
    Handle frame received from frontend via WebSocket

    Expected data:
    {
        "data": "base64_encoded_image_data",
        "timestamp": 1234567890
    }

    Emits:
    - 'detections': List of detections found in the frame
    - 'error': If processing fails
    """
    global frontend_clip_recorder

    # Skip frame if still processing previous one (avoid lag/buffering)
    with processing_frame_lock:
        if processing_frame_flag['processing']:
            return
        processing_frame_flag['processing'] = True

    try:
        # Validate data
        if not data or 'data' not in data:
            emit('error', {
                'message': 'Missing frame data',
                'timestamp': int(time.time() * 1000)
            })
            with processing_frame_lock:
                processing_frame_flag['processing'] = False
            return

        # Decode frame from base64
        frame = decode_base64_image(data['data'])

        if frame is None:
            emit('error', {
                'message': 'Could not decode frame data',
                'timestamp': int(time.time() * 1000)
            })
            with processing_frame_lock:
                processing_frame_flag['processing'] = False
            return

        # Update stats
        with stats_lock:
            stats['frames_received_from_frontend'] += 1

        # Process frame with object detector
        detections = []
        processing_time = 0

        with processor_lock:
            if frame_processor is not None:
                try:
                    start_time = time.time()

                    # Create frame data object
                    from src.app.models.frame_data import FrameData
                    frame_timestamp = data.get('timestamp', time.time())
                    frame_data = FrameData(
                        frame_id=f"frontend_{int(time.time() * 1000)}",
                        frame=frame,
                        timestamp=frame_timestamp,
                        metadata={
                            'source': 'frontend_websocket',
                            'timestamp': frame_timestamp
                        }
                    )

                    # Process frame
                    result = frame_processor.process(frame_data)

                    processing_time = (time.time() - start_time) * 1000

                    # Update stats
                    with stats_lock:
                        stats['total_frames_processed'] += 1

                    # Extract detections (even if empty, to clear old detections from UI)
                    if result and result.metadata.get('processor', {}).get('event', False):
                        processor_data = result.metadata.get('processor', {})
                        detections = convert_to_frontend_format(processor_data)

                        # Update detection stats
                        with stats_lock:
                            stats['total_detections'] += len(detections)

                        # Handle clip recording in background (non-blocking)
                        if frontend_clip_recorder and detections:
                            try:
                                frontend_clip_recorder.handle_frame(frame, detections)
                            except Exception as clip_err:
                                logger.warning(f"Clip recorder error (non-fatal): {clip_err}")
                    else:
                        # No detections, send empty array to clear UI
                        detections = []

                    # ALWAYS emit detections (even empty) to keep UI responsive
                    socketio.emit('detections', detections)

                except Exception as e:
                    logger.error(f"Error processing frame from WebSocket: {e}", exc_info=True)
                    emit('error', {
                        'message': f'Processing error: {str(e)}',
                        'timestamp': int(time.time() * 1000)
                    })
            else:
                logger.warning("No frame processor configured")
                emit('error', {
                    'message': 'No frame processor configured',
                    'timestamp': int(time.time() * 1000)
                })

    except Exception as e:
        logger.error(f"Error in handle_frame_websocket: {e}", exc_info=True)
        emit('error', {
            'message': f'Unexpected error: {str(e)}',
            'timestamp': int(time.time() * 1000)
        })
    finally:
        # Always reset processing flag
        with processing_frame_lock:
            processing_frame_flag['processing'] = False


# REST API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    uptime = time.time() - stats['server_start_time']

    return jsonify({
        'status': 'healthy',
        'service': 'VisionFlow v2 Backend',
        'version': '2.0.0',
        'uptime_seconds': round(uptime, 2),
        'stats': {
            'total_frames_processed': stats['total_frames_processed'],
            'total_detections': stats['total_detections'],
            'connected_clients': stats['connected_clients']
        }
    }), 200


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get current statistics"""
    uptime = time.time() - stats['server_start_time']

    return jsonify({
        'uptime_seconds': round(uptime, 2),
        'total_frames_processed': stats['total_frames_processed'],
        'total_detections': stats['total_detections'],
        'connected_clients': stats['connected_clients'],
        'avg_detections_per_frame': (
            stats['total_detections'] / stats['total_frames_processed']
            if stats['total_frames_processed'] > 0 else 0
        )
    }), 200


@app.route('/detections/latest', methods=['GET'])
def get_latest_detections():
    """Get latest detection results (REST endpoint for polling)"""
    with frame_lock:
        if current_detections is None:
            return jsonify({
                'detections': [],
                'timestamp': int(time.time() * 1000)
            }), 200

        processor_data = current_detections.get('processor', {})
        detections_list = convert_to_frontend_format(processor_data)

        return jsonify({
            'detections': detections_list,
            'timestamp': int(time.time() * 1000),
            'frame_info': processor_data.get('frame_info', {}),
            'performance': processor_data.get('performance', {})
        }), 200


@app.route('/video_feed')
def video_feed():
    """
    MJPEG video streaming endpoint
    Streams frames from the backend capturer in real-time
    """
    def generate():
        """Generate MJPEG stream from current frames"""
        try:
            while True:
                with frame_lock:
                    if current_frame is not None:
                        # Encode frame as JPEG
                        _, jpeg = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_bytes = jpeg.tobytes()

                        # Yield frame in MJPEG format
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Small delay to control frame rate (~30 FPS)
                time.sleep(0.033)

        except GeneratorExit:
            logger.info("Client disconnected from video feed")
        except Exception as e:
            logger.error(f"Error in video_feed generator: {e}")

    return app.response_class(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'VisionFlow v2 Backend',
        'version': '2.0.0',
        'description': 'Real-time object detection streaming server',
        'endpoints': {
            'health': '/health',
            'stats': '/stats',
            'latest_detections': '/detections/latest',
            'video_feed': '/video_feed (MJPEG stream)',
            'process_frame': '/process/frame (POST)',
            'detection_filter': '/detection/filter (POST, GET)',
            'websocket': 'ws://localhost:5001'
        }
    }), 200


@app.route('/detection/filter', methods=['GET', 'POST'])
def detection_filter():
    """
    Get or set detection class filter

    GET: Returns current active filter
    POST: Updates active filter

    Expected JSON body for POST:
    {
        "filter_classes": ["person", "cell phone"],  // or null for all classes
        "confidence_threshold": 0.5  // optional
    }

    Returns:
    {
        "status": "success",
        "active_filters": ["person", "cell phone"],
        "confidence_threshold": 0.5,
        "available_classes": ["person", "bicycle", "car", ...]
    }
    """
    global active_class_filter

    if request.method == 'GET':
        # Return current filter state
        with filter_lock:
            current_filter = active_class_filter

        # Get available classes from processor
        available_classes = []
        with processor_lock:
            if frame_processor is not None:
                if hasattr(frame_processor, 'class_names'):
                    available_classes = frame_processor.class_names
                elif hasattr(frame_processor, 'classes'):
                    available_classes = frame_processor.classes

        return jsonify({
            'status': 'success',
            'active_filters': current_filter,
            'available_classes': available_classes
        }), 200

    elif request.method == 'POST':
        try:
            data = request.get_json()

            if not data:
                return jsonify({
                    'error': 'Missing request body',
                    'message': 'Request body must contain JSON data'
                }), 400

            # Get filter classes from request
            new_filter = data.get('filter_classes')
            confidence_threshold = data.get('confidence_threshold')

            # Validate filter_classes
            if new_filter is not None and not isinstance(new_filter, list):
                return jsonify({
                    'error': 'Invalid filter_classes',
                    'message': 'filter_classes must be a list or null'
                }), 400

            # Update global filter
            with filter_lock:
                active_class_filter = new_filter

            # Update processor if it supports dynamic filtering
            with processor_lock:
                if frame_processor is not None:
                    if hasattr(frame_processor, 'set_filter_classes'):
                        frame_processor.set_filter_classes(new_filter)
                        logger.info(f"Updated detection filter: {new_filter}")

                    if hasattr(frame_processor, 'confidence_threshold') and confidence_threshold is not None:
                        frame_processor.confidence_threshold = confidence_threshold
                        logger.info(f"Updated confidence threshold: {confidence_threshold}")

            return jsonify({
                'status': 'success',
                'active_filters': new_filter,
                'confidence_threshold': confidence_threshold,
                'message': 'Detection filter updated successfully'
            }), 200

        except Exception as e:
            logger.error(f"Error updating detection filter: {e}")
            return jsonify({
                'error': 'Server error',
                'message': str(e)
            }), 500


@app.route('/process/frame', methods=['POST'])
def process_frame():
    """
    Process a single frame from the frontend

    Expected JSON body:
    {
        "frame": "base64_encoded_image_data",
        "timestamp": 1234567890
    }

    Returns:
    {
        "detections": [...],
        "processing_time_ms": 123.45,
        "timestamp": 1234567890
    }
    """
    global frontend_clip_recorder
    try:
        # Get JSON data
        data = request.get_json()

        if not data or 'frame' not in data:
            return jsonify({
                'error': 'Missing frame data',
                'message': 'Request body must contain "frame" field with base64 encoded image'
            }), 400

        # Decode frame
        frame = decode_base64_image(data['frame'])

        if frame is None:
            return jsonify({
                'error': 'Invalid frame data',
                'message': 'Could not decode base64 image'
            }), 400

        # Update stats
        with stats_lock:
            stats['frames_received_from_frontend'] += 1

        # Process frame with object detector
        detections = []
        processing_time = 0

        with processor_lock:
            if frame_processor is not None:
                try:
                    start_time = time.time()

                    # Create frame data object
                    from src.app.models.frame_data import FrameData
                    frame_timestamp = data.get('timestamp', time.time())
                    frame_data = FrameData(
                        frame_id=f"frontend_{int(time.time() * 1000)}",
                        frame=frame,
                        timestamp=frame_timestamp,
                        metadata={
                            'source': 'frontend',
                            'timestamp': frame_timestamp
                        }
                    )

                    # Process frame
                    result = frame_processor.process(frame_data)

                    processing_time = (time.time() - start_time) * 1000

                    # Extract detections
                    if result and result.metadata.get('processor', {}).get('event', False):
                        processor_data = result.metadata.get('processor', {})
                        detections = convert_to_frontend_format(processor_data)

                        # Update stats
                        with stats_lock:
                            stats['total_frames_processed'] += 1
                            stats['total_detections'] += len(detections)

                        # Broadcast to connected WebSocket clients
                        if detections:
                            socketio.emit('detections', detections)

                    # Clip generation from frontend frames
                    if frontend_clip_recorder is not None:
                        try:
                            frontend_clip_recorder.handle_frame(frame, detections)
                        except Exception as exc:  # noqa: BLE001
                            logger.error(f"Failed to handle frontend clip recording: {exc}")

                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    return jsonify({
                        'error': 'Processing error',
                        'message': str(e)
                    }), 500
            else:
                logger.warning("Frame processor not available")
                return jsonify({
                    'error': 'Processor not available',
                    'message': 'Object detector is not initialized'
                }), 503

        # Return results
        return jsonify({
            'detections': detections,
            'processing_time_ms': round(processing_time, 2),
            'timestamp': data.get('timestamp', int(time.time() * 1000)),
            'frame_info': {
                'width': frame.shape[1],
                'height': frame.shape[0],
                'channels': frame.shape[2] if len(frame.shape) > 2 else 1
            }
        }), 200

    except Exception as e:
        logger.error(f"Error in process_frame endpoint: {e}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500


def run_server(host='0.0.0.0', port=5000, debug=False, clip_config=None):
    """
    Start the Flask-SocketIO server

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 5000)
        debug: Enable debug mode (default: False)
        clip_config: Optional dict for frontend clip recording (videos to B2/Mongo)
    """
    global frontend_clip_recorder

    if clip_config:
        try:
            frontend_clip_recorder = FrontendClipRecorder(clip_config)
            logger.info("Frontend clip recorder initialized")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to initialize frontend clip recorder: {exc}")
            frontend_clip_recorder = None
    logger.info("=" * 60)
    logger.info(f"Starting VisionFlow v2 Web Server on {host}:{port}")
    logger.info("=" * 60)

    try:
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=False,
            log_output=not debug
        )
    except Exception as e:
        logger.error(f"Error starting web server: {e}")
        raise


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run server
    run_server(host='0.0.0.0', port=5000, debug=True)
