#!/usr/bin/env python3
"""
VisionFlow v2 Web Server
Flask + SocketIO server for real-time detection streaming to frontend
"""

import logging
import time
import numpy as np
import cv2
import base64
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from threading import Lock

# Prefer eventlet for proper WebSocket support; fall back to threading if unavailable
try:
    import eventlet

    eventlet.monkey_patch()
    SOCKETIO_ASYNC_MODE = 'eventlet'
except ImportError:
    eventlet = None
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

if SOCKETIO_ASYNC_MODE != 'eventlet':
    logger.warning(
        "Eventlet not installed; falling back to threading mode. "
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


def set_frame_processor(processor):
    """
    Set the object detector processor for processing frames from frontend

    Args:
        processor: ObjectDetector instance
    """
    global frame_processor
    with processor_lock:
        frame_processor = processor
    logger.info("Frame processor registered for frontend frame processing")


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

    with frame_lock:
        current_frame = frame_data
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
            logger.debug(f"Broadcasted {len(detections_list)} detections to {len(connected_clients)} clients")


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
            'process_frame': '/process/frame (POST)',
            'websocket': 'ws://localhost:5001'
        }
    }), 200


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
                            logger.debug(f"Broadcasted {len(detections)} detections from frontend frame")

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


def run_server(host='0.0.0.0', port=5000, debug=False):
    """
    Start the Flask-SocketIO server

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 5000)
        debug: Enable debug mode (default: False)
    """
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
