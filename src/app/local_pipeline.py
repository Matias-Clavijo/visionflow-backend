#!/usr/bin/env python3
"""
Local single-process pipeline:
- WebcamCapturer + YOLO11 detector + EventPoster uploads (B2 + Mongo)
- Native OpenCV window ("VisionFlow Live") with bounding boxes
- Live metrics overlay (FPS, inference latency, CPU%, RAM, time since last detection)
- Press 'q' to exit cleanly
"""

# ============================================================================
# DETECTION CONFIGURATION - Change this to switch detection targets
# ============================================================================
DETECTION_CLASS = "person"  # Options: "person" or "cell phone"
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections (0.0 to 1.0)
API_PORT = 5002  # Port for HTTP API to change detection settings
WEBSOCKET_PORT = 5003  # Port for WebSocket metrics (different from API_PORT)
ENABLE_WEBSOCKET = True  # Enable WebSocket for real-time metrics to frontend
# ============================================================================

import queue
import sys
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
load_dotenv(".env.local")
import time
from collections import deque
from pathlib import Path
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

import cv2
import numpy as np
import psutil

# WebSocket support (optional - graceful fallback if not available)
try:
    from flask import Flask
    from flask_cors import CORS
    from flask_socketio import SocketIO
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("⚠ flask-socketio not available - WebSocket metrics disabled")

# Ensure repo root on sys.path when running as a script
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging

# Import directly to avoid gevent monkey-patching from main_with_web
from src.app.core.capturers.webcam_capturer import WebcamCapturer
from src.app.core.events_manager.events_poster import EventPoster
from src.app.core.processors.object_detector_yolo11 import ObjectDetectorYOLO11, detect_hardware
from src.app.models.frame_data import FrameData

# Global state for API control
api_state = {
    "detector": None,
    "current_class": DETECTION_CLASS,
    "current_confidence": CONFIDENCE_THRESHOLD,
}
api_lock = threading.Lock()

# WebSocket server instance (if enabled)
socketio_instance = None
flask_app = None

# Video streaming
video_frame_queue = queue.Queue(maxsize=2)  # Queue for MJPEG streaming


class MetricTracker:
    """Track min/max/average for a single metric."""

    def __init__(self):
        self.min_value: float | None = None
        self.max_value: float | None = None
        self.total: float = 0.0
        self.count: int = 0

    def update(self, value: float | int | None):
        if value is None:
            return

        value = float(value)
        self.count += 1
        self.total += value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    def summary(self) -> dict:
        if self.count == 0:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}

        return {
            "min": round(self.min_value or 0.0, 1),
            "max": round(self.max_value or 0.0, 1),
            "avg": round(self.total / self.count, 1),
        }


class DetectionAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for detection configuration API"""

    def _set_cors_headers(self):
        """Set CORS headers to allow frontend access"""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests - return current detection settings"""
        if self.path == "/detection/filter":
            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            with api_lock:
                response = {
                    "status": "success",
                    "active_filter": api_state["current_class"],
                    "confidence_threshold": api_state["current_confidence"],
                    "available_classes": ["person", "cell phone"],
                }

            self.wfile.write(json.dumps(response).encode())

        elif self.path == "/health":
            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"status": "healthy", "service": "Local Pipeline API"}).encode()
            )

        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()

    def do_POST(self):
        """Handle POST requests - update detection settings"""
        if self.path == "/detection/filter":
            try:
                # Read request body
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode())

                new_class = data.get("detection_class")
                new_confidence = data.get("confidence_threshold")

                # Validate detection class
                if new_class and new_class not in ["person", "cell phone"]:
                    self.send_response(400)
                    self._set_cors_headers()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "status": "error",
                                "message": "Invalid detection_class. Use 'person' or 'cell phone'",
                            }
                        ).encode()
                    )
                    return

                # Update detector
                with api_lock:
                    detector = api_state["detector"]
                    if detector is None:
                        self.send_response(503)
                        self._set_cors_headers()
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(
                            json.dumps(
                                {"status": "error", "message": "Detector not initialized"}
                            ).encode()
                        )
                        return

                    # Apply new settings
                    if new_class:
                        detector.set_filter_classes([new_class])
                        api_state["current_class"] = new_class

                    if new_confidence is not None:
                        detector.confidence_threshold = float(new_confidence)
                        api_state["current_confidence"] = float(new_confidence)

                    response = {
                        "status": "success",
                        "message": "Detection settings updated",
                        "active_filter": api_state["current_class"],
                        "confidence_threshold": api_state["current_confidence"],
                    }

                self.send_response(200)
                self._set_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                self.send_response(500)
                self._set_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"status": "error", "message": str(e)}).encode()
                )

        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def start_api_server(port: int):
    """Start HTTP API server in a background thread"""
    server = HTTPServer(("0.0.0.0", port), DetectionAPIHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server


def start_websocket_server(port: int):
    """Start Flask/SocketIO server for real-time metrics broadcasting"""
    global socketio_instance, flask_app

    if not WEBSOCKET_AVAILABLE:
        return None

    flask_app = Flask(__name__)
    flask_app.config['SECRET_KEY'] = 'local-pipeline-secret'
    CORS(flask_app, resources={r"/*": {"origins": "*"}})

    socketio_instance = SocketIO(
        flask_app,
        cors_allowed_origins="*",
        async_mode='threading',
        logger=False,
        engineio_logger=False
    )

    @socketio_instance.on('connect')
    def handle_connect():
        print(f"✓ Frontend connected to metrics WebSocket")

    @socketio_instance.on('disconnect')
    def handle_disconnect():
        print(f"⚠ Frontend disconnected from metrics WebSocket")

    # MJPEG video streaming endpoint
    @flask_app.route('/video_feed')
    def video_feed():
        """MJPEG video stream endpoint"""
        def generate():
            while True:
                try:
                    # Get frame from queue (with timeout)
                    frame_data = video_frame_queue.get(timeout=1.0)
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame_data, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error in video feed: {e}")
                    break

        from flask import Response
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    # Run in background thread
    def run_server():
        socketio_instance.run(flask_app, host='0.0.0.0', port=port, debug=False, use_reloader=False, log_output=False, allow_unsafe_werkzeug=True)

    server_thread = threading.Thread(target=run_server, daemon=True, name="SocketIOThread")
    server_thread.start()

    return socketio_instance


def emit_metrics(
    fps: float,
    inference_ms: float,
    cpu: float,
    ram: float,
    detections_count: int,
    breakdown: dict | None = None,
):
    """Emit real-time metrics to connected frontend clients"""
    if socketio_instance is None:
        return

    payload = {
        'fps': round(fps, 1),
        'inference_ms': round(inference_ms, 1),
        'cpu_percent': round(cpu, 1),
        'ram_mb': round(ram, 1),
        'detections_count': detections_count,
        'timestamp': int(time.time() * 1000),
        'detection_class': api_state.get('current_class', 'unknown')
    }

    if breakdown:
        payload.update(breakdown)

    try:
        socketio_instance.emit('metrics', payload)
    except Exception:
        pass  # Silently fail if no clients connected


def emit_clip_event(event_type: str, clip_data: dict):
    """Emit clip creation events to frontend"""
    if socketio_instance is None:
        return

    try:
        socketio_instance.emit('clip_event', {
            'type': event_type,  # 'started', 'completed', 'error'
            'data': clip_data,
            'timestamp': int(time.time() * 1000)
        })
    except Exception:
        pass


class LocalClipRecorder:
    """
    Lightweight clip recorder that reuses EventPoster for writing/uploading clips.
    Collects a rolling buffer of frames and triggers a clip after a detection while
    honoring the cooldown window.
    """

    def __init__(self, clip_config: dict):
        self.poster = EventPoster(clip_config)
        clip_duration = float(clip_config.get("clip_duration", 15.0))
        fps = float(clip_config.get("fps", 25.0))
        buffer_size = max(int(clip_duration * fps), 1)
        self.buffer = deque(maxlen=buffer_size)

        self.min_clip_cooldown = float(
            clip_config.get("min_clip_cooldown", clip_duration)
        )
        self.frames_to_collect_after = max(buffer_size // 2, 1)
        self.detection_triggered = False
        self.frames_after_detection = 0
        self.last_clip_time = 0.0
        self.last_detection_tags: list[dict] = []

    def _normalize_detection(self, det: dict) -> dict:
        """Convert detection bbox to a flat list to match EventPoster expectations."""
        bbox = det.get("bbox", {})
        if isinstance(bbox, dict):
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            w = bbox.get("width", 0)
            h = bbox.get("height", 0)
        else:
            try:
                x, y, w, h = bbox
            except Exception:  # noqa: BLE001
                x = y = w = h = 0

        return {
            "class_id": det.get("class_id", det.get("id", 0)),
            "class_name": det.get("class_name", det.get("class", "unknown")),
            "confidence": det.get("confidence", 0.0),
            "bbox": [x, y, w, h],
        }

    def handle_frame(self, frame: np.ndarray, detections: list[dict]):
        if frame is None:
            return

        # Keep a copy of the frame for later writing (avoid mutation from drawing)
        self.buffer.append(frame.copy())
        has_detections = bool(detections)
        now = time.time()

        if has_detections and not self.detection_triggered:
            if (now - self.last_clip_time) >= self.min_clip_cooldown:
                self.detection_triggered = True
                self.frames_after_detection = 0
                self.last_detection_tags = detections

        if self.detection_triggered:
            self.frames_after_detection += 1
            if self.frames_after_detection >= self.frames_to_collect_after:
                self.detection_triggered = False
                self.frames_after_detection = 0
                self.last_clip_time = now

                normalized = [self._normalize_detection(d) for d in self.last_detection_tags]

                # Emit clip started event
                emit_clip_event('started', {
                    'detections_count': len(normalized),
                    'buffer_frames': len(self.buffer)
                })

                try:
                    self.poster.write_clip_from_numpy_frames(
                        list(self.buffer), normalized, source="local"
                    )
                    # Emit clip completed event
                    emit_clip_event('completed', {
                        'detections_count': len(normalized),
                        'success': True
                    })
                except Exception as e:
                    # Emit clip error event
                    emit_clip_event('error', {
                        'error': str(e)
                    })

    def stop(self):
        """Wait for pending uploads/DB writes to finish."""
        try:
            self.poster.executor.shutdown(wait=True)
        except Exception:
            pass


def draw_detections(frame: np.ndarray, detections: list[dict]):
    for det in detections:
        bbox = det.get("bbox", {})
        x = int(bbox.get("x", bbox[0] if isinstance(bbox, (list, tuple)) else 0))
        y = int(bbox.get("y", bbox[1] if isinstance(bbox, (list, tuple)) else 0))
        w = int(bbox.get("width", bbox[2] if isinstance(bbox, (list, tuple)) else 0))
        h = int(bbox.get("height", bbox[3] if isinstance(bbox, (list, tuple)) else 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{det.get('class_name', 'obj')} {det.get('confidence', 0):.2f}"
        cv2.putText(
            frame,
            label,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def overlay_metrics(
    frame: np.ndarray,
    fps_value: float,
    avg_inference_ms: float,
    cpu_percent: float,
    ram_mb: float,
    last_detection_age: float | None,
):
    lines = [
        f"FPS: {fps_value:.1f}",
        f"Inference: {avg_inference_ms:.1f} ms",
        f"CPU: {cpu_percent:.1f}%  RAM: {ram_mb:.1f} MB",
    ]
    if last_detection_age is not None:
        lines.append(f"Last detection: {last_detection_age:.1f}s ago")

    y = 20
    for text in lines:
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22


def setup_logging():
    """Setup logging for local pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("visionflow_local.log")],
    )
    return logging.getLogger(__name__)


def create_local_config():
    """Create configuration for local pipeline (without gevent dependencies)"""
    hardware_info = detect_hardware()

    config = {
        "webcam_capturer": {
            "name": "MacBookWebcam",
            "device_name": "MacBook Camera",
            "device_index": 0,
            "max_reconnect_attempts": 3,
            "reconnect_delay": 2.0,
            "timeout": 10,
            "buffer_size": 3,
            "target_width": 640,
            "target_height": 480,
            "jpeg_quality": 85,
            "frame_skip": 1,
        },
        "object_detector_yolo11": {
            "name": "yolo11_detector",
            "model_path": "models/yolo11/yolo11n.pt",
            "filter_classes": [DETECTION_CLASS],
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "process_every_n_frames": 1,
            "strategy_for_skipped_frames": "CACHE",
            "auto_export": True,
            "device": None,
        },
        "video_clip_generator": {
            "name": "video_clip_generator",
            "output_dir": "output",
            "use_cloud_storage": True,  # RE-ENABLED for B2 uploads
            "b2_folder_path": "videos",
            "clip_duration": 15,
            "fps": 25.0,
            "codec": "avc1",
            "container": "mp4",
            "quality": 85,
            "max_resolution": [1280, 720],
            "buffer_size": 500,
            "trigger_mode": "time",
            "trigger_interval": 120,
            "max_workers": 2,
            "min_clip_cooldown": 120,
            "use_mongodb": True,
            "mongo_uri": os.getenv('MONGODB_URI'),
            "mongo_database": "visionflow",
            "mongo_collection": "events",
        },
    }
    return config


def main():
    config = create_local_config()
    logger = setup_logging()
    logger.info("Starting local pipeline mode (web server disabled)")
    logger.info("=" * 60)
    logger.info(f"DETECTION TARGET: {DETECTION_CLASS}")
    logger.info(f"CONFIDENCE THRESHOLD: {CONFIDENCE_THRESHOLD}")
    logger.info(f"API SERVER: http://localhost:{API_PORT}")
    logger.info("=" * 60)

    # Start API server first
    try:
        api_server = start_api_server(API_PORT)
        logger.info(f"✓ API server started on port {API_PORT}")
        logger.info(f"  - Health check: http://localhost:{API_PORT}/health")
        logger.info(f"  - Detection filter: http://localhost:{API_PORT}/detection/filter")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        logger.warning("Continuing without API server...")

    # Start WebSocket server for real-time metrics (if enabled)
    if ENABLE_WEBSOCKET and WEBSOCKET_AVAILABLE:
        try:
            start_websocket_server(WEBSOCKET_PORT)
            time.sleep(1)  # Wait for server to start
            logger.info(f"✓ WebSocket server started on port {WEBSOCKET_PORT}")
            logger.info(f"  - Real-time metrics: ws://localhost:{WEBSOCKET_PORT}")
            logger.info(f"  - Clip notifications: ws://localhost:{WEBSOCKET_PORT}/clip_event")
        except Exception as e:
            logger.warning(f"Failed to start WebSocket server: {e}")
            logger.warning("Continuing without WebSocket metrics...")
    elif ENABLE_WEBSOCKET:
        logger.warning("WebSocket enabled but flask-socketio not available")
        logger.warning("Install with: pip install flask flask-cors flask-socketio")

    cam_conf = config["webcam_capturer"]
    det_conf = config["object_detector_yolo11"]
    clip_conf = config["video_clip_generator"]

    # Override detection configuration from top of file
    det_conf["filter_classes"] = [DETECTION_CLASS]
    det_conf["confidence_threshold"] = CONFIDENCE_THRESHOLD

    # Components
    detector = ObjectDetectorYOLO11(det_conf)
    clip_recorder = LocalClipRecorder(clip_conf)

    # Register detector with API
    with api_lock:
        api_state["detector"] = detector
    logger.info("✓ Detector registered with API")

    frame_queue: queue.Queue[FrameData] = queue.Queue(
        maxsize=cam_conf.get("buffer_size", 3) * 2
    )
    capturer = WebcamCapturer(cam_conf)
    capturer.register_output_queue(frame_queue)
    capturer.start()

    # Metrics trackers - Track actual YOLO processing times
    inference_times = deque(maxlen=60)  # Track only actual YOLO inference times
    process = psutil.Process()
    process.cpu_percent(None)  # Prime measurement
    last_detection_time: float | None = None
    frames_processed = 0
    metric_trackers = {
        "fps": MetricTracker(),
        "latency": MetricTracker(),
        "cpu": MetricTracker(),
        "ram": MetricTracker(),
    }

    cv2.namedWindow("VisionFlow Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VisionFlow Live", 960, 540)
    logger.info("Press 'q' in the window to stop")

    try:
        while True:
            try:
                frame_data: FrameData = frame_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            # Process frame with YOLO
            result = detector.process(frame_data)

            if result is None or result.frame is None:
                continue

            frames_processed += 1

            processor_meta = result.metadata.get("processor", {})
            detections = processor_meta.get("tags", []) or []
            if detections:
                last_detection_time = time.time()

            # Send to clip recorder before drawing overlays
            clip_recorder.handle_frame(result.frame, detections)

            display_frame = result.frame.copy()
            draw_detections(display_frame, detections)

            # Extract actual inference time from YOLO processor
            perf_data = processor_meta.get("performance", {})
            processing_times = perf_data.get("processing_time_ms", {})
            inference_ms = processing_times.get("inference", 0)

            # Track inference times for FPS calculation
            if inference_ms > 0:
                inference_times.append(inference_ms)

            # Calculate YOLO processing FPS (frames per second the model can process)
            # This is based on actual inference time, not display/drawing time
            avg_inference_ms = sum(inference_times) / len(inference_times) if inference_times else 0
            fps_value = (1000.0 / avg_inference_ms) if avg_inference_ms > 0 else 0.0

            cpu_percent = process.cpu_percent(interval=None)
            ram_mb = process.memory_info().rss / (1024 * 1024)
            last_det_age = time.time() - last_detection_time if last_detection_time else None

            overlay_metrics(
                display_frame,
                fps_value,
                avg_inference_ms,
                cpu_percent,
                ram_mb,
                last_det_age,
            )

            # Send frame to MJPEG stream (non-blocking)
            try:
                video_frame_queue.put_nowait(display_frame.copy())
            except queue.Full:
                pass  # Skip if queue is full (client is slow)

            # Emit real-time metrics to frontend (every ~10 frames to avoid spam)
            if frames_processed % 10 == 0:
                metric_trackers["fps"].update(fps_value)
                metric_trackers["latency"].update(avg_inference_ms)
                metric_trackers["cpu"].update(cpu_percent)
                metric_trackers["ram"].update(ram_mb)

                breakdown = {
                    "fps_stats": metric_trackers["fps"].summary(),
                    "latency_stats": metric_trackers["latency"].summary(),
                    "cpu_stats": metric_trackers["cpu"].summary(),
                    "ram_stats": metric_trackers["ram"].summary(),
                }

                emit_metrics(
                    fps_value,
                    avg_inference_ms,
                    cpu_percent,
                    ram_mb,
                    len(detections),
                    breakdown,
                )

            cv2.imshow("VisionFlow Live", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Shutting down local pipeline...")
        capturer.stop()
        clip_recorder.stop()
        cv2.destroyAllWindows()
        logger.info("Local pipeline stopped")


if __name__ == "__main__":
    main()
