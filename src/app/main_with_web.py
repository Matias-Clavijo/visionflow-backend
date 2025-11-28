#!/usr/bin/env python3
"""
VisionFlow v2 - Main Application with Web Server Integration
Starts both the detection pipeline and Flask/SocketIO web server
"""

# CRITICAL: Gevent monkey-patching with multiprocessing exclusion
# Patch everything EXCEPT thread/subprocess to avoid conflicts with multiprocessing
import logging

bootstrap_logger = logging.getLogger(__name__)

try:
    from gevent import monkey

    # Only patch socket, ssl, time (avoid patch_select to keep kqueue for pymongo)
    # This allows multiprocessing to work correctly and keeps Mongo DNS resolver happy
    monkey.patch_socket()
    monkey.patch_ssl()
    monkey.patch_time()
    bootstrap_logger.info("✓ Gevent partial monkey-patching applied (multiprocessing-safe)")
except ImportError:
    bootstrap_logger.warning("⚠ Gevent not installed - falling back to threading mode")
import queue
import socket
import sys
import threading
import time
import cv2
import queue as pyqueue
import os
from dotenv import load_dotenv
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.app.core.orchestrators.DirectOrchestrator import DirectOrchestrator
from src.app.core.capturers.rtsp_capturer import RtspCapturer
from src.app.core.capturers.webcam_capturer import WebcamCapturer
from src.app.core.processors.object_detector import ObjectDetector
from src.app.core.processors.object_detector_yolo11 import (
    ObjectDetectorYOLO11,
    detect_hardware,
)
from src.app.core.events_manager.events_poster import EventPoster
from src.app.web_server import (
    run_server,
    update_frame_and_detections,
    set_frame_processor,
)

# Queue for native OpenCV display (frame, metadata)
display_frame_queue: pyqueue.Queue | None = None


def set_display_queue(q: pyqueue.Queue):
    global display_frame_queue
    display_frame_queue = q


def setup_logging(config):
    log_config = config.get("logging", {})
    level_str = log_config.get("level", "INFO")
    log_file = log_config.get("log_file", "visionflow.log")

    # Convert string level to logging constant
    level = getattr(logging, level_str.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )
    return logging.getLogger(__name__)


def create_default_config():
    # Detect hardware for adaptive configuration
    hardware_info = detect_hardware()

    config = {
        # OPTIMIZED: Backend-driven capture for best performance
        "capturer_type": "webcam",  # 'webcam' for local dev, 'rtsp' for production, 'none' for frontend-driven
        "use_yolo11": True,  # Set to False to use YOLOv4 (backward compatibility)
        "rtsp_capturer": {
            "name": "IPCamera01",
            "device_name": "Iphone",
            "rtsp_url": "rtsp://192.168.68.54:8554/preview",
            "max_reconnect_attempts": 5,
            "reconnect_delay": 2.0,
            "max_queue_size": 10000,
            "timeout": 15,
            # Quality reduction settings for better performance
            "target_width": 640,  # Reduce to 640px width
            "target_height": 480,  # Reduce to 480px height
            "jpeg_quality": 80,  # JPEG compression quality (0-100)
            "frame_skip": 2,  # Process every 2nd frame
        },
        "webcam_capturer": {
            "name": "MacBookWebcam",
            "device_name": "MacBook Camera",
            "device_index": 0,  # 0 = default camera
            "max_reconnect_attempts": 3,
            "reconnect_delay": 2.0,
            "timeout": 10,
            "buffer_size": 3,
            # Quality settings for better performance
            "target_width": 640,  # Reduce to 640px width
            "target_height": 480,  # Reduce to 480px height
            "jpeg_quality": 85,  # JPEG compression quality (0-100)
            "frame_skip": 1,  # Process every frame (1 = no skip)
        },
        # YOLOv4 configuration (backward compatibility)
        "object_detector": {
            "name": "object_tagger",
            "model_path": "models/yolo/yolov4.weights",
            "classes_path": "models/yolo/coco.names",
            "config_path": "models/yolo/yolov4.cfg",
            "process_every_n_frames": 1,
            "strategy_for_skipped_frames": "CACHE",
        },
        # YOLO11 configuration (new default)
        "object_detector_yolo11": {
            "name": "yolo11_detector",
            "model_path": "models/yolo11/yolo11n.pt",
            "filter_classes": [
                "person"
            ],  # Default: only detect persons (change via API)
            "confidence_threshold": 0.5,
            "process_every_n_frames": 1,
            "strategy_for_skipped_frames": "CACHE",
            "auto_export": True,  # Auto-export to optimal format (CoreML, TensorRT, OpenVINO)
            "device": None,  # None = auto-detect, or 'mps', 'cuda', 'cpu'
        },
        "video_clip_generator": {
            "name": "video_clip_generator",
            "output_dir": "output",
            "use_cloud_storage": True,
            "b2_folder_path": "videos",
            "clip_duration": 15,  # Reduced from 30s to 15s
            "fps": 25.0,  # Reduced from 30 to 25 FPS
            "codec": "avc1",  # H.264 - Compatible with ALL browsers (Chrome, Firefox, Safari, Edge)
            "container": "mp4",
            "quality": 85,  # Reduced from 100 to 85 (still high quality, faster encoding)
            "max_resolution": [1280, 720],
            "buffer_size": 500,  # Reduced from 1000 to 500
            "trigger_mode": "time",
            "trigger_interval": 120,  # OPTIMIZED: Clip every 2 minutes (was 30s)
            "max_workers": 2,  # Reduced from 3 to 2 workers
            "min_clip_cooldown": 120,  # OPTIMIZED: 2 min cooldown (was 35s) - prevents spam
            "use_mongodb": True,
            "mongo_uri": os.getenv("MONGODB_URI"),
            "mongo_database": "visionflow",
            "mongo_collection": "events",
        },
        "orchestrator": {
            # Adaptive configuration based on hardware
            "pool_shape": hardware_info["recommended_frame_size"],
            "pool_buffers": hardware_info["recommended_pool_buffers"],
            "capturer_queue_size": 1000,  # Reduced for macOS Queue limits (was 10000)
            "processor_queue_size": 500,  # Reduced for macOS Queue limits (was 50000)
        },
        "logging": {"level": "INFO", "log_file": "visionflow.log"},
        "web_server": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 5001,  # Changed from 5000 to avoid macOS AirPlay conflict
            "debug": False,
        },
    }
    return config


def initialize_components(config, logger):
    logger.info("Initializing VisionFlow2 components...")

    # Determine which capturer to use
    capturer_type = config.get("capturer_type", "none")

    capturer = None
    if capturer_type == "hybrid":
        # Try RTSP first, fallback to Webcam
        logger.info("Hybrid mode: Attempting RTSP connection first...")
        try:
            capturer = RtspCapturer(config["rtsp_capturer"])
            logger.info(f"✓ RTSP Capturer '{capturer.name}' initialized successfully")
        except Exception as e:
            logger.warning(f"RTSP connection failed: {e}")
            logger.info("Falling back to Webcam...")
            try:
                capturer = WebcamCapturer(config["webcam_capturer"])
                logger.info(
                    f"✓ Webcam Capturer '{capturer.name}' initialized successfully"
                )
            except Exception as e2:
                logger.error(f"✗ Failed to initialize Webcam Capturer: {e2}")
                return None, None
    elif capturer_type == "webcam":
        logger.info("Initializing Webcam Capturer...")
        try:
            capturer = WebcamCapturer(config["webcam_capturer"])
            logger.info(f"✓ Webcam Capturer '{capturer.name}' initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Webcam Capturer: {e}")
            return None, None
    elif capturer_type == "rtsp":
        logger.info("Initializing RTSP Capturer...")
        try:
            capturer = RtspCapturer(config["rtsp_capturer"])
            logger.info(f"✓ RTSP Capturer '{capturer.name}' initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize RTSP Capturer: {e}")
            return None, None
    else:
        logger.info("No capturer configured - expecting frames from frontend")

    # Initialize object detector (YOLO11 or YOLOv4)
    use_yolo11 = config.get("use_yolo11", True)

    if use_yolo11:
        logger.info("Initializing YOLO11 Object Detector...")
        try:
            object_detector = ObjectDetectorYOLO11(config["object_detector_yolo11"])
            logger.info(
                f"✓ YOLO11 Detector '{object_detector.name}' initialized successfully"
            )
            logger.info(f"  Hardware: {object_detector.hardware_info['device_name']}")
            logger.info(f"  Model: {object_detector.model_path}")
            logger.info(f"  Average FPS: {object_detector.fps:.1f}")
        except Exception as e:
            logger.error(f"✗ Failed to initialize YOLO11 Detector: {e}")
            logger.warning("Falling back to YOLOv4...")
            use_yolo11 = False

    if not use_yolo11:
        logger.info("Initializing YOLOv4 Object Detector...")
        try:
            object_detector = ObjectDetector(config["object_detector"])
            logger.info(
                f"✓ YOLOv4 Detector '{object_detector.name}' initialized successfully"
            )
        except Exception as e:
            logger.error(f"✗ Failed to initialize Object Detector: {e}")
            logger.warning(
                "Continuing without Object Detector - check model files exist"
            )
            object_detector = None

    return capturer, object_detector


class WebSocketBroadcaster:
    """
    Wrapper class to broadcast detection results to web clients
    Intercepts frame processing to send updates via WebSocket
    """

    def __init__(self, processor):
        self.processor = processor
        self.name = processor.name + "_web_broadcaster"
        self.frame_count = 0
        self.broadcast_every_n_frames = 1  # Broadcast every frame for smooth MJPEG

    def process(self, data):
        """Process frame and broadcast results to web clients"""
        # Call original processor
        result = self.processor.process(data)

        # Always broadcast to web clients and local display
        if result is not None and result.frame is not None:
            try:
                update_frame_and_detections(result.frame.copy(), result.metadata)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Error broadcasting to web: {e}")

            # Send to local display queue (main thread will show)
            try:
                global display_frame_queue
                if display_frame_queue is not None:
                    if display_frame_queue.full():
                        try:
                            display_frame_queue.get_nowait()
                        except Exception:
                            pass
                    display_frame_queue.put_nowait((result.frame.copy(), result.metadata))
            except Exception:
                pass

        return result


def setup_orchestrator(components, config, logger):
    capturer, object_detector = components

    logger.info("Setting up DirectOrchestrator...")

    orchestrator = DirectOrchestrator(
        pool_shape=config["orchestrator"]["pool_shape"],
        pool_buffers=config["orchestrator"]["pool_buffers"],
    )

    logger.info(f"Registering capturer: {capturer.name}")
    orchestrator.register_capturer(
        capturer, config["orchestrator"]["capturer_queue_size"]
    )

    if object_detector:
        logger.info(f"Registering processor: {object_detector.name}")

        # Wrap processor with WebSocket broadcaster
        web_broadcaster = WebSocketBroadcaster(object_detector)

        orchestrator.register_processor(
            web_broadcaster, config["orchestrator"]["processor_queue_size"]
        )
        logger.info("✓ WebSocket broadcasting enabled for detections")
    else:
        logger.warning(
            "Skipping processor registration - Object Detector not available"
        )

    logger.info("Registering events manager")
    orchestrator.register_events_manager(EventPoster, config["video_clip_generator"])

    logger.info("Building orchestrator pipeline...")
    try:
        orchestrator.build()
        logger.info("✓ Orchestrator pipeline built successfully")
    except Exception as e:
        logger.error(f"✗ Failed to build orchestrator pipeline: {e}")
        return None

    return orchestrator


def run_display_loop(orchestrator):
    """Show live video with boxes in a native OpenCV window (blocks main thread)."""
    cv2.namedWindow("VisionFlow Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VisionFlow Live", 960, 540)
    try:
        while True:
            try:
                frame, _meta = display_frame_queue.get(timeout=0.05)
                cv2.imshow("VisionFlow Live", frame)
            except pyqueue.Empty:
                pass
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            with orchestrator._running_lock:
                if orchestrator._running_flag.value == 0:
                    break
    finally:
        cv2.destroyAllWindows()
        orchestrator.stop()


def start_web_server(config, logger):
    """Start Flask/SocketIO web server in a separate thread"""
    web_config = config.get("web_server", {})

    if not web_config.get("enabled", True):
        logger.info("Web server disabled in configuration")
        return None

    host = web_config.get("host", "0.0.0.0")
    port = web_config.get("port", 5000)
    debug = web_config.get("debug", False)

    # Fail fast with a clear message if the port is already taken
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError as e:
        raise RuntimeError(
            f"Port {port} is already in use on {host}. "
            "Stop the other process or change web_server.port in config."
        ) from e

    logger.info("=" * 60)
    logger.info(f"Starting Web Server on {host}:{port}")
    logger.info("=" * 60)

    error_queue: queue.SimpleQueue[Exception] = queue.SimpleQueue()

    def run_with_error_capture():
        try:
            run_server(host, port, debug, config.get("video_clip_generator"))
        except Exception as exc:  # noqa: BLE001
            error_queue.put(exc)

    # Start web server in separate daemon thread
    server_thread = threading.Thread(
        target=run_with_error_capture, daemon=True, name="WebServerThread"
    )
    server_thread.start()

    # Wait a moment for server to start
    time.sleep(2)

    if not error_queue.empty():
        raise RuntimeError(f"Web server failed to start: {error_queue.get()}")

    if not server_thread.is_alive():
        raise RuntimeError("Web server thread terminated unexpectedly during startup")

    logger.info("✓ Web server started successfully")
    logger.info(f"   - REST API: http://{host}:{port}")
    logger.info(f"   - WebSocket: ws://{host}:{port}")
    logger.info(f"   - Health Check: http://{host}:{port}/health")

    return server_thread


def main():
    load_dotenv()
    load_dotenv(".env.local")
    try:
        config = create_default_config()

        logger = setup_logging(config)
        logger.info("=" * 60)
        logger.info("Starting VisionFlow2 Application with Web Server")
        logger.info("=" * 60)
        logger.info("Configuration loaded successfully")

        # Start web server first
        web_server_thread = start_web_server(config, logger)

        # Initialize detection components
        capturer, object_detector = initialize_components(config, logger)

        if object_detector is None:
            logger.error("Object detector is required. Exiting.")
            sys.exit(1)

        # Register object detector with web server for frontend frame processing
        set_frame_processor(object_detector)
        logger.info("✓ Object detector registered with web server")

        # If we have a capturer, run the full pipeline with orchestrator
        if capturer is not None:
            logger.info("Running with capturer - starting full pipeline")

            # Setup orchestrator
            orchestrator = setup_orchestrator(
                (capturer, object_detector), config, logger
            )
            if not orchestrator:
                logger.error("Failed to setup orchestrator. Exiting.")
                sys.exit(1)

            logger.info("=" * 60)
            logger.info("Starting VisionFlow2 Pipeline")
            logger.info("=" * 60)
            logger.info("Press Ctrl+C to stop the application")

            # Prepare display queue and start pipeline in background
            disp_queue = pyqueue.Queue(maxsize=2)
            set_display_queue(disp_queue)

            pipeline_thread = threading.Thread(
                target=orchestrator.run, daemon=True, name="OrchestratorThread"
            )
            pipeline_thread.start()

            # Block on display loop; press 'q' to exit
            run_display_loop(orchestrator)

            orchestrator.stop()
            pipeline_thread.join(timeout=2)

        # Frontend mode (either by config or by fallback)
        if capturer is None:
            # Frontend-only mode - just keep server running
            logger.info("=" * 60)
            logger.info("Running in FRONTEND MODE")
            logger.info("=" * 60)
            logger.info("Waiting for frames from frontend...")
            logger.info(
                f"Send POST requests to: http://localhost:{config['web_server']['port']}/process/frame"
            )
            logger.info("Press Ctrl+C to stop the application")

            # Keep main thread alive
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("Shutdown requested by user")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Unexpected error in main application: {e}")
        logger.exception("Full error traceback:")
        sys.exit(1)
    finally:
        logger.info("VisionFlow2 application stopped")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
