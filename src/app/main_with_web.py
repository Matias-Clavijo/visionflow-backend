#!/usr/bin/env python3
"""
VisionFlow v2 - Main Application with Web Server Integration
Starts both the detection pipeline and Flask/SocketIO web server
"""

import logging
import sys
import threading
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.app.core.orchestrators.DirectOrchestrator import DirectOrchestrator
from src.app.core.capturers.rtsp_capturer import RtspCapturer
from src.app.core.capturers.webcam_capturer import WebcamCapturer
from src.app.core.processors.object_detector import ObjectDetector
from src.app.core.events_manager.events_poster import EventPoster
from src.app.web_server import run_server, update_frame_and_detections, set_frame_processor


def setup_logging(config):
    log_config = config.get('logging', {})
    level_str = log_config.get('level', 'INFO')
    log_file = log_config.get('log_file', 'visionflow.log')

    # Convert string level to logging constant
    level = getattr(logging, level_str.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)


def create_default_config():
    config = {
        'capturer_type': 'none',  # 'none' (frontend), 'webcam', or 'rtsp'

        'rtsp_capturer': {
            "name": "IPCamera01",
            "device_name": "Iphone",
            "rtsp_url": "rtsp://192.168.68.54:8554/preview",
            "max_reconnect_attempts": 5,
            "reconnect_delay": 2.0,
            "max_queue_size": 10000,
            "timeout": 15,
            # Quality reduction settings for better performance
            "target_width": 640,        # Reduce to 640px width
            "target_height": 480,       # Reduce to 480px height
            "jpeg_quality": 80,         # JPEG compression quality (0-100)
            "frame_skip": 2             # Process every 2nd frame
        },
        'webcam_capturer': {
            "name": "MacBookWebcam",
            "device_name": "MacBook Camera",
            "device_index": 0,          # 0 = default camera
            "max_reconnect_attempts": 3,
            "reconnect_delay": 2.0,
            "timeout": 10,
            "buffer_size": 3,
            # Quality settings for better performance
            "target_width": 640,        # Reduce to 640px width
            "target_height": 480,       # Reduce to 480px height
            "jpeg_quality": 85,         # JPEG compression quality (0-100)
            "frame_skip": 1             # Process every frame (1 = no skip)
        },
        'object_detector': {
            "name": "object_tagger",
            "model_path": "models/yolo/yolov4.weights",
            "classes_path": "models/yolo/coco.names",
            "config_path": "models/yolo/yolov4.cfg",
            "process_every_n_frames": 1,
            "strategy_for_skipped_frames": "CACHE"
        },
        'video_clip_generator': {
            "name": "video_clip_generator",
            "output_dir": "output",
            "use_cloud_storage": True,
            "b2_folder_path": "videos",
            "clip_duration": 30,
            "fps": 25.0,
            "codec": "mp4v",
            "container": "mp4",
            "quality": 100,
            "max_resolution": [1280, 720],
            "buffer_size": 1000,
            "trigger_mode": "time",
            "trigger_interval": 30,
            "max_workers": 3,
            "use_mongodb": True,
            "mongo_uri": "mongodb+srv://tesis:ucu2025tesis@visionflow.92xlyhu.mongodb.net/?retryWrites=true&w=majority&appName=visionflow",
            "mongo_database": "visionflow",
            "mongo_collection": "events"
        },
        'orchestrator': {
            'pool_shape': (640, 480, 3),  # Match RTSP capturer resolution to reduce memory
            'pool_buffers': 100,  # Reduced for macOS shared memory limits
            'capturer_queue_size': 1000,  # Reduced for macOS Queue limits (was 10000)
            'processor_queue_size': 500  # Reduced for macOS Queue limits (was 50000)
        },
        'logging': {
            'level': 'INFO',
            'log_file': 'visionflow.log'
        },
        'web_server': {
            'enabled': True,
            'host': '0.0.0.0',
            'port': 5001,  # Changed from 5000 to avoid macOS AirPlay conflict
            'debug': False
        }
    }
    return config


def initialize_components(config, logger):
    logger.info("Initializing VisionFlow2 components...")

    # Determine which capturer to use
    capturer_type = config.get('capturer_type', 'none')

    capturer = None
    if capturer_type == 'webcam':
        logger.info("Initializing Webcam Capturer...")
        try:
            capturer = WebcamCapturer(config['webcam_capturer'])
            logger.info(f"✓ Webcam Capturer '{capturer.name}' initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Webcam Capturer: {e}")
            return None, None
    elif capturer_type == 'rtsp':
        logger.info("Initializing RTSP Capturer...")
        try:
            capturer = RtspCapturer(config['rtsp_capturer'])
            logger.info(f"✓ RTSP Capturer '{capturer.name}' initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize RTSP Capturer: {e}")
            return None, None
    else:
        logger.info("No capturer configured - expecting frames from frontend")

    logger.info("Initializing Object Detector...")
    try:
        object_detector = ObjectDetector(config['object_detector'])
        logger.info(f"✓ Object Detector '{object_detector.name}' initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize Object Detector: {e}")
        logger.warning("Continuing without Object Detector - check model files exist")
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
        self.broadcast_every_n_frames = 2  # Broadcast every 2nd frame to reduce bandwidth

    def process(self, data):
        """Process frame and broadcast results to web clients"""
        # Call original processor
        result = self.processor.process(data)

        # Only broadcast every N frames to reduce load
        self.frame_count += 1
        if self.frame_count % self.broadcast_every_n_frames == 0:
            # Broadcast to web clients if there are detections
            if result and result.metadata.get('processor', {}).get('event', False):
                try:
                    update_frame_and_detections(result.frame, result.metadata)
                except Exception as e:
                    # Don't let web errors crash the pipeline
                    logging.getLogger(__name__).warning(f"Error broadcasting to web: {e}")

        return result


def setup_orchestrator(components, config, logger):
    capturer, object_detector = components

    logger.info("Setting up DirectOrchestrator...")

    orchestrator = DirectOrchestrator(
        pool_shape=config['orchestrator']['pool_shape'],
        pool_buffers=config['orchestrator']['pool_buffers']
    )

    logger.info(f"Registering capturer: {capturer.name}")
    orchestrator.register_capturer(
        capturer,
        config['orchestrator']['capturer_queue_size']
    )

    if object_detector:
        logger.info(f"Registering processor: {object_detector.name}")

        # Wrap processor with WebSocket broadcaster
        web_broadcaster = WebSocketBroadcaster(object_detector)

        orchestrator.register_processor(
            web_broadcaster,
            config['orchestrator']['processor_queue_size']
        )
        logger.info("✓ WebSocket broadcasting enabled for detections")
    else:
        logger.warning("Skipping processor registration - Object Detector not available")

    logger.info(f"Registering events manager")
    orchestrator.register_events_manager(EventPoster, config['video_clip_generator'])

    logger.info("Building orchestrator pipeline...")
    try:
        orchestrator.build()
        logger.info("✓ Orchestrator pipeline built successfully")
    except Exception as e:
        logger.error(f"✗ Failed to build orchestrator pipeline: {e}")
        return None

    return orchestrator


def start_web_server(config, logger):
    """Start Flask/SocketIO web server in a separate thread"""
    web_config = config.get('web_server', {})

    if not web_config.get('enabled', True):
        logger.info("Web server disabled in configuration")
        return None

    host = web_config.get('host', '0.0.0.0')
    port = web_config.get('port', 5000)
    debug = web_config.get('debug', False)

    logger.info("=" * 60)
    logger.info(f"Starting Web Server on {host}:{port}")
    logger.info("=" * 60)

    # Start web server in separate daemon thread
    server_thread = threading.Thread(
        target=run_server,
        args=(host, port, debug),
        daemon=True,
        name="WebServerThread"
    )
    server_thread.start()

    # Wait a moment for server to start
    time.sleep(2)

    logger.info("✓ Web server started successfully")
    logger.info(f"   - REST API: http://{host}:{port}")
    logger.info(f"   - WebSocket: ws://{host}:{port}")
    logger.info(f"   - Health Check: http://{host}:{port}/health")

    return server_thread


def main():
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

            # Setup and run orchestrator
            orchestrator = setup_orchestrator((capturer, object_detector), config, logger)
            if not orchestrator:
                logger.error("Failed to setup orchestrator. Exiting.")
                sys.exit(1)

            logger.info("=" * 60)
            logger.info("Starting VisionFlow2 Pipeline")
            logger.info("=" * 60)
            logger.info("Press Ctrl+C to stop the application")

            # Run the main pipeline (blocking)
            orchestrator.run()
        else:
            # Frontend-only mode - just keep server running
            logger.info("=" * 60)
            logger.info("Running in FRONTEND MODE")
            logger.info("=" * 60)
            logger.info("Waiting for frames from frontend...")
            logger.info(f"Send POST requests to: http://localhost:{config['web_server']['port']}/process/frame")
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
