import uuid
import cv2
import time
import threading
import logging

from src.app.models.frame_data import FrameData

class WebcamCapturer:
    """
    Captures frames from a local webcam (e.g., MacBook camera)
    Uses OpenCV VideoCapture with device index (0 for default camera)
    """

    def __init__(self, params):
        self.logger = logging.getLogger(self.__class__.__name__)

        self._ensure_required_data(params)

        self.name = params.get("name", "Webcam")
        self.device_name = params.get("device_name", "LocalWebcam")
        self.device_index = params.get("device_index", 0)  # 0 = default camera

        self.timeout = params.get("timeout", 10)
        self.buffer_size = params.get("buffer_size", 3)

        # Frame processing options
        self.target_width = params.get("target_width", None)
        self.target_height = params.get("target_height", None)
        self.jpeg_quality = params.get("jpeg_quality", 85)
        self.frame_skip = params.get("frame_skip", 1)  # Process every Nth frame

        self.cap = None
        self.running = False
        self.output_queue = None

        self.reconnect_attempts = 0
        self.max_reconnect_attempts = params.get("max_reconnect_attempts", 3)
        self.reconnect_delay = params.get("reconnect_delay", 2.0)

        self.thread = None
        self.frame_count = 0

    def _ensure_required_data(self, params):
        if not params.get("name"):
            params["name"] = "Webcam"

    def register_output_queue(self, output_queue):
        self.output_queue = output_queue
        self.logger.info(f"Registered external output queue for {self.name}")

    def _configure_capture(self):
        """Configure webcam properties"""
        if not self.cap:
            return
        try:
            # Set buffer size (smaller for webcam = lower latency)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # Set resolution if specified
            if self.target_width and self.target_height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

            # Set FPS (30 is good for webcam)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        except Exception as e:
            self.logger.warning(f"Error configuring capture properties: {str(e)}")

    def get_stream_info(self):
        """Get webcam stream information"""
        if not self.cap or not self.cap.isOpened():
            return None

        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            return {
                'device_index': self.device_index,
                'resolution': f"{width}x{height}",
                'fps': fps,
                'width': width,
                'height': height
            }
        except Exception as e:
            self.logger.error(f"Error getting stream info: {str(e)}")
            return None

    def _reconnect(self):
        """Attempt to reconnect to webcam"""
        try:
            if self.cap:
                self.cap.release()

            self.logger.info(f"Attempting to reconnect to webcam (device {self.device_index})...")
            self.cap = cv2.VideoCapture(self.device_index)
            self._configure_capture()

            if self.cap.isOpened():
                self.logger.info(f"Successfully reconnected to webcam")
                self.reconnect_attempts = 0
                return True
            else:
                self.reconnect_attempts += 1
                return False

        except Exception as e:
            self.logger.error(f"Error during reconnection: {str(e)}")
            self.reconnect_attempts += 1
            return False

    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        self.logger.info(f"Starting webcam capture loop for device {self.device_index}")

        consecutive_failures = 0
        max_consecutive_failures = 10

        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    if not self._reconnect():
                        if self.reconnect_attempts >= self.max_reconnect_attempts:
                            self.logger.error(f"Max reconnection attempts reached. Stopping capture.")
                            break
                        time.sleep(self.reconnect_delay)
                        continue

                ret, frame = self.cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    self.logger.warning(f"Failed to read frame from webcam (attempt {consecutive_failures}/{max_consecutive_failures})")

                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error("Too many consecutive frame read failures. Attempting reconnect...")
                        consecutive_failures = 0
                        if not self._reconnect():
                            time.sleep(self.reconnect_delay)
                    continue

                # Reset failure counter on successful read
                consecutive_failures = 0

                # Frame skipping logic
                self.frame_count += 1
                if self.frame_skip > 1 and self.frame_count % self.frame_skip != 0:
                    continue

                # Resize frame if target dimensions specified
                if self.target_width and self.target_height:
                    if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                        frame = cv2.resize(frame, (self.target_width, self.target_height))

                # Create FrameData object
                frame_id = str(uuid.uuid4())
                frame_data = FrameData(
                    frame_id=frame_id,
                    frame=frame,
                    metadata={
                        'capturer_name': self.name,
                        'device_name': self.device_name,
                        'device_index': self.device_index,
                        'timestamp': time.time(),
                        'frame_count': self.frame_count,
                        'shape': frame.shape
                    }
                )

                # Put frame in output queue
                if self.output_queue is not None:
                    try:
                        self.output_queue.put_nowait(frame_data)
                    except:
                        # Queue is full, skip this frame
                        pass

            except Exception as e:
                self.logger.error(f"Error in capture loop: {str(e)}")
                time.sleep(0.1)

        # Cleanup
        if self.cap:
            self.cap.release()
        self.logger.info("Webcam capture loop stopped")

    def start(self):
        """Start capturing from webcam"""
        if self.running:
            self.logger.warning("Webcam capture is already running")
            return

        self.logger.info(f"Connecting to webcam device {self.device_index}...")

        try:
            self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open webcam device {self.device_index}")

            self._configure_capture()

            # Get and log stream info
            stream_info = self.get_stream_info()
            if stream_info:
                self.logger.info(f"Webcam stream info: {stream_info}")

            # Start capture thread
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

            self.logger.info(f"✓ Webcam capture started successfully on device {self.device_index}")

        except Exception as e:
            self.logger.error(f"Failed to start webcam capture: {str(e)}")
            if self.cap:
                self.cap.release()
            raise RuntimeError(f"Could not start webcam capture: {str(e)}")

    def stop(self):
        """Stop capturing from webcam"""
        if not self.running:
            return

        self.logger.info("Stopping webcam capture...")
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        if self.cap:
            self.cap.release()

        self.logger.info("Webcam capture stopped")

    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
