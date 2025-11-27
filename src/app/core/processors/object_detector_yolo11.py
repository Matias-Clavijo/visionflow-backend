from collections import deque
import logging
import os
import platform
import time

import cv2
import psutil

from src.app.core.utils.string_utils import resolve_path
from src.app.models.frame_data import FrameData

logger = logging.getLogger(__name__)


def detect_hardware():
    info = {
        'device': 'cpu',
        'device_name': 'CPU',
        'export_format': None,
        'recommended_pool_buffers': 100,
        'recommended_frame_size': (480, 640, 3)
    }

    try:
        import torch

        if torch.backends.mps.is_available():
            info['device'] = 'mps'
            info['device_name'] = f'Apple {platform.processor()} (MPS)'
            info['export_format'] = 'coreml'
            info['recommended_pool_buffers'] = 100
            info['recommended_frame_size'] = (480, 640, 3)
            logger.info("Detected Apple Silicon with Metal Performance Shaders")

        elif torch.cuda.is_available():
            cuda_device = torch.cuda.get_device_name(0)
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            info['device'] = 'cuda'
            info['device_name'] = cuda_device
            info['export_format'] = 'engine'  # TensorRT

            if total_memory_gb < 6:
                logger.info(f"⚡ Detected light device: {cuda_device} ({total_memory_gb:.1f}GB)")
                info['recommended_pool_buffers'] = 30
                info['recommended_frame_size'] = (416, 416, 3)
            else:
                logger.info(f"Detected CUDA GPU: {cuda_device} ({total_memory_gb:.1f}GB)")
                info['recommended_pool_buffers'] = 100
                info['recommended_frame_size'] = (480, 640, 3)

        else:
            cpu_info = platform.processor() or platform.machine()
            ram_gb = psutil.virtual_memory().total / (1024**3)

            info['device'] = 'cpu'
            info['device_name'] = f'{cpu_info} ({ram_gb:.1f}GB RAM)'
            info['export_format'] = 'openvino' if 'Intel' in cpu_info else None
            info['recommended_pool_buffers'] = 100
            info['recommended_frame_size'] = (480, 640, 3)
            logger.info(f"💻 Detected CPU: {cpu_info}")

    except ImportError:
        logger.warning("PyTorch not installed, defaulting to CPU mode")

    return info


class ObjectDetectorYOLO11:
    def __init__(self, params):
        self.name = params.get("name", "yolo11_detector")
        self.model_path = resolve_path(params.get("model_path", "models/yolo11/yolo11n.pt"))

        self.filter_classes = params.get("filter_classes", None)  # None = all classes
        self.confidence_threshold = params.get("confidence_threshold", 0.5)

        self.process_every_n_frames = params.get("process_every_n_frames", 1)
        self.strategy_for_skipped_frames = params.get("strategy_for_skipped_frames", "CACHE")

        self.auto_export = params.get("auto_export", True)

        self.hardware_info = detect_hardware()
        self.device = params.get("device") or self.hardware_info['device']

        self.frame_count = 0
        self.frames_processed = 0
        self.processing_times = deque(maxlen=1000)
        self.inference_times = deque(maxlen=1000)
        self.errors_count = 0

        self.last_detections = []
        self.has_to_process = True
        self.frames_to_cached = params.get("frames_to_cached", 100)
        self.last_similar_frame_with_no_events = None

        self.phasher = cv2.img_hash.PHash_create()

        self.model = None
        self.class_names = []

        self._load_model()

        logger.info(f"✓ {self.name} initialized")
        logger.info(f"  Device: {self.hardware_info['device_name']}")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Classes: {len(self.class_names)} total")
        if self.filter_classes:
            logger.info(f"  Filter: {self.filter_classes}")

    def _load_model(self):
        try:
            from ultralytics import YOLO

            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}")
                logger.info("YOLO11 will auto-download on first use")
                self.model_path = "yolo11n.pt"

            logger.info(f"Loading YOLO11 model from {self.model_path}...")
            self.model = YOLO(self.model_path)

            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            else:
                self.class_names = self._get_coco_classes()

            logger.info(f"Model loaded successfully with {len(self.class_names)} classes")

            if self.auto_export and self.hardware_info['export_format']:
                self._try_export_model()

            if hasattr(self.model, 'to'):
                self.model.to(self.device)
                logger.info(f"Model moved to device: {self.device}")

        except ImportError as e:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            raise ImportError("ultralytics package required for YOLO11") from e
        except Exception as e:
            logger.error(f"Error loading YOLO11 model: {e}")
            raise

    def _try_export_model(self):
        export_format = self.hardware_info['export_format']

        if not export_format:
            return

        export_dir = os.path.dirname(self.model_path)
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]

        export_extensions = {
            'coreml': '.mlpackage',
            'engine': '.engine',
            'openvino': '_openvino_model'
        }

        export_ext = export_extensions.get(export_format, '')
        export_path = os.path.join(export_dir, f"{model_name}{export_ext}")

        if os.path.exists(export_path):
            logger.info(f"Optimized model already exists: {export_path}")
            return

        logger.info(f"Exporting model to {export_format} format...")

        try:
            export_path = self.model.export(format=export_format, device=self.device)
            logger.info(f"Model exported successfully to {export_path}")

        except Exception as e:
            logger.warning(f"Export to {export_format} failed: {e}, Continuing with standard .pt model")

    def _get_coco_classes(self):
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def frames_similar_phash(self, frame1, frame2, threshold=5):
        """Check if two frames are similar using perceptual hashing"""
        hash1 = self.phasher.compute(frame1)
        hash2 = self.phasher.compute(frame2)
        distance = self.phasher.compare(hash1, hash2)
        return distance <= threshold

    def set_filter_classes(self, classes):
        """
        Update the class filter dynamically

        Args:
            classes (list or None): List of class names to detect, or None for all
        """
        self.filter_classes = classes
        if classes:
            logger.info(f"Detection filter updated: {classes}")
        else:
            logger.info("Detection filter cleared (detecting all classes)")

    def _get_class_indices(self):
        """Get class indices for filtering"""
        if not self.filter_classes:
            return None

        indices = []
        for class_name in self.filter_classes:
            if class_name in self.class_names:
                indices.append(self.class_names.index(class_name))

        return indices if indices else None

    def process(self, data: FrameData | None) -> FrameData | None:
        start_time = time.perf_counter()

        # Determine if we should process this frame
        should_process = (self.frame_count % self.process_every_n_frames == 0)

        if should_process:
            if not self.has_to_process:
                if self.frames_processed >= self.frames_to_cached:
                    should_process = True
                    self.frames_processed = 0
                    self.has_to_process = True
                else:
                    should_process = False
                    self.frames_processed += 1
            elif self.last_similar_frame_with_no_events is not None:
                same_image = self.frames_similar_phash(
                    data.frame,
                    self.last_similar_frame_with_no_events,
                )
                if same_image:
                    should_process = False
                    self.frames_processed += 1

        self.frame_count += 1
        frame = data.frame

        # Return cached detections if not processing
        if not should_process:
            if self.strategy_for_skipped_frames == "DROP":
                return None

            if self.strategy_for_skipped_frames == "CACHE":
                end_time = time.perf_counter()
                if self.last_detections:
                    total_time = (end_time - start_time) * 1000
                    data.metadata["processor"] = {
                        "count": len(self.last_detections),
                        "tags": self.last_detections,
                        "event": len(self.last_detections) != 0,
                        "cached": True,
                        "frame_info": {
                            "width": frame.shape[1],
                            "height": frame.shape[0],
                            "channels": frame.shape[2] if len(frame.shape) > 2 else 1
                        },
                        "performance": {
                            "processing_time_ms": {
                                "total": total_time,
                                "inference": 0
                            }
                        }
                    }
                return data

        # Process frame with YOLO11
        try:
            inference_start = time.perf_counter()

            # Get class indices for filtering
            class_indices = self._get_class_indices()

            # Run inference
            if class_indices is not None:
                results = self.model(frame, classes=class_indices, verbose=False, conf=self.confidence_threshold)
            else:
                results = self.model(frame, verbose=False, conf=self.confidence_threshold)

            inference_end = time.perf_counter()
            inference_ms = (inference_end - inference_start) * 1000
            self.inference_times.append(inference_ms)

            # Extract detections
            detections = []
            if results and len(results) > 0:
                result = results[0]  # First image result

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    # Batch transfer from GPU/MPS to CPU (much faster than individual transfers)
                    boxes_xyxy = boxes.xyxy.cpu().numpy()
                    boxes_cls = boxes.cls.cpu().numpy()
                    boxes_conf = boxes.conf.cpu().numpy()

                    for i in range(len(boxes)):
                        # Get box coordinates (already on CPU)
                        x1, y1, x2, y2 = boxes_xyxy[i]

                        # Convert to xywh format
                        x = int(x1)
                        y = int(y1)
                        w = int(x2 - x1)
                        h = int(y2 - y1)

                        # Get class and confidence (already on CPU)
                        class_id = int(boxes_cls[i])
                        confidence = float(boxes_conf[i])
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"

                        detection = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": {
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "center_x": x + w // 2,
                                "center_y": y + h // 2
                            }
                        }
                        detections.append(detection)

            end_time = time.perf_counter()
            total_processing_ms = (end_time - start_time) * 1000
            self.processing_times.append(total_processing_ms)

            self.last_detections = detections.copy()
            self.has_to_process = not (len(detections) != 0)

            if len(detections) == 0:
                self.last_similar_frame_with_no_events = frame.copy()

            data.metadata["processor"] = {
                "count": len(detections),
                "cached": False,
                "tags": detections,
                "event": len(detections) != 0,
                "frame_info": {
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "channels": frame.shape[2] if len(frame.shape) > 2 else 1
                },
                "model_info": {
                    "model": "YOLO11n",
                    "device": self.device,
                    "confidence_threshold": self.confidence_threshold,
                    "filter_classes": self.filter_classes
                },
                "performance": {
                    "processing_time_ms": {
                        "total": round(total_processing_ms, 2),
                        "inference": round(inference_ms, 2)
                    }
                }
            }

        except Exception as e:
            self.errors_count += 1
            logger.error(f"Error processing frame {self.frame_count}: {e}")
            data.metadata["processor"] = {"error": str(e)}

        return data

    @property
    def average_processing_time_ms(self):
        """Get average processing time"""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    @property
    def fps(self):
        """Get average FPS"""
        avg_time = self.average_processing_time_ms
        if avg_time > 0:
            return 1000.0 / avg_time
        return 0.0
