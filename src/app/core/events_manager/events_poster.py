import logging

import cv2
import numpy as np
import os
import time
from collections import deque
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from src.app.core.utils.string_utils import resolve_path
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

from src.app.models.frame_data import FrameData
from src.app.models.frames_queue_manager import FrameDescriptor
from src.app.models.shared_frame import SharedFramePool


class EventPoster:
    def __init__(self, params, on_clip_created=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = params or {}

        self.output_dir = resolve_path(self.config.get("output_dir", "output/video_clips"))
        self.name = self.config.get("name")
        self.clip_duration = self.config.get("clip_duration", 10.0)
        self.fps = self.config.get("fps", 30.0)
        self.codec = self.config.get("codec", "avc1")
        self.container = self.config.get("container", "mp4")

        self.max_resolution = self.config.get("max_resolution", None)
        self.buffer_size = self.config.get("buffer_size", 1000)

        self.use_cloud_storage = self.config.get("use_cloud_storage", False)
        self.b2_app_key_id = self.config.get("b2_app_key_id", "005a7351082aa2d0000000001")
        self.b2_app_key = self.config.get("b2_app_key", "K005HOQbGe1cEaos7n3PSkB9KvdIhao")
        self.b2_bucket_name = self.config.get("b2_bucket_name", "visionflow-v1")
        self.b2_folder_path = self.config.get("b2_folder_path", "")
        self.keep_local_copy = self.config.get("keep_local_copy", True)

        self.use_mongodb = self.config.get("use_mongodb", False)
        self.mongo_uri = self.config.get("mongo_uri", None)
        self.mongo_host = self.config.get("mongo_host", "localhost")
        self.mongo_port = self.config.get("mongo_port", 27017)
        self.mongo_database = self.config.get("mongo_database", "visionflow")
        self.mongo_collection = self.config.get("mongo_collection", "video_clips")
        self.mongo_username = self.config.get("mongo_username", None)
        self.mongo_password = self.config.get("mongo_password", None)

        # Optional callback to notify clip lifecycle (started/completed/failed)
        self.on_clip_created = on_clip_created

        self.b2_api = None
        self.b2_bucket = None
        if self.use_cloud_storage:
            self._initialize_b2_api()

        # Initialize MongoDB connection
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_collection_obj = None
        if self.use_mongodb:
            self._initialize_mongodb()

        self.last_trigger_time = time.time()
        self.frames_per_clip = int(self.clip_duration * self.fps)

        self.max_workers = params.get("max_workers", 1)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.previous_frame_context_count = int(self.config.get("previous_frame_context_count", 100))
        self.after_frame_context_count = int(self.config.get("after_frame_context_count", 100))
        self.actual_frame_context_count = 0
        self.event_detected = False
        self.event_descriptor: FrameDescriptor | None = None
        self.frames_data_pool: SharedFramePool | None = None
        self.frames_descriptors: deque = deque(maxlen=self.previous_frame_context_count + self.after_frame_context_count + 1)

        # Anti-overlap: minimum cooldown between clips (in seconds)
        # This prevents generating overlapping clips from continuous detections
        self.min_clip_cooldown = params.get("min_clip_cooldown", self.clip_duration)  # Default: same as clip duration
        self.last_clip_generated_time = 0

        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info(f"VideoClipGenerator initialized:")
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info(f"  Clip duration: {self.clip_duration}s")
        self.logger.info(f"  FPS: {self.fps}")
        self.logger.info(f"  Frames per clip: {self.frames_per_clip}")
        self.logger.info(f"  Buffer size: {self.buffer_size}")
        self.logger.info(f"  Anti-overlap cooldown: {self.min_clip_cooldown}s (prevents overlapping clips)")
        self.logger.info(f"  Cloud storage enabled: {self.use_cloud_storage}")
        if self.use_cloud_storage:
            self.logger.info(f"  B2 bucket: {self.b2_bucket_name}")
            if self.b2_folder_path:
                self.logger.info(f"  B2 folder: {self.b2_folder_path}")
            self.logger.info(f"  Keep local copy: {self.keep_local_copy}")
        self.logger.info(f"  MongoDB enabled: {self.use_mongodb}")
        if self.use_mongodb:
            if self.mongo_uri:
                self.logger.info(
                    f"  MongoDB URI: {self.mongo_uri[:50]}...")  # Solo mostrar primeros 50 chars por seguridad
            else:
                self.logger.info(f"  MongoDB host: {self.mongo_host}:{self.mongo_port}")
            self.logger.info(f"  MongoDB database: {self.mongo_database}")
            self.logger.info(f"  MongoDB collection: {self.mongo_collection}")

    def register_pool(self, pool: SharedFramePool):
        self.frames_data_pool = pool

    def _initialize_b2_api(self):
        try:
            info = InMemoryAccountInfo()
            self.b2_api = B2Api(info)
            self.b2_api.authorize_account("production", self.b2_app_key_id, self.b2_app_key)
            self.b2_bucket = self.b2_api.get_bucket_by_name(self.b2_bucket_name)
            self.logger.info(f"Successfully connected to B2 bucket: {self.b2_bucket_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize B2 API: {e}")
            self.use_cloud_storage = False

    def _initialize_mongodb(self):
        try:
            if self.mongo_uri:
                connection_string = self.mongo_uri
                self.logger.info(f"Using MongoDB URI connection")
            else:
                if self.mongo_username and self.mongo_password:
                    connection_string = f"mongodb://{self.mongo_username}:{self.mongo_password}@{self.mongo_host}:{self.mongo_port}/"
                else:
                    connection_string = f"mongodb://{self.mongo_host}:{self.mongo_port}/"
                self.logger.info(f"Using MongoDB host: {self.mongo_host}:{self.mongo_port}")

            self.mongo_client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)

            self.mongo_client.admin.command('ping')

            self.mongo_db = self.mongo_client[self.mongo_database]
            self.mongo_collection_obj = self.mongo_db[self.mongo_collection]

            self.logger.info(f"Successfully connected to MongoDB: {self.mongo_database}.{self.mongo_collection}")

        except ConnectionFailure as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            self.use_mongodb = False
        except Exception as e:
            self.logger.error(f"Error initializing MongoDB: {e}")
            self.use_mongodb = False

    def _upload_to_b2(self, local_filepath, filename, folder="videos"):
        def upload_file():
            try:
                if not self.b2_bucket:
                    self.logger.error("B2 bucket not initialized")
                    return False

                bucket_filename = f"{folder}/{filename}"

                self.b2_bucket.upload_local_file(
                    local_file=local_filepath,
                    file_name=bucket_filename
                )

                self.logger.info(f"Successfully uploaded {bucket_filename} to B2 bucket")

                if not self.keep_local_copy:
                    try:
                        os.remove(local_filepath)
                        self.logger.info(f"Deleted local file: {filename}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete local file {filename}: {e}")

                return True

            except Exception as e:
                self.logger.error(f"Failed to upload {filename} to B2: {e}")
                return False

        if self.use_cloud_storage and self.b2_bucket:
            self.executor.submit(upload_file)
        else:
            self.logger.warning("Cloud storage not available for upload")

    def _save_metadata_to_mongodb(self, video_metadata):
        def save_metadata():
            try:
                if self.mongo_collection_obj is None:
                    self.logger.error("MongoDB collection not initialized")
                    return False

                if 'created_at' not in video_metadata:
                    video_metadata['created_at'] = datetime.now(timezone.utc)

                result = self.mongo_collection_obj.insert_one(video_metadata)

                self.logger.info(f"Successfully saved metadata to MongoDB with ID: {result.inserted_id}")
                return True

            except PyMongoError as e:
                self.logger.error(f"Failed to save metadata to MongoDB: {e}")
                return False
            except Exception as e:
                self.logger.error(f"Unexpected error saving metadata: {e}")
                return False

        if self.use_mongodb is not None and self.mongo_collection_obj is not None:
            self.executor.submit(save_metadata)
        else:
            self.logger.warning("MongoDB not available for metadata storage")

    def _resize_frame_if_needed(self, frame):
        if self.max_resolution is None:
            return frame

        height, width = frame.shape[:2]
        max_width, max_height = self.max_resolution

        width_scale = max_width / width if width > max_width else 1.0
        height_scale = max_height / height if height > max_height else 1.0
        scale = min(width_scale, height_scale)

        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return frame

    def write_clip_from_numpy_frames(self, frames, detections, source="frontend"):
        """Create a video clip from a list of numpy frames (BGR) and upload/save metadata."""
        if not frames:
            self.logger.warning("No frames provided for clip generation")
            return

        now_sec = int(time.time())
        # Use frame_id as filename base (without .mp4 extension)
        frame_id = f"clip_{source}_{now_sec}"
        filename = f"{frame_id}.mp4"

        def write_and_upload():
            writer = None
            try:
                # Notify start
                if self.on_clip_created:
                    try:
                        self.on_clip_created({
                            "status": "started",
                            "frame_id": frame_id,
                            "timestamp": now_sec,
                            "filename": filename,
                            "source": source,
                        })
                    except Exception as emit_exc:  # noqa: BLE001
                        self.logger.error(f"Failed to emit clip_created(started): {emit_exc}")

                sample_frame = self._resize_frame_if_needed(frames[0])
                height, width = sample_frame.shape[:2]

                filepath = os.path.join(self.output_dir, filename)
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                writer = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))

                written = 0
                for frame in frames:
                    frame_resized = self._resize_frame_if_needed(frame)
                    if frame_resized is None or frame_resized.size == 0:
                        continue
                    if len(frame_resized.shape) != 3 or frame_resized.shape[2] != 3:
                        continue
                    if frame_resized.dtype != np.uint8:
                        frame_resized = frame_resized.astype(np.uint8)

                    writer.write(frame_resized)
                    written += 1

                writer.release()
                writer = None
                if written == 0:
                    self.logger.warning("No valid frames written; skipping upload/metadata")
                    return

                # Upload video
                storage_path = f"{self.b2_folder_path or 'videos'}/{filename}"
                self._upload_to_b2(filepath, filename, folder=self.b2_folder_path or "videos")

                # Save metadata
                tags = []
                for det in detections:
                    try:
                        bbox = det.get("bbox", [0, 0, 0, 0])
                        x, y, w, h = bbox
                        tags.append({
                            "class_id": det.get("class_id", 0),
                            "class_name": det.get("class", det.get("class_name", "unknown")),
                            "confidence": det.get("confidence", 0.0),
                            "bbox": {
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "center_x": x + w / 2,
                                "center_y": y + h / 2,
                            },
                        })
                    except Exception:
                        continue

                video_metadata = {
                    "frame_id": frame_id,
                    "device": self.config.get("device_name", source),
                    "timestamp": now_sec,
                    "filename": filename,
                    "source": source,
                    # Store as datetime object for proper sorting in MongoDB
                    "created_at": datetime.now(timezone.utc),
                    "processor": {
                        "event": True,
                        "count": len(tags),
                        "tags": tags,
                        "frame_info": {
                            "width": width,
                            "height": height,
                            "channels": 3,
                        },
                    },
                }
                self._save_metadata_to_mongodb(video_metadata)

                # Notify via callback (fire and forget)
                if self.on_clip_created:
                    try:
                        self.on_clip_created({
                            "status": "completed",
                            "frame_id": video_metadata["frame_id"],
                            "timestamp": video_metadata["timestamp"],
                            "filename": filename,
                            "storage_path": storage_path,
                            "bucket": self.b2_bucket_name,
                            "source": source,
                            "detection_count": len(tags),
                            "video_url": storage_path,
                        })
                    except Exception as emit_exc:  # noqa: BLE001
                        self.logger.error(f"Failed to emit clip_created callback: {emit_exc}")

            except Exception as e:
                self.logger.error(f"Failed to generate/upload frontend clip: {e}")
                if self.on_clip_created:
                    try:
                        self.on_clip_created({
                            "status": "failed",
                            "frame_id": frame_id,
                            "timestamp": now_sec,
                            "filename": filename,
                            "source": source,
                            "error": str(e),
                        })
                    except Exception as emit_exc:  # noqa: BLE001
                        self.logger.error(f"Failed to emit clip_created(failed): {emit_exc}")
            finally:
                if writer is not None:
                    writer.release()

        # Run in background to avoid blocking request handling
        self.executor.submit(write_and_upload)

    def _should_trigger_clip(self, descriptor: FrameDescriptor) -> bool:
        if not self.event_detected:
            has_event = descriptor.metadata.get("processor", {}).get("event", False)

            # Check if we're in cooldown period (anti-overlap)
            current_time = time.time()
            time_since_last_clip = current_time - self.last_clip_generated_time

            if has_event:
                if time_since_last_clip < self.min_clip_cooldown:
                    return False
                else:
                    self.event_detected = True
                    self.event_descriptor = descriptor
                    self.logger.info(f"Event detected! Starting clip generation (cooldown satisfied: {time_since_last_clip:.1f}s)")
        else:
            self.actual_frame_context_count += 1

        return self.actual_frame_context_count == (self.after_frame_context_count + self.previous_frame_context_count) + 1

    def _get_clip_frames(self, descriptor: FrameDescriptor):
        result = []

        for _ in range(len(self.frames_descriptors)):
            descriptor = self.frames_descriptors.pop()
            frame = self.frames_data_pool.to_numpy(descriptor.shm_idx)

            data = FrameData(
                frame_id=descriptor.frame_id,
                frame=frame,
                timestamp=None,
                metadata=descriptor.metadata,
                descriptor = descriptor
            )

            result.insert(0, data)

        return result

    def _write_video_clip_async(self, frames_data, filename):
        def write_video():
            writer = None
            try:
                filepath = os.path.join(self.output_dir, filename)

                if not frames_data:
                    self.logger.warning(f"No frames to write for clip: {filename}")
                    return

                sample_frame = self._resize_frame_if_needed(frames_data[0].frame)
                height, width = sample_frame.shape[:2]

                self.logger.info(f"Creating video writer: {filename}")

                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                writer = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))

                frames_written = 0
                for frame_data in frames_data:
                    try:
                        # frame_data is a FrameData object, not a dictionary
                        frame = self._resize_frame_if_needed(frame_data.frame)
                        
                        # Additional validation before writing
                        if frame is None or frame.size == 0:
                            self.logger.error(f"Invalid frame for writing: {frame_data.frame_id}")
                            continue
                            
                        # Check frame properties
                        if len(frame.shape) != 3 or frame.shape[2] != 3:
                            self.logger.error(f"Invalid frame format for video writing: {frame.shape}")
                            continue
                            
                        if frame.dtype != np.uint8:
                            self.logger.warning(f"Converting frame dtype from {frame.dtype} to uint8")
                            frame = frame.astype(np.uint8)

                        if frames_written == 0:
                            self.logger.info(f"First frame properties - Shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
                        
                        writer.write(frame)
                        frames_written += 1
                        self.frames_data_pool.release(frame_data.descriptor.shm_idx)


                    except Exception as e:
                        self.logger.error(f"Error writing frame to video: {e}")
                        continue

                writer.release()

                if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:  # At least 1KB
                    self.logger.info(f"Successfully created video clip: {filename} ({frames_written} frames)")

                    if self.use_cloud_storage:
                        self._upload_to_b2(filepath, filename)
                else:
                    self.logger.error(f"Video file seems corrupted or too small: {filename}")

            except Exception as e:
                self.logger.error(f"Error creating video clip {filename}: {e}")
            finally:
                if writer is not None and writer.isOpened():
                    writer.release()

        self.executor.submit(write_video)
        self.actual_frame_context_count = 0
        self.event_detected = False

        # Update last clip generated time for anti-overlap
        self.last_clip_generated_time = time.time()
        self.logger.info(f"Clip generation triggered. Next clip can be generated after {self.min_clip_cooldown}s cooldown.")



    def _generate_clip(self, descriptor: FrameDescriptor):
        try:
            clip_frames = self._get_clip_frames(descriptor)

            if len(clip_frames) < 5:
                self.logger.warning(f"Not enough frames for clip generation: {len(clip_frames)}")
                return

            base_filename = self.event_descriptor.frame_id
            filename = f"{base_filename}.{self.container}"

            self._write_video_clip_async(clip_frames, filename)
            self._save_metadata_to_mongodb(self.event_descriptor.metadata)

            self.last_trigger_time = time.time()

        except Exception as e:
            self.logger.error(f"Error generating clip: {e}")

    def _add_descriptor(self, descriptor: FrameDescriptor):
        try:
            if len(self.frames_descriptors) == self.frames_descriptors.maxlen:
                if self.frames_data_pool:
                    evicted = self.frames_descriptors.pop()
                    self.frames_data_pool.release(evicted.shm_idx)
            # Always append the incoming descriptor (do not overwrite it with evicted)
            self.frames_descriptors.append(descriptor)
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")

    def process(self, descriptor: FrameDescriptor):
        start_time = time.perf_counter()
        try:
            self._add_descriptor(descriptor)
            if self._should_trigger_clip(descriptor):
                self._generate_clip(descriptor)

            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000

        except Exception as e:
            self.logger.error(f"Error processing frame in VideoClipGenerator: {e}")
            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000
