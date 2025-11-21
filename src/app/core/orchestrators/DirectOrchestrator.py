import logging
import time
import multiprocessing as mp
import threading

from src.app.models.frame_data import FrameDataDescriptor, FrameData
from src.app.models.frames_queue_manager import FrameQueueManager
from src.app.models.shared_frame import SharedFramePool

logger = logging.getLogger(__name__)


class DirectOrchestrator:
    def __init__(self, pool_shape=(1080,1920,3), pool_buffers=10):
        # Prefer fork on POSIX to avoid pickling issues (OpenCV VideoCapture etc.)
        start_method = "fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method()
        self.ctx = mp.get_context(start_method)
        
        self._lock = self.ctx.Lock()
        
        self.capturer = None
        self.capture_queue = None  # Now stores actual Queue objects
        self.capture_queue_size = None  # Store queue sizes separately
        self.capturer_queue_manager: FrameQueueManager | None = None

        self.pool_buffers = pool_buffers
        self.pool_shape = pool_shape

        self.processors = []
        self.processor_processes = []  # Store multiprocessing.Process objects
        self.processors_queues = {}
        self.processors_queue_manager: FrameQueueManager | None = None

        self.events_managers = {}

        self.frames_processed = 0

        # Use multiprocessing primitives for cross-process communication
        self._running_flag = self.ctx.Value('i', 0)  # Shared integer for running state
        self._running_lock = self.ctx.Lock()
        self.is_running = False

    def register_capturer(self, capturer, queue_size):
        output_queue = self.ctx.Queue(maxsize=queue_size)
        
        capturer.register_output_queue(output_queue)
        
        self.capturer = capturer
        self.capture_queue = output_queue
        self.capture_queue_size = queue_size
        
        logger.info(f"Registered capturer '{capturer.name}' with queue size {queue_size}")
        return self

    def register_processor(self, processor, queue_size=10):
        self.processors.append(processor)
        self.processors_queues[processor.name] = queue_size
        return self

    def register_events_manager(self, manager_cls, params):
        for processor in self.processors:
            self.events_managers[processor.name] = manager_cls(params)
        return self

    def build(self):
        manager = FrameQueueManager(
            SharedFramePool(
                n_buffers=self.pool_buffers,
                shape=self.pool_shape,
                name="CAPTURER",
                ctx=self.ctx,
            ),
            ctx=self.ctx,
        )
        for processor in self.processors:
            manager.create_queue(processor.name, self.capture_queue_size)
        self.capturer_queue_manager = manager

        manager = FrameQueueManager(
            SharedFramePool(
                n_buffers=self.pool_buffers,
                shape=self.pool_shape,
                name="EVENTOS",
                ctx=self.ctx,
            ),
            ctx=self.ctx,
        )
        for processor in self.processors:
            manager.create_queue(processor.name, self.processors_queues[processor.name])
            self.events_managers[processor.name].register_pool(manager.get_pool())

        self.processors_queue_manager = manager
        

    def _block_worker_events(self, processor_name: str, events_manager):
        try:
            while True:
                # Check if we should stop
                with self._running_lock:
                    if self._running_flag.value == 0:
                        break
                        
                try:
                    queue = self.processors_queue_manager.get_queue(processor_name)

                    if queue is not None:
                        try:
                            frame_descriptor = queue.get_nowait()
                        except Exception as e:
                            time.sleep(0.001)  # Small sleep to prevent busy waiting
                            continue
                    else:
                        time.sleep(1)
                        continue

                    events_manager.process(frame_descriptor)

                except Exception as e:
                    print("error")
                    logger.error(f"Error in block {processor_name}: {str(e)}")
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Critical error in worker thread for {processor_name}: {str(e)}")
        finally:
            logger.info(f"Worker thread for block {processor_name} stopped.")

    def _block_worker_processor(self, processor_name: str, processor):
        processed_frame_count = 0
        try:
            while True:
                # Check if we should stop
                with self._running_lock:
                    if self._running_flag.value == 0:
                        break
                        
                try:
                    specific_processor_queue = self.capturer_queue_manager.get_queue(processor_name)
                    pool = self.capturer_queue_manager.get_pool()

                    if specific_processor_queue is not None:
                        try:
                            frame_descriptor: FrameDataDescriptor = specific_processor_queue.get_nowait()
                            frame = pool.to_numpy(frame_descriptor.shm_idx)
                        except:
                            time.sleep(0.001)  # Small sleep to prevent busy waiting
                            continue
                    else:
                        time.sleep(1)
                        continue

                    data = FrameData(
                        frame_id=frame_descriptor.frame_id,
                        frame=frame,
                        timestamp=None,
                        metadata=frame_descriptor.metadata
                    )

                    frame_data: FrameData = processor.process(data)
                    # Detach frame from shared buffer before releasing it
                    if frame_data is not None and frame_data.frame is not None:
                        frame_copy = frame_data.frame.copy()
                        frame_data.frame = frame_copy
                        # Still put into processor queue if downstream needed
                        self.processors_queue_manager.put_frame_in_queue(
                            frame_data.frame_id,
                            frame_copy,
                            processor_name,
                            frame_data.metadata
                        )

                    pool.release(frame_descriptor.shm_idx)
                    
                    processed_frame_count += 1

                except Exception as e:
                    logger.error(f"Error in block {processor_name}: {str(e)}")
                    time.sleep(0.1)

        except Exception as e:
            print("error")
            logger.error(f"Critical error in worker thread for {processor_name}: {str(e)}")
        finally:
            logger.info(f"Worker thread for block {processor_name} stopped.")


    def start_events_manager(self):
        for processor in self.processors:
            # Use threads so web_server globals (frame buffer/socket) stay in-process
            t = threading.Thread(
                target=self._block_worker_events,
                args=(processor.name, self.events_managers[processor.name]),
                daemon=True,
                name=f"{processor.name}-events-thread",
            )
            t.start()
            self.processor_processes.append(t)
            logger.info(f"Started events thread for {processor.name}")

    def start_processors(self):
        for processor in self.processors:
            # Run processors in threads to avoid pickling cv2.VideoCapture and to share memory with web server
            t = threading.Thread(
                target=self._block_worker_processor,
                args=(processor.name, processor),
                daemon=True,
                name=f"{processor.name}-processor-thread",
            )
            t.start()
            self.processor_processes.append(t)
            logger.info(f"Started processor thread for {processor.name}")

    def run(self):
        with self._running_lock:
            self.is_running = True
            self._running_flag.value = 1

        self.capturer.start()
        logger.info(f"Started block: {self.capturer.name}")

        try:
            self.start_events_manager()
            self.start_processors()
            frame_count = 0
            while True:
                with self._running_lock:
                    if self._running_flag.value == 0:
                        break
                man = self.capturer_queue_manager
                capturer_queue = self.capture_queue
                # Check if there are frames in the capturer's output queue
                if not capturer_queue.empty():
                    try:
                        frame_data = capturer_queue.get_nowait()
                        if frame_data is not None:
                            # Process frame and broadcast with detector metadata (keeps boxes)
                            self.frames_processed += 1
                            
                            # Put frame in each processor's queue individually to avoid reference counting issues
                            for processor in self.processors:
                                man.put_frame_in_queue(
                                    frame_id=frame_data.frame_id,
                                    frame_array=frame_data.frame,
                                    queue_name=processor.name,
                                    metadata=frame_data.metadata
                                )
                            frame_count += 1

                    except Exception as e:
                        print(f"errorrr {e}")
                        continue

        except KeyboardInterrupt:
            logger.info("Interrupted by user, terminating pipeline...")
        except Exception as e:
            print("error")
            logger.error(f"Unexpected error in pipeline: {str(e)}")
            raise
        finally:
            self.cleanup()
    
    def stop(self):
        """Stop the orchestrator gracefully"""
        with self._running_lock:
            self.is_running = False
            self._running_flag.value = 0
        logger.info("DirectOrchestrator stop requested")
    
    def cleanup(self):
        logger.info("Cleaning up shared memory resources...")

        try:
            self.capturer_queue_manager.get_pool().cleanup()
            logger.info(f"Cleaned up capturer pool")
        except Exception as e:
            logger.warning(f"Error cleaning up capturer pool: {e}")
        
        # Clean up processor queue manager
        if self.processors_queue_manager:
            try:
                self.processors_queue_manager.get_pool().cleanup()
                logger.info("Cleaned up processor pool")
            except Exception as e:
                logger.warning(f"Error cleaning up processor pool: {e}")
        
        logger.info("Shared memory cleanup completed")
