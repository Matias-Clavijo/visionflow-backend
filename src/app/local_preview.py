#!/usr/bin/env python3
"""
Single-process local preview for VisionFlow:
- Opens webcam
- Runs YOLO11 detector
- Draws bounding boxes
- Shows a live OpenCV window (press 'q' to quit)
This bypasses the web server and multiprocessing to avoid shared-memory issues.
"""

import cv2
import time
import sys

from src.app.main_with_web import create_default_config
from src.app.core.processors.object_detector_yolo11 import ObjectDetectorYOLO11
from src.app.models.frame_data import FrameData


def main():
    config = create_default_config()
    cam_conf = config["webcam_capturer"]
    det_conf = config["object_detector_yolo11"]

    # Init detector
    detector = ObjectDetectorYOLO11(det_conf)

    # Open webcam
    cap = cv2.VideoCapture(cam_conf.get("device_index", 0))
    if not cap.isOpened():
        print(f"Could not open webcam device {cam_conf.get('device_index', 0)}")
        sys.exit(1)

    # Configure basic properties
    if cam_conf.get("target_width") and cam_conf.get("target_height"):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_conf["target_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_conf["target_height"])
    cap.set(cv2.CAP_PROP_FPS, cam_conf.get("fps", 30))

    cv2.namedWindow("VisionFlow Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VisionFlow Live", 960, 540)

    print("Press 'q' to quit")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        fd = FrameData(
            frame_id=str(time.time()),
            frame=frame,
            timestamp=time.time(),
            metadata={"capturer_name": cam_conf.get("name", "Webcam")},
        )

        result = detector.process(fd)
        if result is None:
            result = fd

        # Draw boxes if present
        meta = result.metadata.get("processor", {})
        for tag in meta.get("tags", []):
            bbox = tag.get("bbox", {})
            x, y = int(bbox.get("x", 0)), int(bbox.get("y", 0))
            w, h = int(bbox.get("width", 0)), int(bbox.get("height", 0))
            cv2.rectangle(result.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{tag.get('class_name', 'obj')} {tag.get('confidence', 0):.2f}"
            cv2.putText(
                result.frame,
                label,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("VisionFlow Live", result.frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
