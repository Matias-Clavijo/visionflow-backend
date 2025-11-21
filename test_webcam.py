#!/usr/bin/env python3
"""
Quick webcam test - verifies camera access and OpenCV functionality
"""

import cv2
import sys

print("Testing webcam access...")
print(f"OpenCV version: {cv2.__version__}")

# Try to open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot open webcam")
    print("\nPossible solutions:")
    print("1. Check camera permissions in System Settings > Privacy & Security > Camera")
    print("2. Grant Terminal.app or Python access to camera")
    print("3. Try a different camera index (currently using 0)")
    sys.exit(1)

# Get camera properties
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"✅ Webcam opened successfully!")
print(f"   Resolution: {int(width)}x{int(height)}")
print(f"   FPS: {fps}")

# Try to read a frame
ret, frame = cap.read()

if ret:
    print(f"✅ Frame captured successfully!")
    print(f"   Frame shape: {frame.shape}")
    print(f"   Frame dtype: {frame.dtype}")
else:
    print("❌ ERROR: Cannot read frame from webcam")
    cap.release()
    sys.exit(1)

# Try to read a few more frames
print("\nTesting continuous capture (5 frames)...")
for i in range(5):
    ret, frame = cap.read()
    if ret:
        print(f"   Frame {i+1}/5: OK ({frame.shape})")
    else:
        print(f"   Frame {i+1}/5: FAILED")
        break

cap.release()
print("\n✅ Webcam test completed successfully!")
print("You can now run the main application.")
