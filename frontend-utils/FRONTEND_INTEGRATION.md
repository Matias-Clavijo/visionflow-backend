# VisionFlow v2 - Frontend Integration Guide

## Overview

VisionFlow v2 backend is configured to receive video frames from your Next.js frontend, process them with YOLOv4 object detection, and return detections in real-time.

**Architecture Flow:**
```
Frontend (Browser Camera) → Capture Frame → Send to Backend → YOLOv4 Inference → Return Detections
```

## Backend Setup

### 1. Start the Backend Server

```bash
cd visionflow-v2
./start.sh
```

The backend will start in **FRONTEND MODE** on port **5001**:
- REST API: `http://localhost:5001`
- WebSocket: `ws://localhost:5001`
- Frame Processing Endpoint: `POST http://localhost:5001/process/frame`

### 2. Backend Configuration

The backend is configured with `capturer_type: 'none'` to expect frames from the frontend instead of capturing from a local camera or RTSP stream.

## Frontend Integration

### API Endpoint

**POST** `http://localhost:5001/process/frame`

**Request Body (JSON):**
```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "timestamp": 1730833245123
}
```

**Response (JSON):**
```json
{
  "detections": [
    {
      "id": "1730833245123-person",
      "class": "person",
      "confidence": 0.89,
      "bbox": [100, 50, 200, 400],
      "timestamp": 1730833245123
    }
  ],
  "processing_time_ms": 45.23,
  "timestamp": 1730833245123,
  "frame_info": {
    "width": 640,
    "height": 480,
    "channels": 3
  }
}
```

### Implementation Example (Next.js/React)

#### 1. Capture Video from Browser Camera

```typescript
// hooks/useCamera.ts
import { useRef, useEffect, useState } from 'react';

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          }
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setIsReady(true);
          };
        }
      } catch (error) {
        console.error('Error accessing camera:', error);
      }
    }

    startCamera();

    return () => {
      const stream = videoRef.current?.srcObject as MediaStream;
      stream?.getTracks().forEach(track => track.stop());
    };
  }, []);

  return { videoRef, isReady };
}
```

#### 2. Capture and Send Frames to Backend

```typescript
// utils/frameProcessor.ts
export async function captureAndProcessFrame(
  videoElement: HTMLVideoElement,
  apiUrl: string = 'http://localhost:5001'
): Promise<Detection[]> {
  // Create canvas to capture frame
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;

  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not get canvas context');

  // Draw current video frame to canvas
  ctx.drawImage(videoElement, 0, 0);

  // Convert canvas to base64 JPEG
  const base64Frame = canvas.toDataURL('image/jpeg', 0.8);

  // Send to backend
  const response = await fetch(`${apiUrl}/process/frame`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      frame: base64Frame,
      timestamp: Date.now()
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  return data.detections || [];
}
```

#### 3. Component Example with Real-Time Detection

```typescript
// components/LiveDetection.tsx
'use client';

import { useEffect, useState, useRef } from 'react';
import { useCamera } from '@/hooks/useCamera';
import { captureAndProcessFrame } from '@/utils/frameProcessor';

interface Detection {
  id: string;
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
  timestamp: number;
}

export default function LiveDetection() {
  const { videoRef, isReady } = useCamera();
  const [detections, setDetections] = useState<Detection[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [fps, setFps] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    if (!isReady || !videoRef.current) return;

    let frameCount = 0;
    let lastTime = Date.now();

    // Process frames at ~10 FPS (adjust as needed)
    intervalRef.current = setInterval(async () => {
      if (isProcessing || !videoRef.current) return;

      setIsProcessing(true);
      try {
        const newDetections = await captureAndProcessFrame(videoRef.current);
        setDetections(newDetections);

        // Calculate FPS
        frameCount++;
        const now = Date.now();
        if (now - lastTime >= 1000) {
          setFps(frameCount);
          frameCount = 0;
          lastTime = now;
        }
      } catch (error) {
        console.error('Error processing frame:', error);
      } finally {
        setIsProcessing(false);
      }
    }, 100); // Process every 100ms = ~10 FPS

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isReady, isProcessing]);

  return (
    <div className="relative">
      {/* Video feed */}
      <video
        ref={videoRef}
        className="w-full max-w-2xl rounded-lg"
        autoPlay
        playsInline
        muted
      />

      {/* Detection overlays */}
      <svg
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
        viewBox={`0 0 ${videoRef.current?.videoWidth || 640} ${videoRef.current?.videoHeight || 480}`}
      >
        {detections.map((detection) => {
          const [x, y, width, height] = detection.bbox;
          return (
            <g key={detection.id}>
              {/* Bounding box */}
              <rect
                x={x}
                y={y}
                width={width}
                height={height}
                fill="none"
                stroke="#00ff00"
                strokeWidth="2"
              />
              {/* Label */}
              <text
                x={x}
                y={y - 5}
                fill="#00ff00"
                fontSize="14"
                fontWeight="bold"
              >
                {detection.class} ({(detection.confidence * 100).toFixed(0)}%)
              </text>
            </g>
          );
        })}
      </svg>

      {/* Stats */}
      <div className="absolute top-4 right-4 bg-black/70 text-white p-2 rounded">
        <div>FPS: {fps}</div>
        <div>Detections: {detections.length}</div>
        <div className={isProcessing ? 'text-yellow-400' : 'text-green-400'}>
          {isProcessing ? 'Processing...' : 'Ready'}
        </div>
      </div>
    </div>
  );
}
```

## WebSocket Alternative (Optional)

For even lower latency, you can use WebSocket to receive detections:

```typescript
// utils/websocket.ts
import io from 'socket.io-client';

export function connectToDetectionStream(
  onDetection: (detections: Detection[]) => void
) {
  const socket = io('http://localhost:5001', {
    transports: ['websocket']
  });

  socket.on('connect', () => {
    console.log('Connected to VisionFlow backend');
  });

  socket.on('detections', (detections: Detection[]) => {
    onDetection(detections);
  });

  socket.on('disconnect', () => {
    console.log('Disconnected from VisionFlow backend');
  });

  return socket;
}
```

## Performance Optimization

### 1. Frame Rate Control

Adjust the frame processing rate based on your needs:

```typescript
// Lower rate = less CPU/GPU usage on backend
const FRAME_INTERVAL = 100; // 100ms = ~10 FPS (recommended)
// const FRAME_INTERVAL = 50;  // 50ms = ~20 FPS (high quality)
// const FRAME_INTERVAL = 200; // 200ms = ~5 FPS (low bandwidth)
```

### 2. Image Quality

Adjust JPEG quality when capturing frames:

```typescript
// Lower quality = faster upload, less bandwidth
const base64Frame = canvas.toDataURL('image/jpeg', 0.7); // 70% quality
```

### 3. Resolution

Capture at lower resolution for better performance:

```typescript
const stream = await navigator.mediaDevices.getUserMedia({
  video: {
    width: { ideal: 640 },  // Lower = faster processing
    height: { ideal: 480 },
    facingMode: 'user'
  }
});
```

## Detected Classes

VisionFlow uses YOLOv4 trained on COCO dataset with 80 classes:

- **People**: person
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Objects**: backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
- **Food**: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- **Furniture**: chair, couch, potted plant, bed, dining table, toilet
- **Electronics**: tv, laptop, mouse, remote, keyboard, cell phone
- **Appliances**: microwave, oven, toaster, sink, refrigerator
- **Indoor**: book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## Troubleshooting

### Backend not responding

```bash
# Check if backend is running
curl http://localhost:5001/health

# Expected response:
# {"status":"healthy","service":"VisionFlow v2 Backend",...}
```

### CORS errors

The backend is configured with CORS allowing all origins. If you still get CORS errors, check your frontend is sending the correct `Content-Type: application/json` header.

### Camera permission denied

Make sure your browser has permission to access the camera. On macOS, check System Preferences → Security & Privacy → Camera.

### Slow processing

- Reduce frame rate (increase `FRAME_INTERVAL`)
- Lower image quality (reduce JPEG quality parameter)
- Lower capture resolution (640x480 recommended)
- Check backend logs for YOLOv4 inference time

## Example cURL Test

Test the backend with a sample frame:

```bash
# Capture a test frame (you'll need a base64 encoded image)
curl -X POST http://localhost:5001/process/frame \
  -H "Content-Type: application/json" \
  -d '{
    "frame": "data:image/jpeg;base64,YOUR_BASE64_IMAGE_HERE",
    "timestamp": 1730833245123
  }'
```

## Next Steps

1. Start the backend: `./start.sh`
2. Integrate the camera capture hook in your Next.js app
3. Add the `captureAndProcessFrame` utility
4. Create a component that displays video + detections
5. Test with your webcam!

---

**Backend Port**: 5001
**Frontend Expected**: Next.js/React with TypeScript
**Detection Format**: Compatible with your existing Detection interface
