# VisionFlow v2

Real-time object detection system with YOLOv4 and Flask/SocketIO web server for live streaming to web frontends.

## Features

- 🎥 Real-time RTSP camera stream processing
- 🤖 YOLOv4 object detection (80 COCO classes)
- 🚀 High-performance multiprocessing architecture with shared memory
- 🌐 Flask/SocketIO web server for real-time detection broadcasting
- 📹 Event-triggered video clip generation
- ☁️ Cloud storage integration (Backblaze B2)
- 📊 MongoDB metadata persistence
- 🔧 Hardware acceleration support (CUDA, OpenVINO)

## Quick Start

### 1. Download YOLOv4 Weights

```bash
cd models/yolo
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
cd ../..
```

### 2. Install Dependencies

```bash
# Automatic (creates venv and installs dependencies)
./start.sh

# Or manual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure RTSP Camera

Edit `src/app/main_with_web.py`:

```python
'rtsp_capturer': {
    'rtsp_url': 'rtsp://YOUR_CAMERA_IP:8554/stream',  # Change this
    'target_width': 640,
    'target_height': 480,
}
```

### 4. Run the Application

```bash
./start.sh
```

The server will start on:
- **REST API**: http://localhost:5000
- **WebSocket**: ws://localhost:5000
- **Health Check**: http://localhost:5000/health

## Frontend Integration (Next.js)

Your Next.js frontend can connect automatically:

### 1. Set environment variable (`.env.local`):

```bash
NEXT_PUBLIC_WS_URL=http://localhost:5000
```

### 2. Start frontend:

```bash
npm run dev
```

### 3. Detection Format

The backend broadcasts detections in this format:

```typescript
interface Detection {
  id: string;              // "{timestamp}-{class}"
  class: string;           // "person", "car", etc.
  confidence: number;      // 0.0 - 1.0
  bbox: [x, y, w, h];     // Bounding box
  timestamp: number;       // Unix timestamp (ms)
}
```

## Architecture

```
RTSP Camera → Capturer → [Shared Memory Pool] → Processor (YOLOv4)
                                                      ↓
                                              WebSocket Broadcast
                                                      ↓
                                                Next.js Frontend
                                                      ↓
                                              (Real-time display)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and stats |
| `/stats` | GET | Server statistics |
| `/detections/latest` | GET | Latest detection results |
| `/` | GET | API information |

## WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client → Server | Client connection |
| `disconnect` | Client → Server | Client disconnection |
| `detections` | Server → Client | Real-time detection broadcast |
| `ping` | Client → Server | Keep-alive |
| `pong` | Server → Client | Keep-alive response |

## Project Structure

```
visionflow-v2/
├── src/app/
│   ├── main.py                    # Detection pipeline only
│   ├── main_with_web.py           # Pipeline + Web server (use this!)
│   ├── web_server.py              # Flask/SocketIO server
│   ├── core/
│   │   ├── capturers/             # RTSP/WebCam capture
│   │   ├── processors/            # YOLOv4 detection
│   │   ├── orchestrators/         # Pipeline coordination
│   │   └── events_manager/        # Video clip generation
│   └── models/                    # Data structures
├── models/yolo/
│   ├── yolov4.weights            # Download separately
│   ├── yolov4.cfg                # Model architecture
│   └── coco.names                # 80 COCO classes
├── requirements.txt              # Python dependencies
├── start.sh                      # Quick start script
└── CLAUDE.md                     # Detailed documentation
```

## Configuration

Edit `src/app/main_with_web.py` to configure:

### RTSP Camera

```python
'rtsp_capturer': {
    'rtsp_url': 'rtsp://192.168.68.54:8554/preview',
    'target_width': 640,
    'target_height': 480,
    'frame_skip': 2,
}
```

### Object Detection

```python
'object_detector': {
    'model_path': 'models/yolo/yolov4.weights',
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'process_every_n_frames': 1,
}
```

### Web Server

```python
'web_server': {
    'enabled': True,
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
}
```

### Cloud Storage (Optional)

```python
'video_clip_generator': {
    'use_cloud_storage': True,
    'b2_bucket_name': 'visionflow-v1',
    'use_mongodb': True,
    'mongo_uri': 'mongodb://...',
}
```

## Performance Tuning

| Parameter | Impact | Recommendation |
|-----------|--------|----------------|
| `frame_skip` | Higher = lower CPU usage | 2 (process every 2nd frame) |
| `process_every_n_frames` | Higher = faster | 1 for real-time |
| `target_width/height` | Higher = better quality | 640x480 for balance |
| `confidence_threshold` | Higher = fewer false positives | 0.5 default |
| `broadcast_every_n_frames` | Higher = less bandwidth | 2 (broadcast every 2nd) |

## Testing

### 1. Test Backend Health

```bash
curl http://localhost:5000/health
```

### 2. Test WebSocket

```bash
# Install websocat
brew install websocat

# Connect
websocat ws://localhost:5000/socket.io/?EIO=4&transport=websocket
```

### 3. Test with Frontend

```bash
# Terminal 1: Backend
./start.sh

# Terminal 2: Frontend
cd /path/to/nextjs-frontend
npm run dev

# Open http://localhost:3000
```

## Troubleshooting

### Missing YOLOv4 Weights

```bash
cd models/yolo
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

### RTSP Connection Failed

- Verify camera IP and RTSP URL
- Check network connectivity
- Test with VLC: `vlc rtsp://YOUR_CAMERA_IP:8554/stream`

### Frontend Can't Connect

- Ensure backend is running: `curl http://localhost:5000/health`
- Check `NEXT_PUBLIC_WS_URL` in frontend `.env.local`
- Verify port 5000 is not blocked by firewall

### Low FPS / High CPU

- Increase `frame_skip` (e.g., `frame_skip: 3`)
- Increase `process_every_n_frames` (e.g., `2` or `3`)
- Reduce `target_width/height` (e.g., `320x240`)
- Enable CUDA if you have a GPU

## Requirements

- Python 3.8+
- OpenCV 4.8+
- CUDA-compatible GPU (optional, for acceleration)
- MongoDB (optional, for metadata storage)
- Backblaze B2 account (optional, for cloud storage)

## License

This project is for educational purposes (Trabajo de Grado - Ingeniería Informática).

## Documentation

See [CLAUDE.md](CLAUDE.md) for comprehensive documentation including:
- Detailed architecture explanation
- Multiprocessing implementation
- Frontend integration guide
- Performance optimization tips
- Troubleshooting guide

## Credits

- **YOLOv4**: [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- **COCO Dataset**: [cocodataset.org](https://cocodataset.org/)
- **OpenCV**: [opencv.org](https://opencv.org/)

---

**Developed for**: Trabajo de Grado - Computer Vision en Tiempo Real
**Technology Stack**: Python • OpenCV • YOLOv4 • Flask • SocketIO • MongoDB • Backblaze B2
