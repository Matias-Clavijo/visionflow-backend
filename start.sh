#!/bin/bash
# VisionFlow v2 - Startup Script

set -e

echo "================================"
echo "Starting VisionFlow v2"
echo "================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"
echo ""

# Check if YOLOv4 weights exist
if [ ! -f "models/yolo/yolov4.weights" ]; then
    echo "WARNING: YOLOv4 weights file not found!"
    echo "Please download it:"
    echo "  cd models/yolo"
    echo "  wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ YOLOv4 weights found"
    echo ""
fi

# Start application
echo "================================"
echo "Starting VisionFlow v2..."
echo "================================"
echo ""
echo "Web Server will be available at:"
echo "  - REST API: http://localhost:5001"
echo "  - WebSocket: ws://localhost:5001"
echo "  - Health Check: http://localhost:5001/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd src/app
python3 main_with_web.py
