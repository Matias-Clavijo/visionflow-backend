#!/usr/bin/env python3
"""
YOLO11 Setup Script
Automatically downloads model, detects hardware, and exports to optimal format
"""

import os
import sys
import platform
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required packages are installed"""
    logger.info("=" * 60)
    logger.info("Checking dependencies...")
    logger.info("=" * 60)

    missing = []

    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__} installed")
    except ImportError:
        logger.error("✗ PyTorch not installed")
        missing.append("torch")

    try:
        import torchvision
        logger.info(f"✓ TorchVision {torchvision.__version__} installed")
    except ImportError:
        logger.error("✗ TorchVision not installed")
        missing.append("torchvision")

    try:
        import ultralytics
        logger.info(f"✓ Ultralytics {ultralytics.__version__} installed")
    except ImportError:
        logger.error("✗ Ultralytics not installed")
        missing.append("ultralytics")

    if missing:
        logger.error("\n" + "=" * 60)
        logger.error("Missing dependencies detected!")
        logger.error("=" * 60)
        logger.error("Please install missing packages:")
        logger.error(f"  pip3 install {' '.join(missing)}")
        logger.error("\nOr install all requirements:")
        logger.error("  pip3 install -r requirements.txt")
        return False

    logger.info("\n✓ All dependencies installed\n")
    return True


def detect_hardware():
    """Detect hardware and display recommendations"""
    logger.info("=" * 60)
    logger.info("Detecting Hardware...")
    logger.info("=" * 60)

    try:
        from src.app.core.processors.object_detector_yolo11 import detect_hardware as hw_detect
        hw_info = hw_detect()

        logger.info(f"Device: {hw_info['device_name']}")
        logger.info(f"Compute: {hw_info['device'].upper()}")
        logger.info(f"Recommended export format: {hw_info['export_format'] or 'None (use .pt)'}")
        logger.info(f"Recommended pool buffers: {hw_info['recommended_pool_buffers']}")
        logger.info(f"Recommended frame size: {hw_info['recommended_frame_size']}")

        return hw_info
    except Exception as e:
        logger.error(f"Error detecting hardware: {e}")
        return None


def setup_model_directory():
    """Create model directory if it doesn't exist"""
    model_dir = project_root / "models" / "yolo11"

    if not model_dir.exists():
        logger.info(f"Creating model directory: {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("✓ Model directory created")
    else:
        logger.info(f"✓ Model directory exists: {model_dir}")

    return model_dir


def download_model(model_dir):
    """Download YOLO11n model if it doesn't exist"""
    logger.info("\n" + "=" * 60)
    logger.info("Downloading YOLO11 Model...")
    logger.info("=" * 60)

    model_path = model_dir / "yolo11n.pt"

    if model_path.exists():
        logger.info(f"✓ Model already exists: {model_path}")
        logger.info(f"  Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        return model_path

    logger.info("Downloading YOLO11n.pt (~6MB)...")
    logger.info("This will download from Ultralytics GitHub releases...")

    try:
        from ultralytics import YOLO

        # YOLO class auto-downloads if model doesn't exist
        logger.info(f"Loading model (will auto-download to {model_path})...")
        model = YOLO("yolo11n.pt")

        # Move to our models directory
        import shutil
        from pathlib import Path

        # Find where ultralytics downloaded it (usually in home directory)
        home = Path.home()
        ultralytics_cache = home / ".ultralytics" / "models" / "yolo11n.pt"

        if ultralytics_cache.exists():
            logger.info(f"Copying model to {model_path}...")
            shutil.copy(ultralytics_cache, model_path)
            logger.info(f"✓ Model downloaded successfully")
            logger.info(f"  Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        else:
            logger.warning("Model downloaded by Ultralytics but path unknown")
            logger.info(f"You can manually download from:")
            logger.info("  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt")
            logger.info(f"And place it at: {model_path}")

        return model_path

    except Exception as e:
        logger.error(f"✗ Error downloading model: {e}")
        logger.info("\nManual download instructions:")
        logger.info("1. Download from: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt")
        logger.info(f"2. Place file at: {model_path}")
        return None


def export_model(model_path, hw_info):
    """Export model to optimal format for hardware"""
    if not hw_info or not hw_info['export_format']:
        logger.info("\n✓ No export needed (using .pt format)")
        return

    export_format = hw_info['export_format']

    logger.info("\n" + "=" * 60)
    logger.info(f"Exporting Model to {export_format.upper()}...")
    logger.info("=" * 60)
    logger.info("This may take a few minutes...")

    try:
        from ultralytics import YOLO

        model = YOLO(model_path)

        logger.info(f"Exporting to {export_format} format...")
        export_path = model.export(format=export_format, device=hw_info['device'])

        logger.info(f"✓ Model exported successfully")
        logger.info(f"  Export path: {export_path}")

        return export_path

    except Exception as e:
        logger.error(f"✗ Export failed: {e}")
        logger.warning("Will use standard .pt model instead")
        return None


def create_readme(model_dir):
    """Create README with instructions"""
    readme_path = model_dir / "README.md"

    content = """# YOLO11 Models Directory

This directory contains YOLO11 models and their exported versions.

## Files

- `yolo11n.pt` - Standard PyTorch model (~6MB)
- `yolo11n_openvino_model/` - Intel OpenVINO optimized (if on Intel CPU)
- `yolo11n.mlpackage/` - Apple CoreML optimized (if on Apple Silicon)
- `yolo11n.engine` - NVIDIA TensorRT optimized (if on CUDA GPU)

## Manual Download

If auto-download failed, download manually:

1. Go to: https://github.com/ultralytics/assets/releases
2. Download: `yolo11n.pt` (~6MB)
3. Place in this directory

## Supported Models

You can also use other YOLO11 variants:

| Model | Size | Speed | mAP |
|-------|------|-------|-----|
| yolo11n.pt | 6MB | Fastest | 37.3% |
| yolo11s.pt | 22MB | Fast | 44.9% |
| yolo11m.pt | 52MB | Medium | 50.2% |
| yolo11l.pt | 88MB | Slow | 52.9% |
| yolo11x.pt | 144MB | Slowest | 54.7% |

For continuous live inference, **yolo11n.pt is recommended**.

## Configuration

Edit `src/app/main_with_web.py` to change model:

```python
'object_detector_yolo11': {
    "model_path": "models/yolo11/yolo11n.pt",  # Change here
    ...
}
```

## Export Formats

The system auto-exports to optimal format:

- **MacBook M4/Apple Silicon**: CoreML (60-100 FPS)
- **Intel CPU**: OpenVINO (20-30 FPS)
- **NVIDIA GPU**: TensorRT (15-25 FPS on Jetson Nano)

## Class Filtering

Filter specific classes via API:

```bash
# Detect only persons
curl -X POST http://localhost:5001/detection/filter \\
  -H "Content-Type: application/json" \\
  -d '{"filter_classes": ["person"]}'

# Detect only cell phones
curl -X POST http://localhost:5001/detection/filter \\
  -H "Content-Type: application/json" \\
  -d '{"filter_classes": ["cell phone"]}'

# Detect both
curl -X POST http://localhost:5001/detection/filter \\
  -H "Content-Type: application/json" \\
  -d '{"filter_classes": ["person", "cell phone"]}'

# Detect all (clear filter)
curl -X POST http://localhost:5001/detection/filter \\
  -H "Content-Type: application/json" \\
  -d '{"filter_classes": null}'
```

## COCO Classes

YOLO11n is trained on COCO dataset with 80 classes:

- person (class 0)
- bicycle, car, motorcycle, airplane, bus, train, truck, boat
- cell phone (class 67)
- laptop, mouse, remote, keyboard
- ... and many more

See full list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
"""

    with open(readme_path, 'w') as f:
        f.write(content)

    logger.info(f"\n✓ README created: {readme_path}")


def main():
    """Main setup function"""
    logger.info("\n" + "=" * 60)
    logger.info("YOLO11 Setup Script")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.system()} {platform.machine()}")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info("")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Detect hardware
    hw_info = detect_hardware()

    # Setup model directory
    model_dir = setup_model_directory()

    # Download model
    model_path = download_model(model_dir)

    if not model_path or not model_path.exists():
        logger.error("\n✗ Model download failed")
        logger.info("\nSetup incomplete. Please download model manually.")
        sys.exit(1)

    # Export model to optimal format
    if hw_info:
        export_model(model_path, hw_info)

    # Create README
    create_readme(model_dir)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("✓ Setup Complete!")
    logger.info("=" * 60)
    logger.info(f"\nModel installed at: {model_path}")

    if hw_info:
        logger.info(f"\nHardware Configuration:")
        logger.info(f"  Device: {hw_info['device_name']}")
        logger.info(f"  Compute: {hw_info['device'].upper()}")
        logger.info(f"  Pool buffers: {hw_info['recommended_pool_buffers']}")
        logger.info(f"  Frame size: {hw_info['recommended_frame_size']}")

    logger.info(f"\nNext steps:")
    logger.info("1. Start backend: cd visionflow-v2 && python3 src/app/main_with_web.py")
    logger.info("2. Check health: curl http://localhost:5001/health")
    logger.info("3. Test filter: curl http://localhost:5001/detection/filter")
    logger.info("")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
