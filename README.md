# 🚗 MareArts Road Objects Detection

[![PyPI version](https://badge.fury.io/py/marearts-road-objects.svg)](https://badge.fury.io/py/marearts-road-objects)
[![Downloads](https://pepy.tech/badge/marearts-road-objects)](https://pepy.tech/project/marearts-road-objects)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Windows](https://img.shields.io/badge/os-Windows-blue.svg)](https://www.microsoft.com/windows)
[![Linux](https://img.shields.io/badge/os-Linux-green.svg)](https://www.linux.org/)
[![macOS](https://img.shields.io/badge/os-macOS-silver.svg)](https://www.apple.com/macos/)

A high-performance Python package for road object detection. Detect persons, 4-wheeled vehicles, and 2-wheeled vehicles in images with advanced YOLO-based neural networks.

## ✨ Features

- 🚗 **Multi-class Detection**: Detects persons, cars/trucks, and motorcycles/bicycles
- ⚡ **GPU Acceleration**: NVIDIA CUDA, TensorRT, and DirectML support
- 🛠️ **CLI Interface**: Easy command-line tools (`marearts-robj` or `marearts-road-objects`)
- 📦 **Multiple Model Sizes**: Small (50MB), medium (100MB), large (200MB)
- 🌐 **Cross-platform**: Windows, macOS, and Linux support
- 🔑 **Unified License**: Same license works for both [MareArts-ANPR](https://github.com/MareArts/MareArts-ANPR) and Road Objects

## 🚀 Quick Start

### Installation

```bash
# Basic installation (CPU)
pip install marearts-road-objects

# With GPU acceleration (recommended)
pip install marearts-road-objects[gpu]          # NVIDIA
pip install marearts-road-objects[directml]     # Windows GPU
pip install marearts-road-objects[all-gpu]      # All GPU support
```

### Get Your License

**Subscribe**: [MareArts ANPR/LPR Solution](https://study.marearts.com/p/anpr-lpr-solution.html)  
**Note**: One license works for both ANPR and Road Objects packages!

### Configure License

```bash
# Interactive setup (recommended)
marearts-robj config

# Or set environment variables
export MAREARTS_ANPR_USERNAME="your-email@domain.com"
export MAREARTS_ANPR_SERIAL_KEY="your-serial-key"
```

### Basic Usage

```bash
# Detect objects in an image
marearts-robj detect traffic.jpg

# Use larger model with custom settings
marearts-robj detect highway.jpg --model large --confidence 0.7 --output result.jpg

# Check GPU acceleration
marearts-robj gpu-info
```

## 🐍 Python API

### Simple Detection

```python
import cv2
from marearts_road_objects import create_detector, download_model

# License credentials
username = "your-email@domain.com"
serial_key = "your-serial-key"

# Download and initialize detector
model_path = download_model("medium", username, serial_key)
detector = create_detector(model_path, username, serial_key, model_size="medium")

# Detect objects
image = cv2.imread("traffic_scene.jpg")
result = detector.detect(image)

# Print results
print(f"Processing time: {result['processing_time_ms']}ms")
print(f"Total objects: {result['total_objects']}")

for detection in result['detections']:
    print(f"{detection['id']}. {detection['class']} ({detection['subclass']})")
    print(f"   Confidence: {detection['confidence']}")
    print(f"   Bounding box: {detection['bbox']}")
```

### Combined with ANPR

```python
# Same license works for both packages!
from marearts_road_objects import create_detector, download_model
from marearts_anpr import ma_anpr_detector, ma_anpr_ocr, marearts_anpr_from_cv2

username = "your-email@domain.com"
serial_key = "your-serial-key"  # Same key for both!

# Initialize road objects detector
road_model = download_model("medium", username, serial_key)
road_detector = create_detector(road_model, username, serial_key, "medium")

# Initialize ANPR detector and OCR
anpr_detector = ma_anpr_detector("v11_middle", username, serial_key)
anpr_ocr = ma_anpr_ocr("v11_euplus", username, serial_key)

# Analyze traffic scene
image = cv2.imread("traffic.jpg")
vehicles = road_detector.detect(image)                    # Detect vehicles/persons
plates = marearts_anpr_from_cv2(anpr_detector, anpr_ocr, image)  # Detect and OCR license plates

print(f"Found {vehicles['total_objects']} road objects, {len(plates)} license plates")
```

## 📊 Output Format

The detection results come in a clean, structured JSON format:

```python
{
  "processing_time_ms": 45.2,        # Processing time in milliseconds
  "total_objects": 3,                # Number of detected objects
  "detections": [                    # List of detected objects
    {
      "id": 1,                       # Sequential object ID
      "class": "person",             # Main class (person, 4-wheels, 2-wheels)
      "subclass": "pedestrian",      # Specific subclass (pedestrian, car, truck, bike)
      "confidence": 0.89,            # Detection confidence (0.0 - 1.0)
      "bbox": [120, 150, 180, 280]   # Bounding box [x1, y1, x2, y2]
    },
    {
      "id": 2,
      "class": "4-wheels", 
      "subclass": "car",
      "confidence": 0.76,
      "bbox": [300, 200, 450, 320]
    }
  ]
}
```

## 🎯 Model Information

| Model | Speed | Accuracy | Size | Use Case |
|-------|-------|----------|------|----------|
| Small | Fastest | Good | 50MB | Real-time, mobile |
| Medium | Balanced | Better | 100MB | General purpose |
| Large | Slower | Best | 200MB | High accuracy needs |

**Detection Classes & Subclasses:**
- **person** (Pedestrians and people) → **pedestrian**
- **4-wheels** (Cars, trucks, buses, vans) → **car** (small) or **truck** (large)
- **2-wheels** (Motorcycles, bicycles, scooters) → **bike**

## 🛠️ CLI Reference

### Available Commands

```bash
marearts-robj config         # Configure license
marearts-robj gpu-info       # Check GPU support
marearts-robj detect IMAGE   # Detect objects
marearts-robj download       # Download models
marearts-robj validate       # Validate license
```

### Detection Examples

```bash
# Basic detection
marearts-robj detect image.jpg

# Advanced options
marearts-robj detect highway.jpg \
  --model large \
  --confidence 0.8 \
  --output detected_highway.jpg

# Batch processing
for img in *.jpg; do
  marearts-robj detect "$img" --output "detected_$img"
done
```

### Model Management

```bash
# Download specific models
marearts-robj download --model small
marearts-robj download --model large

# Check what's available
python -c "from marearts_road_objects import get_available_models; print(get_available_models())"
```

## ⚡ GPU Acceleration

### Check GPU Support

```bash
marearts-robj gpu-info
```

**Expected output with GPU:**
```
🚀 CUDAExecutionProvider (GPU)
⚡ CPUExecutionProvider
GPU Acceleration: ENABLED
```

### Performance Comparison

| Configuration | Small Model | Medium Model | Large Model |
|---------------|-------------|--------------|-------------|
| CPU (Intel i7) | ~100ms | ~200ms | ~400ms |
| NVIDIA RTX 3080 | ~15ms | ~25ms | ~45ms |
| DirectML (Windows) | ~30ms | ~60ms | ~120ms |

### GPU Requirements

**NVIDIA**: CUDA 11.8+ and cuDNN 8.6+  
**Windows DirectML**: Windows 10 v1903+ with compatible GPU  
**Memory**: 4GB+ GPU memory recommended for large models

## 💡 Code Examples

Ready-to-run examples are available in the [`examples/`](examples/) directory:

- **`basic_detection.py`** - Simple image detection
- **`combined_anpr_robj.py`** - Use both ANPR and Road Objects
- **`webcam_detection.py`** - Real-time webcam processing
- **`batch_processing.py`** - Process multiple images
- **`cli_examples.sh`** - Complete CLI usage guide

```bash
# Run an example
python examples/basic_detection.py
```

## 🆘 Support

- **License**: [Get your subscription](https://study.marearts.com/p/anpr-lpr-solution.html)
- **Issues**: [GitHub Issues](https://github.com/MareArts/MareArts-Road-Objects/issues)
- **Email**: hello@marearts.com

## 🔗 Related Packages

**MareArts AI Ecosystem** (same license for all):
- **[marearts-anpr](https://pypi.org/project/marearts-anpr/)** - License plate recognition
- **[marearts-crystal](https://pypi.org/project/marearts-crystal/)** - Licensing framework  
- **[marearts-xcolor](https://pypi.org/project/marearts-xcolor/)** - Color space conversions

---

**© 2024 MareArts. All rights reserved.**

*Get started with road object detection in minutes. One license, multiple AI packages, endless possibilities.*