# YOLO-UAV-Object-Tracking

🤖 **Real-time Object Detection and Tracking for UAV Applications**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorRT](https://img.shields.io/badge/TensorRT-7.1+-green.svg)](https://developer.nvidia.com/tensorrt)
[![Deep SORT](https://img.shields.io/badge/Deep-SORT-blue.svg)](https://github.com/nwojke/deep_sort)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

## 📋 Overview

This project implements **real-time object detection and tracking** for Unmanned Aerial Vehicles (UAVs) using:

- **YOLOv3** for object detection
- **TensorRT** for GPU-accelerated inference
- **Deep SORT** for multi-object tracking
- **Jetson NX** platform optimization

Designed for UAV applications including:
- 🎯 Aerial surveillance and monitoring
- 🔍 Infrastructure inspection
- 🚁 Autonomous navigation
- 📹 Real-time video analysis

## 🚀 Quick Start

### Prerequisites

- **Hardware**: NVIDIA Jetson NX or similar
- **Software**: 
  - Python 3.7+
  - CUDA 11.0+
  - TensorRT 7.1+
  - OpenCV 4.0+
  - PyCUDA

### Installation

```bash
# Clone the repository
git clone https://github.com/qzhou711/YOLO-UAV-Object-Tracking.git
cd YOLO-UAV-Object-Tracking

# Install Python dependencies
pip install -r yolo/requirements.txt

# Install additional dependencies
pip install pycuda==2019.1.1 onnx==1.4.1

# Compile TensorRT plugins
cd plugins
make

# Generate TensorRT engine
cd ../yolo
bash darknet2onnx.sh
bash onnx2trt.sh
```

### Basic Usage

```bash
# Image detection
python trt_yolo.py --image path/to/image.jpg -m yolov3-416

# Video detection with tracking
python trt_yolo_with_screen.py --video path/to/video.mp4 -m yolov3-416

# Camera input
python trt_yolo.py --video 0 -m yolov3-416
```

### Python API

```python
from utils.yolo_with_plugins import TrtYOLO
from utils.visualization import BBoxVisualization
import cv2

# Initialize YOLO model
model = TrtYOLO('yolov3-416', (416, 416), num_classes=1)

# Load image
img = cv2.imread('test.jpg')

# Run detection
boxes, confs, classes = model.detect(img, conf_thresh=0.3)

# Visualize
vis = BBoxVisualization(class_map)
result = vis.draw_bboxes(img, boxes, confs, classes)
cv2.imwrite('result.jpg', result)
```

## 📁 Project Structure

```
YOLO-UAV-Object-Tracking/
├── trt_yolo.py                    # Main detection script
├── trt_yolo_with_screen.py        # Detection with tracking visualization
├── eval_yolo.py                   # Model evaluation script
├── setup.py                       # Package setup
│
├── deep_sort/                     # Deep SORT tracking algorithm
│   ├── sort.py                    # SORT tracker implementation
│   ├── tracker.py                 # Deep SORT tracker
│   ├── detection.py               # Detection class
│   ├── kalman_filter.py           # Kalman filter for tracking
│   ├── nn_matching.py             # Nearest neighbor matching
│   ├── linear_assignment.py       # Hungarian algorithm
│   └── iou_matching.py            # IoU-based matching
│
├── utils/                         # Utility functions
│   ├── yolo_with_plugins.py       # YOLO with TensorRT plugins
│   ├── yolo_classes.py            # Class name mappings
│   ├── camera.py                  # Camera input handling
│   ├── display.py                 # Window management
│   ├── visualization.py           # Bounding box visualization
│   └── preprocessing.py           # Image preprocessing
│
├── yolo/                          # YOLO model files
│   ├── darknet2onnx/              # Darknet to ONNX conversion
│   ├── onnx2trt/                  # ONNX to TensorRT conversion
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # YOLO module documentation
│
├── plugins/                       # TensorRT plugins
│   ├── libyolo_layer.so           # Compiled plugin
│   └── README.md
│
└── README.md                      # This file
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    YOLO-UAV-Object-Tracking                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Input      │───>│   YOLOv3     │───>│   TensorRT   │  │
│  │  (Video/Img) │    │   Detector   │    │  Optimized   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                           │                      │          │
│                           v                      v          │
│                    ┌──────────────┐    ┌──────────────┐   │
│                    │   BBox       │    │   Deep SORT  │   │
│                    │  Drawing     │<───│   Tracker    │   │
│                    └──────────────┘    └──────────────┘   │
│                           │                      │          │
│                           v                      v          │
│                    ┌──────────────────────────────────┐   │
│                    │         Output Display            │   │
│                    │  (BBoxes + Track IDs + FPS)       │   │
│                    └──────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ Configuration

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | yolov3-416 | YOLO model name (yolov3-416, yolov3-608) |
| `category_num` | 1 | Number of object classes |
| `conf_th` | 0.3 | Confidence threshold for detection |
| `nms_th` | 0.5 | Non-Maximum Suppression threshold |

### Supported Models

- **yolov3-416**: 416x416 input, balanced speed/accuracy
- **yolov3-608**: 608x608 input, higher accuracy

## 📊 Performance

Typical performance on Jetson NX:

| Metric | Value |
|--------|-------|
| FPS | 15-30 |
| Inference Time | 30-60ms |
| Memory Usage | 2-4 GB |

*Note: Performance varies based on model size and input resolution.*

## 🎯 Supported Classes

Default configuration supports:
- Insulator detection (0: 'insulator')

Modify `utils/yolo_classes.py` to add custom classes:

```python
CLASS_MAP = {
    0: 'insulator',
    1: 'person',
    2: 'car',
    3: 'drone',
    # Add more classes...
}
```

## 🔧 Advanced Usage

### Deep SORT Tracking

```python
from deep_sort import DeepSortTracker
from utils.yolo_with_plugins import TrtYOLO

# Initialize
yolo = TrtYOLO('yolov3-416', (416, 416), num_classes=1)
tracker = DeepSortTracker(max_dist=0.2, max_iou_distance=0.7)

# Process frame
boxes, confs, classes = yolo.detect(frame, conf_thresh=0.3)
tracks = tracker.update(boxes, confs, classes, frame)

# Get track IDs
for track in tracks:
    track_id = track.track_id
    bbox = track.bbox
```

### TensorRT Engine Generation

```bash
# Convert Darknet to ONNX
cd yolo/darknet2onnx
python yolo_to_onnx.py

# Convert ONNX to TensorRT
cd ../onnx2trt
python onnx2trt.py -o yolov3.onnx -e yolov3.trt -p FP16
```

## 📝 Keyboard Controls

| Key | Action |
|-----|--------|
| `ESC` | Quit program |
| `F` | Toggle fullscreen |
| Any key | Exit on key press |

## ⚠️ Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce input resolution (use yolov3-416 instead of yolov3-608)
   - Close other GPU processes

2. **TensorRT engine not found**
   - Generate engine: `bash yolo/darknet2onnx.sh && bash yolo/onnx2trt.sh`
   - Check model path: `yolo/darknet/{model}.trt`

3. **Camera not opening**
   - Check camera index: try `--video 0` or `--video 1`
   - Verify camera is connected and drivers installed

### Performance Optimization

- Use FP16 precision for faster inference
- Enable TensorRT optimization
- Reduce input resolution if needed
- Use Jetson Power Mode 15W (MAXN)

## 📚 References

- [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Deep SORT](https://github.com/nwojke/deep_sort)
- [Jetson TX2/ NX](https://developer.nvidia.com/embedded/jetson-nx-developer-kit)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## 📄 License

This project is licensed under the GPL v3 License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project is designed for UAV (Drone) applications. Ensure compliance with local regulations when deploying for real-world use.
