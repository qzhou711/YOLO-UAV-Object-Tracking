#!/usr/bin/env python3
"""
YOLO-UAV-Object-Tracking

Real-time object detection and tracking on UAV (Unmanned Aerial Vehicle) platforms
using YOLOv3 with TensorRT optimization and Deep SORT tracking algorithm.

This project is designed to run on NVIDIA Jetson NX for real-time UAV applications
such as surveillance, inspection, and autonomous navigation.

Author: qzhou711
Date: 2020-10-26
"""

import os
import time
import argparse

import cv2
import numpy as np

# CUDA initialization
import pycuda.autoinit  # noqa: F401 - Required for CUDA driver initialization
import pycuda.driver as cuda

from multiprocessing import Process, Queue
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO


# =============================================================================
# Configuration
# =============================================================================

# Class ID to name mapping (default: insulator detection)
CLASS_NAME_MAP = {
    0: 'insulator',
}

# Window name for display
WINDOW_NAME = 'TrtYOLODemo'

# Set CUDA visible device (single GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """
    Parse command line arguments for YOLO detection.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    desc = (
        'Real-time object detection and tracking with TensorRT YOLO on Jetson NX. '
        'Supports video streams, camera input, and image files.'
    )

    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)

    parser.add_argument(
        '-c', '--category_num',
        type=int,
        default=1,
        help='Number of object categories (default: 1)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='yolov3-416',
        help='YOLO model name (default: yolov3-416)'
    )
    parser.add_argument(
        '-t', '--conf_th',
        type=float,
        default=0.3,
        help='Confidence threshold for detection (default: 0.3)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input source: image path, video path, or camera index'
    )

    return parser.parse_args()


# =============================================================================
# Detection Loop
# =============================================================================

def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """
    Continuously capture frames and perform object detection.

    This function runs in a loop, capturing images from the camera/source
    and running YOLO detection on each frame. Detection results are
    visualized with bounding boxes.

    Args:
        cam: Camera or video source object with read() method
        trt_yolo: TensorRT optimized YOLO detector instance
        conf_th: Confidence threshold for filtering detections
        vis: BBoxVisualization instance for drawing boxes
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()

    while True:
        # Capture frame
        img = cam.read()
        if img is None:
            break

        # Run YOLO detection
        boxes, confs, clss = trt_yolo.detect(img, conf_th)

        # Visualize results
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)

        # Save result image
        cv2.imwrite("./result.jpg", img)

        # Calculate FPS
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # Exponential moving average for smooth FPS display
        fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
        tic = toc

        # Exit on any key press
        key = cv2.waitKey(1)
        if key > 0:
            break


def yolo_detection(args):
    """
    Main detection pipeline.

    Initializes the YOLO model and camera, then runs the detection loop.

    Args:
        args: Parsed command line arguments
    """
    # Validate category number
    if args.category_num <= 0:
        raise SystemExit(f'ERROR: invalid category_num ({args.category_num})!')

    # Check model file exists
    model_path = f'yolo/darknet/{args.model}.trt'
    if not os.path.isfile(model_path):
        raise SystemExit(f'ERROR: model file not found ({model_path})!')

    # Get class dictionary
    cls_dict = get_cls_dict(args.category_num)

    # Parse model dimensions from name (e.g., 'yolov3-416' -> 416x416)
    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit(f'ERROR: invalid yolo_dim ({yolo_dim})!')
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)

    # Validate dimensions are multiples of 32 (YOLO requirement)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit(f'ERROR: yolo_dim must be multiple of 32 ({yolo_dim})!')

    # Initialize camera
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # Initialize TensorRT YOLO model
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)

    # Initialize visualization
    vis = BBoxVisualization(cls_dict)

    # Run detection loop
    loop_and_detect(cam, trt_yolo, conf_th=args.conf_th, vis=vis)

    # Cleanup
    cam.release()


# =============================================================================
# Main Entry Point
# =============================================================================

class DetectionProcess(Process):
    """
    Multiprocessing wrapper for YOLO detection.

    Runs detection in a separate process for better resource management
    on Jetson platforms.
    """

    def __init__(self):
        """Initialize the detection process."""
        super().__init__()
        self.args = None

    def run(self):
        """Execute the detection pipeline."""
        print('[INFO] Detection process started')
        yolo_detection(self.args)
        print('[INFO] Detection process finished')


def main():
    """
    Main entry point for YOLO object detection.

    Parses arguments and starts the detection process.
    """
    args = parse_args()
    print(f'[INFO] Arguments: {args}')

    # Validate input
    if args.category_num <= 0:
        raise SystemExit(f'ERROR: invalid category_num ({args.category_num})!')

    # Check model file
    model_path = f'yolo/darknet/{args.model}.trt'
    if not os.path.isfile(model_path):
        raise SystemExit(f'ERROR: model file not found ({model_path})!')

    # Get class dictionary
    cls_dict = get_cls_dict(args.category_num)

    # Parse model dimensions
    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit(f'ERROR: invalid yolo_dim ({yolo_dim})!')
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)

    # Validate dimensions
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit(f'ERROR: yolo_dim must be multiple of 32 ({yolo_dim})!')

    # Initialize camera
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # Initialize YOLO model
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)

    # Initialize visualization
    vis = BBoxVisualization(cls_dict)

    # Run detection
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis)

    # Cleanup
    cam.release()


if __name__ == '__main__':
    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set

    main()
