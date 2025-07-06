#!/usr/bin/env python3

"""
Debug the SmartYOLODetector initialization issue
"""

import os
import sys
import json
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_detector():
    """Debug the detector initialization"""
    
    print("=== Debugging SmartYOLODetector ===")
    
    # Import the detector
    from yolo_smart_service import SmartYOLODetector
    
    # Create detector with debug
    print("Creating detector...")
    detector = SmartYOLODetector()
    
    print(f"Config: {detector.config}")
    print(f"YOLO enabled: {detector.config.get('yolo_enabled')}")
    print(f"Model path: {detector.config.get('model_path')}")
    print(f"Custom model path: {detector.config.get('custom_model_path')}")
    print(f"YOLO model: {detector.yolo_model}")
    print(f"YOLO model type: {type(detector.yolo_model)}")
    
    # Check if model file exists
    model_path = detector.config.get('model_path')
    if model_path and os.path.exists(model_path):
        print(f"✓ Model file exists: {model_path}")
    else:
        print(f"✗ Model file not found: {model_path}")
    
    # Try to manually load
    print("\nTrying manual YOLO loading...")
    try:
        from ultralytics import YOLO
        manual_model = YOLO(detector.config.get('model_path'))
        print("✓ Manual YOLO loading successful")
        print(f"Manual model type: {type(manual_model)}")
    except Exception as e:
        print(f"✗ Manual YOLO loading failed: {e}")
    
    # Test detection method choice
    print("\nTesting should_use_yolo...")
    image_size = (640, 480)
    confidence = 0.5
    should_use = detector.should_use_yolo(image_size, confidence)
    print(f"Should use YOLO: {should_use}")
    
    # Test with force method
    print("\nTesting with force_method='yolo'...")
    import numpy as np
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    try:
        detections, method, processing_time, recommendation = detector.detect_wounds(
            test_image, 0.5, force_method="yolo"
        )
        print(f"Detection method used: {method}")
        print(f"Number of detections: {len(detections)}")
    except Exception as e:
        print(f"Detection test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_detector()