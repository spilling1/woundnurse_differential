#!/usr/bin/env python3

"""
Test YOLO model loading directly
"""

import os
import sys
from pathlib import Path

# Add logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_yolo_loading():
    """Test YOLO model loading directly"""
    
    print("=== Testing YOLO Model Loading ===")
    
    # Test 1: Check if ultralytics is available
    try:
        from ultralytics import YOLO
        print("✓ ultralytics import successful")
    except ImportError as e:
        print(f"✗ ultralytics import failed: {e}")
        return False
    
    # Test 2: Check if model files exist
    model_files = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✓ {model_file} exists")
        else:
            print(f"✗ {model_file} not found")
    
    # Test 3: Try to load each model
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                print(f"\nTesting {model_file}...")
                model = YOLO(model_file)
                print(f"✓ {model_file} loaded successfully")
                
                # Test prediction on dummy image
                import numpy as np
                dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                results = model(dummy_image, conf=0.5)
                print(f"✓ {model_file} prediction test successful")
                
            except Exception as e:
                print(f"✗ {model_file} loading failed: {e}")
    
    # Test 4: Test actual SmartYOLODetector initialization
    print("\n=== Testing SmartYOLODetector ===")
    try:
        import json
        import time
        
        # Import the detector class
        sys.path.append('.')
        from yolo_smart_service import SmartYOLODetector
        
        detector = SmartYOLODetector()
        
        print(f"YOLO enabled: {detector.config.get('yolo_enabled')}")
        print(f"Model path: {detector.config.get('model_path')}")
        print(f"YOLO model object: {detector.yolo_model}")
        
        if detector.yolo_model is not None:
            print("✓ SmartYOLODetector model loaded successfully")
        else:
            print("✗ SmartYOLODetector model is None")
            
    except Exception as e:
        print(f"✗ SmartYOLODetector test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yolo_loading()