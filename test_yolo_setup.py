#!/usr/bin/env python3
"""
YOLO Test Setup and Configuration Validator
Tests the complete YOLO setup for wound detection
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dependencies():
    """Test if all required dependencies are available"""
    print("=== Testing Dependencies ===")
    
    dependencies = {
        'ultralytics': 'from ultralytics import YOLO',
        'opencv': 'import cv2',
        'torch': 'import torch',
        'PIL': 'from PIL import Image',
        'numpy': 'import numpy as np',
        'fastapi': 'from fastapi import FastAPI'
    }
    
    results = {}
    for name, import_cmd in dependencies.items():
        try:
            exec(import_cmd)
            print(f"  ✓ {name}")
            results[name] = True
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            results[name] = False
    
    return all(results.values())

def test_model_files():
    """Test if YOLO model files exist"""
    print("\n=== Testing Model Files ===")
    
    model_files = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    results = {}
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"  ✓ {model_file} ({size:.1f} MB)")
            results[model_file] = True
        else:
            print(f"  ✗ {model_file} missing")
            results[model_file] = False
    
    return any(results.values())  # At least one model should exist

def test_dataset_structure():
    """Test if training/test dataset is properly structured"""
    print("\n=== Testing Dataset Structure ===")
    
    # Check main dataset directory
    dataset_dir = Path("wound_dataset_body_context")
    if not dataset_dir.exists():
        print("  ✗ Dataset directory missing")
        return False
    
    # Check dataset summary
    summary_file = dataset_dir / "dataset_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"  ✓ Dataset summary: {summary['total_images']} total images")
        print(f"    - Training: {summary['splits']['train']} images")
        print(f"    - Validation: {summary['splits']['val']} images") 
        print(f"    - Test: {summary['splits']['test']} images")
    else:
        print("  ✗ Dataset summary missing")
        return False
    
    # Check class mapping
    class_file = dataset_dir / "class_mapping.json"
    if class_file.exists():
        with open(class_file, 'r') as f:
            classes = json.load(f)
        print(f"  ✓ Class mapping: {classes['num_classes']} classes")
        for i, class_name in enumerate(classes['class_names']):
            print(f"    - {i}: {class_name}")
    else:
        print("  ✗ Class mapping missing")
        return False
    
    # Check test images
    test_images_dir = dataset_dir / "test" / "images"
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        print(f"  ✓ Test images: {len(test_images)} files")
    else:
        print("  ✗ Test images directory missing")
        return False
    
    return True

def test_yolo_model_loading():
    """Test if YOLO models can be loaded"""
    print("\n=== Testing YOLO Model Loading ===")
    
    try:
        from ultralytics import YOLO
        
        # Test loading smallest model first
        model_files = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
        
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    print(f"  Testing {model_file}...")
                    model = YOLO(model_file)
                    print(f"    ✓ Model loaded successfully")
                    
                    # Test with dummy image
                    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
                    results = model(dummy_image, conf=0.5, verbose=False)
                    print(f"    ✓ Inference test passed")
                    
                    return True
                    
                except Exception as e:
                    print(f"    ✗ Failed to load {model_file}: {e}")
                    continue
        
        print("  ✗ No YOLO models could be loaded")
        return False
        
    except ImportError as e:
        print(f"  ✗ Could not import ultralytics: {e}")
        return False

def test_smart_service():
    """Test the smart YOLO service"""
    print("\n=== Testing Smart YOLO Service ===")
    
    service_file = "yolo_smart_service.py"
    if not os.path.exists(service_file):
        print("  ✗ Smart service file missing")
        return False
    
    try:
        # Import the service
        sys.path.insert(0, '.')
        from yolo_smart_service import SmartYOLODetector
        
        # Test initialization
        detector = SmartYOLODetector()
        print("  ✓ SmartYOLODetector initialized")
        
        # Check configuration
        print(f"    - YOLO enabled: {detector.config.get('yolo_enabled', False)}")
        print(f"    - Color detection enabled: {detector.config.get('color_detection_enabled', True)}")
        print(f"    - Model path: {detector.config.get('model_path', 'N/A')}")
        
        # Test with a dummy image
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections, method = detector.detect_wounds(dummy_image, 0.5)
        print(f"    ✓ Detection test passed using {method}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Smart service test failed: {e}")
        return False

def test_service_endpoint():
    """Test if the YOLO service is running"""
    print("\n=== Testing Service Endpoint ===")
    
    try:
        import requests
        
        # Test health endpoint
        response = requests.get("http://localhost:8081/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"  ✓ Service is running: {health_data['status']}")
            print(f"    - Model: {health_data.get('model', 'N/A')}")
            print(f"    - Method: {health_data.get('method', 'N/A')}")
            return True
        else:
            print(f"  ✗ Service returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ✗ Service not responding: {e}")
        return False

def create_test_report():
    """Create a comprehensive test report"""
    print("\n" + "="*50)
    print("YOLO TEST CONFIGURATION REPORT")
    print("="*50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Model Files", test_model_files),
        ("Dataset Structure", test_dataset_structure),
        ("YOLO Model Loading", test_yolo_model_loading),
        ("Smart Service", test_smart_service),
        ("Service Endpoint", test_service_endpoint)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n  ✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! YOLO setup is properly configured.")
        return True
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    create_test_report()