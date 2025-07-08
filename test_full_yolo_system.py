#!/usr/bin/env python3
"""
Test the full YOLO detection system with the wound classification service
"""

import requests
import base64
import json
from PIL import Image
import io

def test_full_system():
    print("=== Testing Full YOLO Detection System ===\n")
    
    # First, check YOLO service health
    health_response = requests.get("http://localhost:8081/health")
    print(f"YOLO Health: {health_response.json()}")
    
    # Check current config
    config_response = requests.get("http://localhost:8081/config")
    print(f"YOLO Config: {config_response.json()}")
    
    # Create a test image that might trigger wound detection
    img = Image.new('RGB', (500, 500), color=(120, 60, 60))  # Dark red
    
    # Add some variation
    pixels = img.load()
    for x in range(150, 350):
        for y in range(150, 350):
            pixels[x, y] = (80, 40, 40)  # Darker spot
    
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Test YOLO detection directly
    print("\n=== Direct YOLO Detection Test ===")
    yolo_payload = {
        "image": img_base64,
        "confidence_threshold": 0.05,
        "include_measurements": True,
        "detect_reference_objects": True,
        "force_method": "yolo"
    }
    
    yolo_response = requests.post("http://localhost:8081/detect", json=yolo_payload)
    yolo_result = yolo_response.json()
    print(f"YOLO Direct Result: {len(yolo_result['detections'])} detections")
    if yolo_result['detections']:
        for i, det in enumerate(yolo_result['detections']):
            print(f"  Detection {i+1}: {det['wound_class']} - {det['confidence']:.3f}")
    
    # Test with backend classifier (if available)
    print("\n=== Backend Classifier Test ===")
    try:
        # Create a mock request to the backend
        backend_payload = {
            "image": img_base64,
            "mimeType": "image/jpeg",
            "model": "gemini-2.5-pro"
        }
        
        # This would normally go through the backend routes
        print("Backend integration test would require full system...")
        
    except Exception as e:
        print(f"Backend test error: {e}")
    
    print("\n=== System Status Summary ===")
    print(f"YOLO Service: {'✓ Active' if health_response.status_code == 200 else '✗ Inactive'}")
    print(f"Detection Threshold: {config_response.json().get('yolo_threshold', 'Unknown')}")
    print(f"Direct Detection: {'✓ Working' if yolo_result['detections'] else '✗ No detections'}")

if __name__ == "__main__":
    test_full_system()