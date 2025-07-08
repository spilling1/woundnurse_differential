#!/usr/bin/env python3
"""
Test YOLO detection with a sample image to verify detection capability
"""

import requests
import base64
import json
from PIL import Image
import io

def test_yolo_detection():
    # Create a test image (simple colored square to test basic functionality)
    img = Image.new('RGB', (500, 500), color='red')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Test YOLO service
    url = "http://localhost:8081/detect"
    payload = {
        "image": img_base64,
        "confidence_threshold": 0.1,  # Very low threshold
        "include_measurements": True,
        "detect_reference_objects": True,
        "force_method": "yolo"
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Also test config
        config_response = requests.get("http://localhost:8081/config")
        print(f"Config: {config_response.json()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_yolo_detection()