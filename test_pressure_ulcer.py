#!/usr/bin/env python3
"""
Test YOLO detection with pressure ulcer images specifically
"""

import requests
import base64
import json
from PIL import Image
import io
import os

def test_pressure_ulcer_detection():
    # Test with multiple confidence thresholds
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Create a darker red image that might look more like a pressure ulcer
    img = Image.new('RGB', (600, 400), color=(139, 69, 19))  # Dark red/brown
    
    # Add some darker spots
    pixels = img.load()
    for x in range(100, 200):
        for y in range(100, 200):
            pixels[x, y] = (80, 40, 40)  # Darker red
    
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    url = "http://localhost:8081/detect"
    
    for threshold in thresholds:
        print(f"\n=== Testing with threshold: {threshold} ===")
        
        # Update config
        config_url = "http://localhost:8081/config"
        config_payload = {"yolo_threshold": threshold}
        config_response = requests.post(config_url, json=config_payload)
        print(f"Config update: {config_response.json()}")
        
        # Test detection
        payload = {
            "image": img_base64,
            "confidence_threshold": threshold,
            "include_measurements": True,
            "detect_reference_objects": True,
            "force_method": "yolo"
        }
        
        try:
            response = requests.post(url, json=payload)
            result = response.json()
            print(f"Detections found: {len(result['detections'])}")
            if result['detections']:
                for i, detection in enumerate(result['detections']):
                    print(f"  Detection {i+1}: {detection['wound_class']} - {detection['confidence']:.3f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_pressure_ulcer_detection()