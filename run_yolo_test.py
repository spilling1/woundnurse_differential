#!/usr/bin/env python3
"""
YOLO Test Runner - Run wound detection on test dataset
"""

import os
import json
import base64
import requests
from pathlib import Path
import time
from PIL import Image
import io

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_wound_detection():
    """Test wound detection on sample images"""
    print("=== YOLO Wound Detection Test ===")
    
    # Check if service is running
    try:
        health_response = requests.get("http://localhost:8081/health", timeout=5)
        if health_response.status_code != 200:
            print("✗ YOLO service not running")
            return False
        print("✓ YOLO service is healthy")
    except:
        print("✗ YOLO service not responding")
        return False
    
    # Get test images
    test_images_dir = Path("wound_dataset_body_context/test/images")
    if not test_images_dir.exists():
        print("✗ Test images directory not found")
        return False
    
    # Get a few test images
    test_images = list(test_images_dir.glob("*.jpg"))[:5]  # Test first 5 images
    if not test_images:
        test_images = list(test_images_dir.glob("*.png"))[:5]
    
    if not test_images:
        print("✗ No test images found")
        return False
    
    print(f"✓ Found {len(test_images)} test images")
    
    # Test each image
    results = []
    for i, image_path in enumerate(test_images):
        print(f"\nTesting image {i+1}/{len(test_images)}: {image_path.name}")
        
        try:
            # Encode image
            image_b64 = encode_image_to_base64(image_path)
            
            # Prepare request
            request_data = {
                "image": image_b64,
                "confidence_threshold": 0.5,
                "include_measurements": True,
                "detect_reference_objects": True
            }
            
            # Send request
            start_time = time.time()
            response = requests.post(
                "http://localhost:8081/detect-wounds",
                json=request_data,
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                detection_count = len(result.get('detections', []))
                method = result.get('method_used', 'Unknown')
                processing_time = end_time - start_time
                
                print(f"  ✓ Detection successful")
                print(f"    - Method: {method}")
                print(f"    - Detections: {detection_count}")
                print(f"    - Processing time: {processing_time:.2f}s")
                
                # Show detection details
                for j, detection in enumerate(result.get('detections', [])):
                    confidence = detection.get('confidence', 0)
                    area = detection.get('area_pixels', 0)
                    print(f"    - Detection {j+1}: {confidence:.2f} confidence, {area} pixels")
                
                results.append({
                    'image': image_path.name,
                    'success': True,
                    'detections': detection_count,
                    'method': method,
                    'time': processing_time
                })
                
            else:
                print(f"  ✗ Detection failed: {response.status_code}")
                print(f"    Error: {response.text}")
                results.append({
                    'image': image_path.name,
                    'success': False,
                    'error': response.text
                })
                
        except Exception as e:
            print(f"  ✗ Error processing image: {e}")
            results.append({
                'image': image_path.name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n=== Test Summary ===")
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    print(f"Successful detections: {successful}/{total}")
    
    if successful > 0:
        avg_time = sum(r['time'] for r in results if r['success']) / successful
        total_detections = sum(r['detections'] for r in results if r['success'])
        print(f"Average processing time: {avg_time:.2f}s")
        print(f"Total wound detections: {total_detections}")
        
        # Method breakdown
        methods = {}
        for r in results:
            if r['success']:
                method = r['method']
                methods[method] = methods.get(method, 0) + 1
        
        print("Detection methods used:")
        for method, count in methods.items():
            print(f"  - {method}: {count} images")
    
    return successful > 0

if __name__ == "__main__":
    test_wound_detection()