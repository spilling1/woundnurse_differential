#!/usr/bin/env python3
"""
Deploy Custom YOLO Model Script
Moves your best.pt model to the correct location and tests it
"""

import os
import shutil
import requests
import time
import json
from pathlib import Path

def deploy_model(source_path="best.pt"):
    """Deploy your trained YOLO model"""
    print("=== Deploying Custom YOLO Model ===")
    
    # Check if source model exists
    if not os.path.exists(source_path):
        print(f"✗ Source model not found: {source_path}")
        print("Please make sure your best.pt file is in the current directory")
        return False
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Copy model to correct location
    target_path = models_dir / "wound_yolo.pt"
    try:
        shutil.copy2(source_path, target_path)
        print(f"✓ Model copied to: {target_path}")
        
        # Check file size
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"  Model size: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"✗ Failed to copy model: {e}")
        return False
    
    # Update configuration to prioritize custom model
    config_file = "yolo_config.json"
    config = {
        "yolo_enabled": True,
        "auto_toggle": True,
        "yolo_threshold": 0.6,
        "color_threshold": 0.5,
        "performance_weight": 0.3,
        "model_path": "yolov8n.pt",
        "custom_model_path": "models/wound_yolo.pt",
        "training_enabled": False,
        "use_custom_model": True
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration updated: {config_file}")
    
    print("\n=== Next Steps ===")
    print("1. Restart the YOLO service to load your custom model")
    print("2. The service will automatically detect and use your trained model")
    print("3. Test the model with: python3 test_custom_model.py")
    
    return True

def test_model_loading():
    """Test if the custom model loads correctly"""
    print("\n=== Testing Custom Model Loading ===")
    
    try:
        from ultralytics import YOLO
        
        model_path = "models/wound_yolo.pt"
        if not os.path.exists(model_path):
            print("✗ Custom model not found")
            return False
        
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        print("✓ Custom model loaded successfully")
        
        # Test inference
        import numpy as np
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, conf=0.5, verbose=False)
        print("✓ Model inference test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def restart_service():
    """Instructions to restart the YOLO service"""
    print("\n=== Service Restart Instructions ===")
    print("The YOLO service needs to be restarted to load your custom model.")
    print("This will happen automatically when the main application restarts.")
    print("Your custom model will be loaded as the primary detection engine.")

if __name__ == "__main__":
    # Check if best.pt exists
    if os.path.exists("best.pt"):
        if deploy_model("best.pt"):
            test_model_loading()
            restart_service()
    else:
        print("Please upload your best.pt file to the root directory first")
        print("Then run: python3 deploy_custom_model.py")