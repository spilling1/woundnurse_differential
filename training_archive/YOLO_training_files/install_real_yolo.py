#!/usr/bin/env python3
"""
Installation script for real YOLO wound detection
This script will install the necessary dependencies and set up the real YOLO service
"""

import subprocess
import sys
import os
import urllib.request
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        print(f"Success: {result.stdout}")
        return True
    except Exception as e:
        print(f"Error running command: {str(e)}")
        return False

def install_dependencies():
    """Install required Python packages"""
    print("Installing YOLO dependencies...")
    
    # Try different installation methods
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "ultralytics",
        "opencv-python",
        "pillow",
        "numpy"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"Failed to install {package}")
            return False
    
    return True

def download_yolo_model():
    """Download YOLOv8 model weights"""
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    model_path = "yolov8n.pt"
    
    if os.path.exists(model_path):
        print(f"Model {model_path} already exists")
        return True
    
    print(f"Downloading YOLOv8 model to {model_path}...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully")
        return True
    except Exception as e:
        print(f"Failed to download model: {str(e)}")
        return False

def test_installation():
    """Test if YOLO installation works"""
    print("\nTesting YOLO installation...")
    test_code = """
import torch
from ultralytics import YOLO
print("PyTorch version:", torch.__version__)
model = YOLO('yolov8n.pt')
print("YOLO model loaded successfully")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ YOLO installation successful!")
            print(result.stdout)
            return True
        else:
            print("❌ YOLO installation failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Main installation process"""
    print("🚀 Installing Real YOLO Wound Detection System")
    print("=" * 50)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Step 2: Download model
    if not download_yolo_model():
        print("❌ Failed to download YOLO model")
        return False
    
    # Step 3: Test installation
    if not test_installation():
        print("❌ Installation test failed")
        return False
    
    print("\n🎉 Real YOLO installation completed successfully!")
    print("\nNext steps:")
    print("1. Run: python yolo_real_service.py")
    print("2. Update your system to use port 8082 instead of 8081")
    print("3. Test with: curl http://localhost:8082/health")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)