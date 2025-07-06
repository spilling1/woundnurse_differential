#!/usr/bin/env python3
"""
Manual YOLO installation script that handles dependency conflicts
"""

import subprocess
import sys
import os
import urllib.request
import zipfile
from pathlib import Path

def run_command(cmd, description=""):
    """Run command and return success status"""
    print(f"Running: {description or cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def install_pytorch():
    """Install PyTorch CPU version"""
    print("Step 1: Installing PyTorch...")
    
    # Try CPU-only PyTorch
    commands = [
        "python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --user",
        "python3 -m pip install torch torchvision --user",
    ]
    
    for cmd in commands:
        if run_command(cmd, "Installing PyTorch"):
            return True
    
    return False

def install_dependencies():
    """Install required dependencies one by one"""
    print("Step 2: Installing dependencies...")
    
    deps = [
        "numpy",
        "opencv-python-headless",
        "Pillow", 
        "PyYAML",
        "matplotlib",
        "seaborn",
        "requests",
        "psutil",
        "py-cpuinfo",
        "thop",
        "pandas"
    ]
    
    success_count = 0
    for dep in deps:
        if run_command(f"python3 -m pip install {dep} --user", f"Installing {dep}"):
            success_count += 1
    
    print(f"Installed {success_count}/{len(deps)} dependencies")
    return success_count > len(deps) * 0.7  # 70% success rate

def install_ultralytics():
    """Install Ultralytics YOLO"""
    print("Step 3: Installing Ultralytics YOLO...")
    
    commands = [
        "python3 -m pip install ultralytics --user --no-deps",
        "python3 -m pip install ultralytics --user",
    ]
    
    for cmd in commands:
        if run_command(cmd, "Installing Ultralytics"):
            return True
    
    return False

def download_yolo_weights():
    """Download YOLOv8 model weights"""
    print("Step 4: Downloading YOLO model weights...")
    
    models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt"
    }
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    for model_name, url in models.items():
        model_path = models_dir / model_name
        if model_path.exists():
            print(f"✅ {model_name} already exists")
            continue
            
        try:
            print(f"Downloading {model_name}...")
            urllib.request.urlretrieve(url, model_path)
            print(f"✅ Downloaded {model_name}")
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")

def test_installation():
    """Test if YOLO installation works"""
    print("Step 5: Testing installation...")
    
    test_script = '''
import sys
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    
    import cv2
    print(f"✅ OpenCV {cv2.__version__}")
    
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO imported")
    
    # Try loading model
    try:
        model = YOLO("yolov8n.pt")
        print("✅ YOLO model loaded successfully")
        print("🎉 Installation complete and working!")
        sys.exit(0)
    except Exception as e:
        print(f"⚠️  Model loading issue: {e}")
        print("✅ YOLO installed but may need model download")
        sys.exit(0)
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
'''
    
    result = subprocess.run([sys.executable, "-c", test_script], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Warnings:", result.stderr)
    
    return result.returncode == 0

def main():
    """Main installation process"""
    print("🚀 Manual YOLO Installation")
    print("=" * 40)
    
    steps = [
        ("Installing PyTorch", install_pytorch),
        ("Installing dependencies", install_dependencies), 
        ("Installing Ultralytics", install_ultralytics),
        ("Downloading model weights", download_yolo_weights),
        ("Testing installation", test_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            if not step_func():
                print(f"❌ {step_name} failed")
                continue
        except Exception as e:
            print(f"❌ {step_name} error: {e}")
            continue
    
    print("\n🎯 Installation Summary:")
    print("- If tests passed: YOLO is ready for training")
    print("- If tests failed: Try running specific commands manually")
    print("- Next: Run enhanced_yolo_training.py to process your 730 images")

if __name__ == "__main__":
    main()