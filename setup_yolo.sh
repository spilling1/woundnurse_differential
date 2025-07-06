#!/bin/bash

echo "🚀 Setting up Real YOLO for Wound Detection"
echo "=============================================="

# Create virtual environment if it doesn't exist
if [ ! -d "yolo_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv yolo_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source yolo_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies one by one to handle conflicts
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Installing OpenCV..."
pip install opencv-python

echo "Installing other dependencies..."
pip install Pillow numpy fastapi uvicorn pydantic

# Try to install ultralytics
echo "Installing Ultralytics YOLO..."
pip install ultralytics

# Test installation
echo "Testing installation..."
python3 -c "
try:
    import torch
    print('✅ PyTorch installed successfully')
    print('PyTorch version:', torch.__version__)
    
    import cv2
    print('✅ OpenCV installed successfully')
    
    from ultralytics import YOLO
    print('✅ Ultralytics YOLO installed successfully')
    
    print('🎉 All dependencies installed successfully!')
    print('You can now run: python yolo_smart_service.py')
    
except ImportError as e:
    print('❌ Installation failed:', str(e))
    print('Try manual installation with: pip install ultralytics torch opencv-python')
"

echo "Setup complete! Virtual environment created at: yolo_env/"
echo "To activate: source yolo_env/bin/activate"
echo "To run service: python yolo_smart_service.py"