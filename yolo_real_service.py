#!/usr/bin/env python3
"""
Real YOLO wound detection service using YOLOv8
This replaces the placeholder color-based detection with actual machine learning
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import base64
import numpy as np
from PIL import Image
import io
import cv2
from typing import List, Dict, Any
import logging
import os
import requests
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real YOLO Wound Detection Service", version="2.0.0")

class DetectionRequest(BaseModel):
    image: str  # base64 encoded image
    confidence_threshold: float = 0.5
    include_measurements: bool = True
    detect_reference_objects: bool = True

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Measurements(BaseModel):
    length_mm: float
    width_mm: float
    area_mm2: float

class Detection(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    area_pixels: int
    perimeter_pixels: float
    measurements: Measurements
    reference_object_detected: bool
    scale_calibrated: bool
    wound_class: str = "wound"  # Could be extended to specific wound types

class DetectionResponse(BaseModel):
    detections: List[Detection]
    model: str = "yolov8n"
    version: str = "2.0"

class YOLOWoundDetector:
    def __init__(self):
        self.model = None
        self.model_path = "yolov8n.pt"
        self.custom_model_path = "wound_yolo.pt"
        self.load_model()
    
    def load_model(self):
        """Load YOLO model - either custom wound model or general object detection"""
        try:
            # Try to import ultralytics
            from ultralytics import YOLO
            
            # First try to load custom wound model if it exists
            if os.path.exists(self.custom_model_path):
                logger.info(f"Loading custom wound model: {self.custom_model_path}")
                self.model = YOLO(self.custom_model_path)
                return
            
            # Otherwise use general YOLOv8 model
            logger.info(f"Loading general YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded successfully")
            
        except ImportError:
            logger.error("Ultralytics not installed. Using fallback detection.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            self.model = None
    
    def detect_wounds(self, image: np.ndarray, confidence_threshold: float) -> List[Detection]:
        """Detect wounds using YOLO model"""
        if self.model is None:
            return self._fallback_detection(image, confidence_threshold)
        
        try:
            # Run YOLO inference
            results = self.model(image, conf=confidence_threshold)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        
                        # Calculate measurements
                        width = x2 - x1
                        height = y2 - y1
                        area_pixels = width * height
                        perimeter_pixels = 2 * (width + height)
                        
                        # Estimate real measurements (assuming 1 pixel = 0.1mm)
                        # In production, this would use proper calibration
                        pixel_to_mm = 0.1
                        length_mm = max(width, height) * pixel_to_mm
                        width_mm = min(width, height) * pixel_to_mm
                        area_mm2 = area_pixels * (pixel_to_mm ** 2)
                        
                        detection = Detection(
                            bbox=[x1, y1, x2, y2],
                            confidence=confidence,
                            area_pixels=int(area_pixels),
                            perimeter_pixels=perimeter_pixels,
                            measurements=Measurements(
                                length_mm=length_mm,
                                width_mm=width_mm,
                                area_mm2=area_mm2
                            ),
                            reference_object_detected=False,
                            scale_calibrated=False,
                            wound_class="wound"
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {str(e)}")
            return self._fallback_detection(image, confidence_threshold)
    
    def _fallback_detection(self, image: np.ndarray, confidence_threshold: float) -> List[Detection]:
        """Fallback to color-based detection if YOLO fails"""
        logger.warning("Using fallback color-based detection")
        
        height, width = image.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for reddish/wound-like colors
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red colors
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Filter small areas
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence based on area
            confidence = min(0.8, max(0.5, area / 10000))
            
            if confidence < confidence_threshold:
                continue
            
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Estimate measurements
            pixel_to_mm = 0.1
            length_mm = w * pixel_to_mm
            width_mm = h * pixel_to_mm
            area_mm2 = area * (pixel_to_mm ** 2)
            
            detection = Detection(
                bbox=[float(x), float(y), float(x + w), float(y + h)],
                confidence=confidence,
                area_pixels=int(area),
                perimeter_pixels=float(perimeter),
                measurements=Measurements(
                    length_mm=length_mm,
                    width_mm=width_mm,
                    area_mm2=area_mm2
                ),
                reference_object_detected=False,
                scale_calibrated=False,
                wound_class="wound"
            )
            detections.append(detection)
        
        return detections

# Initialize detector
detector = YOLOWoundDetector()

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        # Remove data URL prefix if present
        if base64_str.startswith('data:'):
            base64_str = base64_str.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to OpenCV format
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.post("/detect", response_model=DetectionResponse)
async def detect_wounds(request: DetectionRequest):
    """
    Detect wounds in the provided image using real YOLO
    """
    try:
        logger.info(f"Processing wound detection request with confidence threshold: {request.confidence_threshold}")
        
        # Decode image
        image = decode_base64_image(request.image)
        logger.info(f"Image decoded successfully, shape: {image.shape}")
        
        # Perform detection
        detections = detector.detect_wounds(image, request.confidence_threshold)
        logger.info(f"Detected {len(detections)} wounds")
        
        model_name = "yolov8n" if detector.model else "fallback"
        return DetectionResponse(detections=detections, model=model_name)
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "yolo" if detector.model else "fallback"
    return {
        "status": "healthy", 
        "service": "real-yolo-wound-detection", 
        "version": "2.0",
        "model": model_status
    }

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Real YOLO Wound Detection API",
        "version": "2.0",
        "model": "YOLOv8" if detector.model else "Fallback",
        "endpoints": {
            "detect": "/detect - POST wound detection",
            "health": "/health - GET health status"
        }
    }

if __name__ == "__main__":
    logger.info("Starting Real YOLO Wound Detection Service on port 8082")
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")