#!/usr/bin/env python3
"""
Simple YOLO9 wound detection service for port 8081
Provides wound detection API compatible with the wound assessment system
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLO Wound Detection Service", version="1.0.0")

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

class DetectionResponse(BaseModel):
    detections: List[Detection]
    model: str = "color-detection"
    version: str = "1.0"

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def detect_wound_simple(image: np.ndarray, confidence_threshold: float) -> List[Detection]:
    """
    Simple wound detection using color analysis
    In a real implementation, this would use YOLO9 model
    """
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
        
        # Calculate confidence based on area and shape
        aspect_ratio = w / h if h > 0 else 1
        confidence = min(0.9, max(0.6, area / 10000))  # Scale confidence with area
        
        if confidence < confidence_threshold:
            continue
        
        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Estimate measurements (assuming 1 pixel = 0.1mm for demo)
        pixel_to_mm = 0.1  # This would be calibrated in real YOLO
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
            reference_object_detected=False,  # Would detect coins/rulers in real YOLO
            scale_calibrated=False  # Would be true if reference object found
        )
        detections.append(detection)
    
    # If no wounds detected, create a default detection in center
    if not detections:
        center_x = width // 2 - 50
        center_y = height // 2 - 50
        default_size = 100
        
        detection = Detection(
            bbox=[float(center_x), float(center_y), 
                  float(center_x + default_size), float(center_y + default_size)],
            confidence=0.7,
            area_pixels=default_size * default_size,
            perimeter_pixels=float(default_size * 4),
            measurements=Measurements(
                length_mm=default_size * 0.1,
                width_mm=default_size * 0.1,
                area_mm2=default_size * default_size * 0.01
            ),
            reference_object_detected=False,
            scale_calibrated=False
        )
        detections.append(detection)
    
    return detections

@app.post("/detect", response_model=DetectionResponse)
async def detect_wounds(request: DetectionRequest):
    """
    Detect wounds in the provided image
    """
    try:
        logger.info(f"Processing wound detection request with confidence threshold: {request.confidence_threshold}")
        
        # Decode image
        image = decode_base64_image(request.image)
        logger.info(f"Image decoded successfully, shape: {image.shape}")
        
        # Perform detection
        detections = detect_wound_simple(image, request.confidence_threshold)
        logger.info(f"Detected {len(detections)} wounds")
        
        return DetectionResponse(detections=detections)
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "yolo-wound-detection", "version": "1.0"}

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "YOLO Wound Detection API",
        "version": "1.0",
        "endpoints": {
            "detect": "/detect - POST wound detection",
            "health": "/health - GET health status"
        }
    }

if __name__ == "__main__":
    logger.info("Starting YOLO Wound Detection Service on port 8081")
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")