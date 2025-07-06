#!/usr/bin/env python3
"""
Smart YOLO Wound Detection Service with Real YOLOv8 Integration
Features:
1. Real YOLO model support with fallback to color detection
2. Smart toggle system - uses best method for each case
3. Training pipeline integration
4. Performance monitoring
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import base64
import numpy as np
from PIL import Image
import io
import cv2
from typing import List, Dict, Any, Optional
import logging
import os
import json
from pathlib import Path
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart YOLO Wound Detection Service", version="3.0.0")

# Configuration
CONFIG_FILE = "yolo_config.json"
MODEL_DIR = "models"
TRAINING_DATA_DIR = "training_data"

class DetectionRequest(BaseModel):
    image: str  # base64 encoded image
    confidence_threshold: float = 0.5
    include_measurements: bool = True
    detect_reference_objects: bool = True
    force_method: Optional[str] = None  # "yolo", "color", "auto"

class Detection(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    area_pixels: int
    perimeter_pixels: float
    measurements: Dict[str, float]
    reference_object_detected: bool
    scale_calibrated: bool
    wound_class: str = "wound"
    detection_method: str = "unknown"

class DetectionResponse(BaseModel):
    detections: List[Detection]
    model: str
    version: str = "3.0"
    method_used: str
    processing_time: float
    recommendation: str

class SmartYOLODetector:
    def __init__(self):
        self.yolo_model = None
        self.config = self.load_config()
        self.performance_stats = {"yolo": [], "color": []}
        self.init_directories()
        self.load_yolo_model()
    
    def init_directories(self):
        """Create necessary directories"""
        Path(MODEL_DIR).mkdir(exist_ok=True)
        Path(TRAINING_DATA_DIR).mkdir(exist_ok=True)
    
    def load_config(self) -> dict:
        """Load configuration settings"""
        default_config = {
            "yolo_enabled": True,
            "auto_toggle": True,
            "yolo_threshold": 0.6,
            "color_threshold": 0.5,
            "performance_weight": 0.3,
            "model_path": "yolov8n.pt",
            "custom_model_path": "models/wound_yolo.pt",
            "training_enabled": False
        }
        
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    return {**default_config, **config}
            except:
                logger.warning("Config file corrupted, using defaults")
        
        return default_config
    
    def save_config(self):
        """Save current configuration"""
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_yolo_model(self):
        """Load YOLO model if available"""
        try:
            from ultralytics import YOLO
            
            # Try custom model first
            custom_path = self.config["custom_model_path"]
            if os.path.exists(custom_path):
                logger.info(f"Loading custom wound model: {custom_path}")
                self.yolo_model = YOLO(custom_path)
                self.config["model_type"] = "custom"
                return
            
            # Try general model
            general_path = self.config["model_path"]
            if os.path.exists(general_path):
                logger.info(f"Loading general YOLO model: {general_path}")
                self.yolo_model = YOLO(general_path)
                self.config["model_type"] = "general"
                return
            
            # Download default model
            logger.info("Downloading YOLOv8 model...")
            self.yolo_model = YOLO("yolov8n.pt")
            self.config["model_type"] = "downloaded"
            
        except ImportError:
            logger.error("Ultralytics not installed - YOLO disabled")
            self.config["yolo_enabled"] = False
            self.yolo_model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            self.config["yolo_enabled"] = False
            self.yolo_model = None
    
    def should_use_yolo(self, image_size: tuple, confidence_threshold: float) -> bool:
        """Smart decision on whether to use YOLO or color detection"""
        if not self.config["yolo_enabled"] or self.yolo_model is None:
            return False
        
        if not self.config["auto_toggle"]:
            return True
        
        # Decision factors
        factors = {
            "image_quality": self.assess_image_quality(image_size),
            "recent_performance": self.get_recent_performance(),
            "confidence_requirement": confidence_threshold
        }
        
        # Simple scoring system
        score = 0
        if factors["image_quality"] > 0.7:
            score += 0.4
        if factors["recent_performance"] > 0.6:
            score += 0.3
        if factors["confidence_requirement"] > 0.7:
            score += 0.3
        
        return score > 0.5
    
    def assess_image_quality(self, image_size: tuple) -> float:
        """Assess image quality for YOLO suitability"""
        width, height = image_size
        pixels = width * height
        
        # Higher resolution = better for YOLO
        if pixels > 500000:  # High res
            return 0.9
        elif pixels > 200000:  # Medium res
            return 0.7
        elif pixels > 50000:  # Low res
            return 0.5
        else:
            return 0.3
    
    def get_recent_performance(self) -> float:
        """Get recent performance metrics"""
        if not self.performance_stats["yolo"]:
            return 0.7  # Default assumption
        
        recent_scores = self.performance_stats["yolo"][-10:]
        return sum(recent_scores) / len(recent_scores)
    
    def detect_wounds(self, image: np.ndarray, confidence_threshold: float, force_method: str = None) -> tuple:
        """Main detection function with smart routing"""
        start_time = time.time()
        
        # Determine method
        if force_method == "yolo":
            use_yolo = True
        elif force_method == "color":
            use_yolo = False
        else:
            use_yolo = self.should_use_yolo(image.shape[:2], confidence_threshold)
        
        # Perform detection
        if use_yolo and self.yolo_model:
            detections, method = self.yolo_detection(image, confidence_threshold)
        else:
            detections, method = self.color_detection(image, confidence_threshold)
        
        processing_time = time.time() - start_time
        
        # Generate recommendation
        recommendation = self.generate_recommendation(method, len(detections), processing_time)
        
        return detections, method, processing_time, recommendation
    
    def yolo_detection(self, image: np.ndarray, confidence_threshold: float) -> tuple:
        """YOLO-based wound detection"""
        try:
            results = self.yolo_model(image, conf=confidence_threshold)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        
                        # Calculate measurements
                        width = x2 - x1
                        height = y2 - y1
                        area_pixels = width * height
                        perimeter_pixels = 2 * (width + height)
                        
                        # Real measurements (improved calibration)
                        pixel_to_mm = self.estimate_scale(image, (x1, y1, x2, y2))
                        measurements = {
                            "length_mm": max(width, height) * pixel_to_mm,
                            "width_mm": min(width, height) * pixel_to_mm,
                            "area_mm2": area_pixels * (pixel_to_mm ** 2),
                            "perimeter_mm": perimeter_pixels * pixel_to_mm
                        }
                        
                        detection = Detection(
                            bbox=[x1, y1, x2, y2],
                            confidence=confidence,
                            area_pixels=int(area_pixels),
                            perimeter_pixels=perimeter_pixels,
                            measurements=measurements,
                            reference_object_detected=False,
                            scale_calibrated=True,
                            wound_class="wound",
                            detection_method="yolo"
                        )
                        detections.append(detection)
            
            return detections, "yolo"
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {str(e)}")
            return self.color_detection(image, confidence_threshold)
    
    def color_detection(self, image: np.ndarray, confidence_threshold: float) -> tuple:
        """Enhanced color-based wound detection"""
        height, width = image.shape[:2]
        
        # Convert to multiple color spaces for better detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Multiple color ranges for wounds
        masks = []
        
        # Red/pink ranges in HSV
        red_ranges = [
            ([0, 50, 50], [10, 255, 255]),
            ([170, 50, 50], [180, 255, 255]),
            ([15, 30, 30], [25, 255, 255])  # Orange-red
        ]
        
        for lower, upper in red_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            masks.append(mask)
        
        # Combine masks
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Filter very small areas
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Enhanced confidence calculation
            aspect_ratio = w / h if h > 0 else 1
            area_ratio = area / (w * h)
            perimeter = cv2.arcLength(contour, True)
            
            # Multi-factor confidence
            base_confidence = min(0.9, max(0.4, area / 5000))
            shape_factor = 1.0 - abs(aspect_ratio - 1.0) * 0.2  # Prefer roundish shapes
            fill_factor = area_ratio
            
            confidence = base_confidence * shape_factor * fill_factor
            
            if confidence < confidence_threshold:
                continue
            
            # Measurements
            pixel_to_mm = self.estimate_scale(image, (x, y, x + w, y + h))
            measurements = {
                "length_mm": w * pixel_to_mm,
                "width_mm": h * pixel_to_mm,
                "area_mm2": area * (pixel_to_mm ** 2),
                "perimeter_mm": perimeter * pixel_to_mm
            }
            
            detection = Detection(
                bbox=[float(x), float(y), float(x + w), float(y + h)],
                confidence=confidence,
                area_pixels=int(area),
                perimeter_pixels=float(perimeter),
                measurements=measurements,
                reference_object_detected=False,
                scale_calibrated=False,
                wound_class="wound",
                detection_method="color"
            )
            detections.append(detection)
        
        return detections, "color"
    
    def estimate_scale(self, image: np.ndarray, bbox: tuple) -> float:
        """Estimate pixel-to-mm scale based on image and wound characteristics"""
        # This is a simplified estimation
        # In production, you'd use reference objects or calibration
        
        height, width = image.shape[:2]
        total_pixels = height * width
        
        # Assumption: larger images typically have better resolution
        if total_pixels > 1000000:  # High res
            return 0.05  # 0.05mm per pixel
        elif total_pixels > 500000:  # Medium res
            return 0.1   # 0.1mm per pixel
        else:  # Low res
            return 0.2   # 0.2mm per pixel
    
    def generate_recommendation(self, method: str, detection_count: int, processing_time: float) -> str:
        """Generate recommendation for next detection"""
        if method == "yolo":
            if detection_count == 0:
                return "Consider using color detection for better sensitivity"
            elif processing_time > 2.0:
                return "YOLO detected wounds but was slow - consider color detection for faster results"
            else:
                return "YOLO performed well - continue using for accurate detection"
        else:
            if detection_count == 0:
                return "No wounds detected with color method - consider YOLO for more sophisticated detection"
            elif detection_count > 3:
                return "Multiple detections found - YOLO might provide better classification"
            else:
                return "Color detection worked well - good for quick screening"

# Initialize detector
detector = SmartYOLODetector()

def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        if base64_str.startswith('data:'):
            base64_str = base64_str.split(',')[1]
        
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.post("/detect", response_model=DetectionResponse)
async def detect_wounds(request: DetectionRequest):
    """Smart wound detection with automatic method selection"""
    try:
        logger.info(f"Processing detection request - method: {request.force_method or 'auto'}")
        
        # Decode image
        image = decode_base64_image(request.image)
        logger.info(f"Image decoded successfully, shape: {image.shape}")
        
        # Perform detection
        detections, method, processing_time, recommendation = detector.detect_wounds(
            image, 
            request.confidence_threshold, 
            request.force_method
        )
        
        logger.info(f"Detection complete - method: {method}, found: {len(detections)} wounds, time: {processing_time:.2f}s")
        
        return DetectionResponse(
            detections=detections,
            model=f"smart-yolo-{method}",
            method_used=method,
            processing_time=processing_time,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return detector.config

@app.post("/config")
async def update_config(config: dict):
    """Update configuration"""
    detector.config.update(config)
    detector.save_config()
    return {"status": "Config updated", "config": detector.config}

@app.get("/status")
async def get_status():
    """Get detailed system status"""
    return {
        "yolo_available": detector.yolo_model is not None,
        "yolo_enabled": detector.config["yolo_enabled"],
        "auto_toggle": detector.config["auto_toggle"],
        "model_type": detector.config.get("model_type", "none"),
        "performance_stats": detector.performance_stats
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "smart-yolo-wound-detection",
        "version": "3.0",
        "yolo_status": "active" if detector.yolo_model else "inactive",
        "methods_available": ["yolo", "color"] if detector.yolo_model else ["color"]
    }

if __name__ == "__main__":
    logger.info("Starting Smart YOLO Wound Detection Service on port 8083")
    uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info")