#!/usr/bin/env python3
"""
Enhanced CNN for wound detection with sizing and location capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path

class WoundDetectionCNN(nn.Module):
    """Enhanced CNN that detects wounds and provides location/size information"""
    
    def __init__(self, num_classes=6):
        super(WoundDetectionCNN, self).__init__()
        
        # Shared feature extractor
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 32x32
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 16x16
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 8x8
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 4)  # [x_center, y_center, width, height]
        )
        
        # Segmentation head for precise area measurement
        self.segmentation_head = nn.Sequential(
            # Upsample back to original size
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),   # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2),   # 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),  # Binary mask: wound vs background
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classification
        wound_class = self.classifier(features)
        
        # Bounding box
        bbox = torch.sigmoid(self.bbox_regressor(features))  # Normalize to [0,1]
        
        # Segmentation mask
        mask = self.segmentation_head(features)
        
        return {
            'classification': wound_class,
            'bbox': bbox,
            'segmentation': mask
        }

class WoundAnalyzer:
    """Analyze wounds for size and location using enhanced CNN"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cpu')
        
        if model_path and Path(model_path).exists():
            self.model = self.load_model(model_path)
        else:
            self.model = WoundDetectionCNN()
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Wound type mapping
        self.wound_types = {
            0: "background",
            1: "diabetic_ulcer",
            2: "neuropathic_ulcer", 
            3: "pressure_ulcer",
            4: "surgical_wound",
            5: "venous_ulcer"
        }
    
    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = WoundDetectionCNN()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def analyze_wound(self, image_path: str, scale_mm_per_pixel: float = 0.1):
        """
        Comprehensive wound analysis
        
        Args:
            image_path: Path to wound image
            scale_mm_per_pixel: Calibration factor (mm per pixel)
            
        Returns:
            dict: Complete wound analysis
        """
        # Load and preprocess image
        original_image = cv2.imread(image_path)
        original_height, original_width = original_image.shape[:2]
        
        pil_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(pil_image).unsqueeze(0)
        
        # Model prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Extract results
        classification = torch.softmax(outputs['classification'], dim=1)
        confidence, predicted_class = torch.max(classification, 1)
        
        bbox = outputs['bbox'][0].numpy()  # [x_center, y_center, width, height]
        segmentation = outputs['segmentation'][0, 0].numpy()  # Binary mask
        
        # Convert normalized bbox to pixel coordinates
        x_center = bbox[0] * original_width
        y_center = bbox[1] * original_height
        width = bbox[2] * original_width
        height = bbox[3] * original_height
        
        # Calculate bounding box corners
        x1 = int(x_center - width/2)
        y1 = int(y_center - height/2)
        x2 = int(x_center + width/2)
        y2 = int(y_center + height/2)
        
        # Resize segmentation mask to original image size
        segmentation_resized = cv2.resize(segmentation, (original_width, original_height))
        
        # Calculate precise measurements from segmentation
        wound_pixels = np.sum(segmentation_resized > 0.5)
        wound_area_mm2 = wound_pixels * (scale_mm_per_pixel ** 2)
        
        # Calculate perimeter
        binary_mask = (segmentation_resized > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        perimeter_pixels = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter_pixels = cv2.arcLength(largest_contour, True)
        
        perimeter_mm = perimeter_pixels * scale_mm_per_pixel
        
        # Calculate equivalent diameter
        if wound_area_mm2 > 0:
            equivalent_diameter_mm = 2 * np.sqrt(wound_area_mm2 / np.pi)
        else:
            equivalent_diameter_mm = 0
        
        return {
            'wound_classification': {
                'type': self.wound_types[predicted_class.item()],
                'confidence': confidence.item(),
                'all_probabilities': {
                    self.wound_types[i]: prob for i, prob in enumerate(classification[0].numpy())
                }
            },
            'location': {
                'bounding_box': {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'center_x': x_center, 'center_y': y_center
                },
                'relative_position': {
                    'x_percent': (x_center / original_width) * 100,
                    'y_percent': (y_center / original_height) * 100
                }
            },
            'measurements': {
                'area_mm2': wound_area_mm2,
                'area_cm2': wound_area_mm2 / 100,
                'perimeter_mm': perimeter_mm,
                'perimeter_cm': perimeter_mm / 10,
                'equivalent_diameter_mm': equivalent_diameter_mm,
                'bounding_box_width_mm': width * scale_mm_per_pixel,
                'bounding_box_height_mm': height * scale_mm_per_pixel,
                'pixel_count': int(wound_pixels)
            },
            'image_info': {
                'original_width': original_width,
                'original_height': original_height,
                'scale_mm_per_pixel': scale_mm_per_pixel
            },
            'segmentation_available': True
        }
    
    def visualize_analysis(self, image_path: str, analysis: dict, output_path: str = None):
        """Create visualization of wound analysis"""
        image = cv2.imread(image_path)
        
        # Draw bounding box
        bbox = analysis['location']['bounding_box']
        cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 2)
        
        # Add classification text
        wound_type = analysis['wound_classification']['type']
        confidence = analysis['wound_classification']['confidence']
        area_cm2 = analysis['measurements']['area_cm2']
        
        text = f"{wound_type} ({confidence:.2f})"
        cv2.putText(image, text, (bbox['x1'], bbox['y1']-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        size_text = f"Area: {area_cm2:.1f} cm²"
        cv2.putText(image, size_text, (bbox['x1'], bbox['y2']+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image

def create_training_data_with_annotations():
    """Create training data format that includes bounding boxes for enhanced training"""
    
    training_format = {
        "images": [
            {
                "id": 1,
                "file_name": "wound_001.jpg",
                "width": 640,
                "height": 480
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,  # wound type
                "bbox": [100, 150, 200, 180],  # [x, y, width, height]
                "area": 36000,  # pixels
                "segmentation": [],  # polygon points for precise boundary
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 0, "name": "background"},
            {"id": 1, "name": "diabetic_ulcer"},
            {"id": 2, "name": "neuropathic_ulcer"},
            {"id": 3, "name": "pressure_ulcer"},
            {"id": 4, "name": "surgical_wound"},
            {"id": 5, "name": "venous_ulcer"}
        ]
    }
    
    return training_format

def main():
    """Demonstration of enhanced wound analysis"""
    print("Enhanced Wound Detection with Size & Location")
    print("=" * 45)
    
    # Initialize analyzer
    analyzer = WoundAnalyzer()
    
    # Example usage (would need trained model)
    test_image = "test_images_clean"
    
    if Path(test_image).exists():
        image_files = list(Path(test_image).glob("*.jpg"))
        
        if image_files:
            sample_image = str(image_files[0])
            print(f"Analyzing: {sample_image}")
            
            # Analyze wound
            analysis = analyzer.analyze_wound(sample_image, scale_mm_per_pixel=0.1)
            
            print("\nAnalysis Results:")
            print(f"Wound Type: {analysis['wound_classification']['type']}")
            print(f"Confidence: {analysis['wound_classification']['confidence']:.3f}")
            print(f"Area: {analysis['measurements']['area_cm2']:.2f} cm²")
            print(f"Perimeter: {analysis['measurements']['perimeter_cm']:.2f} cm")
            print(f"Location: ({analysis['location']['relative_position']['x_percent']:.1f}%, {analysis['location']['relative_position']['y_percent']:.1f}%)")
            
            # Create visualization
            output_image = analyzer.visualize_analysis(sample_image, analysis, "analyzed_wound.jpg")
            print("Visualization saved: analyzed_wound.jpg")
    
    print("\nCapabilities:")
    print("✓ Wound type classification")
    print("✓ Precise bounding box location")
    print("✓ Accurate area measurement")
    print("✓ Perimeter calculation")
    print("✓ Position coordinates")
    print("✓ Scale calibration support")

if __name__ == "__main__":
    main()