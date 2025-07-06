#!/usr/bin/env python3
"""
Enhanced YOLO Training Pipeline with Body Map Integration
Leverages detailed body image maps and labels for superior wound detection training
"""

import json
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class BodyMapYOLOTraining:
    def __init__(self, dataset_path: str = "enhanced_wound_dataset"):
        self.dataset_path = Path(dataset_path)
        self.body_regions = {
            "head": 0, "neck": 1, "chest": 2, "abdomen": 3,
            "back": 4, "shoulder": 5, "arm": 6, "forearm": 7,
            "hand": 8, "hip": 9, "thigh": 10, "knee": 11,
            "leg": 12, "ankle": 13, "foot": 14, "heel": 15
        }
        
        self.enhanced_classes = [
            # Wound type + body region combinations
            "pressure_ulcer_heel",
            "pressure_ulcer_sacrum", 
            "pressure_ulcer_shoulder",
            "diabetic_foot_ulcer",
            "venous_leg_ulcer",
            "surgical_wound_chest",
            "surgical_wound_abdomen",
            "arterial_ulcer_leg"
        ]
        
        self.setup_enhanced_directories()
    
    def setup_enhanced_directories(self):
        """Create enhanced directory structure"""
        dirs = [
            "images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test",
            "body_maps", "enhanced_labels", "models", "results",
            "anatomical_context", "validation_sets"
        ]
        
        for dir_path in dirs:
            (self.dataset_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    def process_body_map_data(self, body_map_file: str, labels_file: str):
        """
        Process your detailed body map and labels
        
        Args:
            body_map_file: Path to your body image map
            labels_file: Path to your detailed labels file
        """
        logger.info("Processing body map data with anatomical context...")
        
        # Load body map data
        with open(labels_file, 'r') as f:
            if labels_file.endswith('.json'):
                labels_data = json.load(f)
            else:
                # Handle other formats (CSV, XML, etc.)
                labels_data = self.parse_labels_file(labels_file)
        
        enhanced_annotations = []
        
        for image_data in labels_data:
            enhanced_annotation = self.enhance_annotation_with_anatomy(image_data)
            enhanced_annotations.append(enhanced_annotation)
        
        # Save enhanced annotations
        output_file = self.dataset_path / "enhanced_labels" / "processed_annotations.json"
        with open(output_file, 'w') as f:
            json.dump(enhanced_annotations, f, indent=2)
        
        logger.info(f"Enhanced {len(enhanced_annotations)} annotations with anatomical context")
        return enhanced_annotations
    
    def enhance_annotation_with_anatomy(self, image_data: Dict) -> Dict:
        """Enhance wound annotation with anatomical context"""
        enhanced = image_data.copy()
        
        # Extract anatomical information
        body_region = self.identify_body_region(image_data)
        wound_characteristics = self.analyze_wound_characteristics(image_data)
        
        # Create enhanced classification
        enhanced['anatomical_region'] = body_region
        enhanced['wound_characteristics'] = wound_characteristics
        enhanced['enhanced_class'] = self.create_enhanced_class(
            image_data.get('wound_type'), 
            body_region
        )
        
        # Add spatial context
        enhanced['spatial_context'] = self.get_spatial_context(image_data)
        
        return enhanced
    
    def identify_body_region(self, image_data: Dict) -> str:
        """Identify body region from your body map data"""
        # This would use your specific body map format
        # Adapt based on how your body map data is structured
        
        location = image_data.get('location', '')
        anatomical_site = image_data.get('anatomical_site', '')
        
        # Map your location data to standardized body regions
        region_mapping = {
            'foot': 'foot', 'heel': 'heel', 'ankle': 'ankle',
            'leg': 'leg', 'knee': 'knee', 'thigh': 'thigh',
            'hip': 'hip', 'back': 'back', 'shoulder': 'shoulder',
            'sacrum': 'sacrum', 'chest': 'chest'
        }
        
        for key, region in region_mapping.items():
            if key.lower() in location.lower() or key.lower() in anatomical_site.lower():
                return region
        
        return 'unknown'
    
    def analyze_wound_characteristics(self, image_data: Dict) -> Dict:
        """Extract detailed wound characteristics from your labels"""
        characteristics = {}
        
        # Extract from your label format
        characteristics['wound_type'] = image_data.get('wound_type', 'unknown')
        characteristics['stage'] = image_data.get('stage', 'unknown')
        characteristics['size_category'] = image_data.get('size', 'medium')
        characteristics['severity'] = image_data.get('severity', 'moderate')
        
        # Add derived characteristics
        characteristics['high_risk_location'] = self.is_high_risk_location(
            characteristics.get('anatomical_region')
        )
        
        return characteristics
    
    def create_enhanced_class(self, wound_type: str, body_region: str) -> str:
        """Create enhanced classification combining wound type and location"""
        if not wound_type or not body_region:
            return 'unknown_wound'
        
        # Create specific combinations that are clinically meaningful
        combinations = {
            ('pressure_ulcer', 'heel'): 'pressure_ulcer_heel',
            ('pressure_ulcer', 'sacrum'): 'pressure_ulcer_sacrum',
            ('diabetic_ulcer', 'foot'): 'diabetic_foot_ulcer',
            ('venous_ulcer', 'leg'): 'venous_leg_ulcer',
            ('surgical_wound', 'chest'): 'surgical_wound_chest'
        }
        
        key = (wound_type.lower(), body_region.lower())
        return combinations.get(key, f"{wound_type}_{body_region}")
    
    def get_spatial_context(self, image_data: Dict) -> Dict:
        """Get spatial context from body map"""
        return {
            'bilateral': image_data.get('bilateral', False),
            'proximity_to_bone': image_data.get('over_bony_prominence', False),
            'weight_bearing_area': image_data.get('weight_bearing', False),
            'high_pressure_zone': image_data.get('pressure_prone', False)
        }
    
    def is_high_risk_location(self, region: str) -> bool:
        """Identify high-risk anatomical locations"""
        high_risk_regions = ['heel', 'sacrum', 'shoulder', 'hip', 'ankle']
        return region.lower() in high_risk_regions
    
    def create_enhanced_yolo_labels(self, enhanced_annotations: List[Dict]):
        """Create YOLO labels with enhanced classifications"""
        class_mapping = {name: idx for idx, name in enumerate(self.enhanced_classes)}
        
        for annotation in enhanced_annotations:
            # Create YOLO format label
            image_name = annotation.get('image_name', '')
            enhanced_class = annotation.get('enhanced_class', 'unknown_wound')
            
            if enhanced_class not in class_mapping:
                class_mapping[enhanced_class] = len(class_mapping)
            
            class_id = class_mapping[enhanced_class]
            
            # Get bounding box (adapt to your data format)
            bbox = self.extract_bounding_box(annotation)
            
            # Save YOLO label file
            label_file = self.dataset_path / "enhanced_labels" / f"{Path(image_name).stem}.txt"
            with open(label_file, 'w') as f:
                # YOLO format: class_id center_x center_y width height
                f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
        
        # Save class mapping
        with open(self.dataset_path / "enhanced_class_mapping.json", 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        logger.info(f"Created enhanced YOLO labels with {len(class_mapping)} classes")
        return class_mapping
    
    def extract_bounding_box(self, annotation: Dict) -> List[float]:
        """Extract and normalize bounding box from your annotation format"""
        # Adapt this to your specific annotation format
        
        if 'bbox' in annotation:
            return annotation['bbox']
        
        # If you have different format, convert here
        # Example conversions:
        if 'coordinates' in annotation:
            coords = annotation['coordinates']
            # Convert to normalized YOLO format
            return self.convert_to_yolo_format(coords)
        
        # Default center bounding box if no coordinates
        return [0.5, 0.5, 0.3, 0.3]
    
    def convert_to_yolo_format(self, coords: Dict) -> List[float]:
        """Convert coordinates to YOLO format"""
        # This depends on your coordinate format
        # Example for absolute coordinates:
        if 'x1' in coords and 'image_width' in coords:
            x1, y1 = coords['x1'], coords['y1']
            x2, y2 = coords['x2'], coords['y2']
            img_w, img_h = coords['image_width'], coords['image_height']
            
            center_x = (x1 + x2) / 2 / img_w
            center_y = (y1 + y2) / 2 / img_h
            width = abs(x2 - x1) / img_w
            height = abs(y2 - y1) / img_h
            
            return [center_x, center_y, width, height]
        
        return [0.5, 0.5, 0.3, 0.3]
    
    def parse_labels_file(self, labels_file: str) -> List[Dict]:
        """Parse non-JSON label files"""
        # Implement based on your label file format
        # This is a placeholder - adapt to your specific format
        
        labels_data = []
        with open(labels_file, 'r') as f:
            # Example for CSV format
            if labels_file.endswith('.csv'):
                import csv
                reader = csv.DictReader(f)
                labels_data = list(reader)
        
        return labels_data

def main():
    """Main function to process your body map data"""
    print("🗺️  Enhanced YOLO Training with Body Map Integration")
    print("=" * 60)
    
    # Initialize enhanced training
    trainer = BodyMapYOLOTraining()
    
    print("✅ Enhanced training pipeline created")
    print("📋 Next steps:")
    print("1. Provide your body map file path")
    print("2. Provide your labels file path") 
    print("3. Run processing to enhance annotations")
    print("4. Start enhanced YOLO training")
    
    print(f"\n📁 Enhanced dataset directory: {trainer.dataset_path}")
    print(f"🎯 Enhanced classes: {len(trainer.enhanced_classes)} wound-location combinations")
    
    return trainer

if __name__ == "__main__":
    trainer = main()