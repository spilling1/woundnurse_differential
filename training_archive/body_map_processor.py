#!/usr/bin/env python3
"""
Body Map Processor for Wound Detection Training
Processes your detailed anatomical body mapping system for enhanced CNN training
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BodyMapProcessor:
    """Process your anatomical body mapping data for wound detection training"""
    
    def __init__(self):
        # Your body region mappings from the labels
        self.body_regions = {
            # Head and face regions
            'head': list(range(235, 238)) + list(range(272, 282)),
            'face': list(range(273, 282)),
            
            # Upper body regions
            'neck': [238, 283],
            'shoulder': [239, 240, 241, 242, 282, 284, 285, 288],
            'chest': [286, 287],
            'back_upper': [244, 245, 247],
            'back_lower': [248, 249, 250],
            'abdomen_upper': [294, 295, 296],
            'abdomen_lower': [299, 300, 301],
            'ribs': [291, 292],
            
            # Arms and hands
            'arm_upper': [243, 246, 253, 255, 289, 290, 298, 302],
            'elbow': [251, 252, 293, 297],
            'arm_lower': [253, 255, 298, 302],
            'wrist': [256, 257, 303, 307] + [1, 33, 66, 99],
            'hand': [258, 261, 308, 311] + list(range(1, 66)) + list(range(66, 131)),
            
            # Buttocks and hips
            'buttock': [254] + list(range(219, 235)),
            'sacrum': [220, 221],
            'hip': [304, 306],
            
            # Legs and feet
            'thigh_upper': [309, 310, 259, 260],
            'thigh_lower': [312, 313, 262, 263],
            'knee': [314, 315, 264, 265],
            'leg': [316, 317, 266, 267] + list(range(152, 165)) + list(range(175, 185)),
            'ankle': [318, 319, 268, 269] + list(range(156, 159)) + list(range(178, 182)),
            'foot': [320, 321, 270, 271] + list(range(131, 152)) + list(range(182, 218)),
            'heel': [150, 151, 217, 218],
            'toe': list(range(131, 141)) + list(range(165, 175)) + list(range(188, 208))
        }
        
        # High-risk pressure areas for wound development
        self.pressure_zones = {
            'high_risk': [220, 221, 150, 151, 217, 218, 239, 240],  # Sacrum, heels, shoulders
            'medium_risk': [254, 314, 315, 264, 265],  # Buttocks, knees
            'low_risk': list(range(286, 301))  # Chest, abdomen
        }
        
        # Clinical wound classifications enhanced with your body regions
        self.enhanced_wound_classes = {
            'pressure_ulcer_sacrum': {'regions': [220, 221], 'risk': 'high'},
            'pressure_ulcer_heel': {'regions': [150, 151, 217, 218], 'risk': 'high'},
            'pressure_ulcer_shoulder': {'regions': [239, 240, 241, 242], 'risk': 'high'},
            'pressure_ulcer_buttock': {'regions': list(range(223, 235)), 'risk': 'medium'},
            'diabetic_foot_ulcer': {'regions': list(range(131, 152)) + list(range(182, 218)), 'risk': 'high'},
            'venous_leg_ulcer': {'regions': list(range(175, 187)), 'risk': 'medium'},
            'arterial_ulcer_leg': {'regions': list(range(175, 187)), 'risk': 'medium'},
            'surgical_wound_chest': {'regions': [286, 287], 'risk': 'low'},
            'surgical_wound_abdomen': {'regions': [294, 295, 296, 299, 300, 301], 'risk': 'low'}
        }
    
    def identify_body_region_from_coordinates(self, x: float, y: float, image_width: int, image_height: int) -> str:
        """
        Identify body region from wound coordinates using your body mapping system
        This would need calibration with your actual body map images
        """
        
        # Normalize coordinates
        norm_x = x / image_width
        norm_y = y / image_height
        
        # Basic body region identification (you'd refine this with actual body map overlays)
        if norm_y < 0.15:  # Head region
            return 'head'
        elif norm_y < 0.3:  # Upper torso
            if norm_x < 0.3:
                return 'shoulder'
            elif norm_x > 0.7:
                return 'shoulder'
            else:
                return 'chest'
        elif norm_y < 0.5:  # Mid torso
            return 'abdomen_upper'
        elif norm_y < 0.65:  # Lower torso
            if norm_x < 0.4 or norm_x > 0.6:
                return 'hip'
            else:
                return 'abdomen_lower'
        elif norm_y < 0.8:  # Upper legs
            return 'thigh_upper'
        elif norm_y < 0.9:  # Lower legs
            return 'leg'
        else:  # Feet
            return 'foot'
    
    def get_region_id_from_name(self, region_name: str) -> List[int]:
        """Get region IDs from your body mapping system"""
        return self.body_regions.get(region_name, [])
    
    def assess_pressure_risk(self, region_ids: List[int]) -> str:
        """Assess pressure ulcer risk based on anatomical location"""
        for region_id in region_ids:
            if region_id in self.pressure_zones['high_risk']:
                return 'high'
            elif region_id in self.pressure_zones['medium_risk']:
                return 'medium'
        return 'low'
    
    def create_enhanced_annotation(self, wound_data: Dict) -> Dict:
        """
        Create enhanced annotation combining wound data with your body mapping
        
        Expected wound_data format:
        {
            'image_name': 'wound_001.jpg',
            'wound_type': 'pressure_ulcer',
            'bbox': [x, y, width, height],  # normalized coordinates
            'severity': 'moderate',
            'body_region_id': 220,  # from your mapping system
            'additional_info': {...}
        }
        """
        
        # Identify anatomical context
        region_id = wound_data.get('body_region_id')
        region_name = self.get_region_name_from_id(region_id)
        
        # Create enhanced classification
        wound_type = wound_data.get('wound_type', 'unknown')
        enhanced_class = f"{wound_type}_{region_name}"
        
        # Assess clinical risk
        pressure_risk = self.assess_pressure_risk([region_id]) if region_id else 'unknown'
        
        # Enhanced annotation
        enhanced = {
            'image_name': wound_data['image_name'],
            'original_wound_type': wound_type,
            'enhanced_class': enhanced_class,
            'body_region_id': region_id,
            'body_region_name': region_name,
            'bbox': wound_data['bbox'],
            'severity': wound_data.get('severity', 'moderate'),
            'pressure_risk': pressure_risk,
            'clinical_context': {
                'high_pressure_zone': pressure_risk == 'high',
                'weight_bearing': region_name in ['heel', 'foot', 'buttock'],
                'friction_prone': region_name in ['heel', 'sacrum', 'shoulder'],
                'moisture_risk': region_name in ['buttock', 'foot']
            },
            'anatomical_features': self.get_anatomical_features(region_id)
        }
        
        return enhanced
    
    def get_region_name_from_id(self, region_id: int) -> str:
        """Get region name from your body mapping ID"""
        if not region_id:
            return 'unknown'
            
        for region_name, ids in self.body_regions.items():
            if region_id in ids:
                return region_name
        
        return 'unknown'
    
    def get_anatomical_features(self, region_id: int) -> Dict:
        """Get anatomical features for training context"""
        features = {
            'bony_prominence': False,
            'skin_thickness': 'medium',
            'vascular_supply': 'normal',
            'nerve_supply': 'normal'
        }
        
        # High-risk anatomical features
        if region_id in [220, 221]:  # Sacrum
            features.update({
                'bony_prominence': True,
                'skin_thickness': 'thin',
                'pressure_distribution': 'concentrated'
            })
        elif region_id in [150, 151, 217, 218]:  # Heels
            features.update({
                'bony_prominence': True,
                'skin_thickness': 'thick',
                'weight_bearing': True
            })
        elif region_id in [239, 240, 241, 242]:  # Shoulders
            features.update({
                'bony_prominence': True,
                'skin_thickness': 'medium',
                'friction_risk': True
            })
        
        return features
    
    def process_dataset(self, wound_dataset: List[Dict]) -> List[Dict]:
        """Process entire wound dataset with body mapping enhancement"""
        enhanced_dataset = []
        
        for wound_data in wound_dataset:
            try:
                enhanced = self.create_enhanced_annotation(wound_data)
                enhanced_dataset.append(enhanced)
            except Exception as e:
                logger.warning(f"Failed to process {wound_data.get('image_name', 'unknown')}: {e}")
        
        logger.info(f"Enhanced {len(enhanced_dataset)} wound annotations with body mapping")
        return enhanced_dataset
    
    def create_class_mapping(self, enhanced_dataset: List[Dict]) -> Dict[str, int]:
        """Create class mapping for training"""
        unique_classes = set()
        for annotation in enhanced_dataset:
            unique_classes.add(annotation['enhanced_class'])
        
        class_mapping = {class_name: idx for idx, class_name in enumerate(sorted(unique_classes))}
        
        logger.info(f"Created class mapping with {len(class_mapping)} enhanced wound classes")
        return class_mapping
    
    def validate_body_map_integration(self, sample_data: Dict) -> Dict:
        """Validate integration with a sample annotation"""
        result = {
            'valid': True,
            'issues': [],
            'suggestions': []
        }
        
        # Check required fields
        required_fields = ['image_name', 'wound_type', 'bbox']
        for field in required_fields:
            if field not in sample_data:
                result['valid'] = False
                result['issues'].append(f"Missing required field: {field}")
        
        # Check body region mapping
        if 'body_region_id' in sample_data:
            region_id = sample_data['body_region_id']
            if region_id not in range(1, 322):  # Your mapping range
                result['issues'].append(f"Invalid body region ID: {region_id}")
        else:
            result['suggestions'].append("Add 'body_region_id' field for enhanced training")
        
        return result

def main():
    """Test body map processing with sample data"""
    processor = BodyMapProcessor()
    
    # Sample wound data in your format
    sample_wounds = [
        {
            'image_name': 'pressure_ulcer_sacrum_001.jpg',
            'wound_type': 'pressure_ulcer',
            'bbox': [0.45, 0.65, 0.1, 0.08],
            'severity': 'severe',
            'body_region_id': 220  # Left Sacrum from your mapping
        },
        {
            'image_name': 'diabetic_foot_ulcer_001.jpg', 
            'wound_type': 'diabetic_ulcer',
            'bbox': [0.3, 0.9, 0.15, 0.12],
            'severity': 'moderate',
            'body_region_id': 150  # Right Lateral Heel
        }
    ]
    
    # Process with body mapping
    enhanced_dataset = processor.process_dataset(sample_wounds)
    class_mapping = processor.create_class_mapping(enhanced_dataset)
    
    print("Body Map Integration Test Results:")
    print("=" * 50)
    
    for annotation in enhanced_dataset:
        print(f"\nOriginal: {annotation['original_wound_type']}")
        print(f"Enhanced: {annotation['enhanced_class']}")
        print(f"Region: {annotation['body_region_name']} (ID: {annotation['body_region_id']})")
        print(f"Risk Level: {annotation['pressure_risk']}")
        print(f"Clinical Context: {annotation['clinical_context']}")
    
    print(f"\nClass Mapping: {class_mapping}")
    print(f"\nReady to train CNN with {len(enhanced_dataset)} enhanced annotations")
    
    return processor

if __name__ == "__main__":
    processor = main()