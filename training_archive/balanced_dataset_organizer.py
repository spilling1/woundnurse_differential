#!/usr/bin/env python3
"""
Create balanced dataset splits ensuring all wound classes are represented
"""

import pandas as pd
import os
import shutil
from pathlib import Path
import json
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_balanced_dataset():
    """Create balanced dataset with all classes in each split"""
    
    # Load the CSV labels
    csv_path = "wound_dataset_pytorch/Train/wound_locations_Labels_AZH_Train.csv"
    df = pd.read_csv(csv_path)
    
    # Create output directory
    output_dir = Path("wound_dataset_balanced")
    output_dir.mkdir(exist_ok=True)
    
    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Wound type mapping
    wound_type_map = {
        0: "background",      # No wound - 75 images
        1: "diabetic_ulcer",  # D folder - 139 images
        2: "neuropathic_ulcer", # N folder - 75 images
        3: "pressure_ulcer",  # P folder - 100 images
        4: "surgical_wound",  # S folder - 122 images
        5: "venous_ulcer"     # V folder - 185 images
    }
    
    # Process each wound class separately to ensure balanced splits
    annotations = {'train': [], 'val': [], 'test': []}
    
    for class_id, wound_type in wound_type_map.items():
        # Get all images for this class
        class_df = df[df['Labels'] == class_id]
        
        if len(class_df) == 0:
            continue
            
        # Split this class: 70% train, 20% val, 10% test
        train_df, temp_df = train_test_split(class_df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42)  # 0.33 * 0.3 = 0.1
        
        # Process each split
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            for idx, row in split_df.iterrows():
                image_path = row['index']
                location_id = row['Locations']
                wound_label = row['Labels']
                
                # Convert path format (handle both / and \)
                image_path = image_path.replace('\\', '/')
                
                # Find the actual image file
                source_path = Path(f"wound_dataset_pytorch/Train/{image_path}.jpg")
                
                if not source_path.exists():
                    logger.warning(f"Image not found: {source_path}")
                    continue
                
                # Copy image to appropriate split
                dest_path = output_dir / split_name / "images" / source_path.name
                shutil.copy2(source_path, dest_path)
                
                # Create annotation
                annotation = {
                    "image_name": source_path.name,
                    "wound_type": wound_type,
                    "wound_class_id": wound_label,
                    "body_region_id": location_id if location_id != -1 else None,
                    "bbox": [0.5, 0.5, 0.4, 0.4],  # Default center bounding box
                    "confidence": 1.0,  # High confidence - this is labeled data
                    "original_path": image_path,
                    "notes": f"Extracted from {image_path}"
                }
                
                annotations[split_name].append(annotation)
    
    # Save annotations for each split
    for split in ['train', 'val', 'test']:
        annotations_file = output_dir / split / "annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(annotations[split], f, indent=2)
        
        logger.info(f"Created {len(annotations[split])} annotations for {split} split")
    
    # Create class mapping
    class_mapping = {
        'classes': wound_type_map,
        'num_classes': len(wound_type_map),
        'class_names': list(wound_type_map.values())
    }
    
    with open(output_dir / "class_mapping.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Generate balanced dataset summary
    summary = {
        'total_images': sum(len(annotations[split]) for split in ['train', 'val', 'test']),
        'splits': {split: len(annotations[split]) for split in ['train', 'val', 'test']},
        'classes': {}
    }
    
    for split in ['train', 'val', 'test']:
        class_counts = {}
        for ann in annotations[split]:
            wound_type = ann['wound_type']
            class_counts[wound_type] = class_counts.get(wound_type, 0) + 1
        summary['classes'][split] = class_counts
    
    # Save summary
    with open(output_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Main organization function"""
    print("Creating Balanced Wound Training Dataset")
    print("=" * 45)
    
    try:
        summary = create_balanced_dataset()
        
        print("\n📊 Balanced Dataset Summary:")
        print(f"Total Images: {summary['total_images']}")
        print(f"Training: {summary['splits']['train']}")
        print(f"Validation: {summary['splits']['val']}")
        print(f"Testing: {summary['splits']['test']}")
        
        print("\n🏷️  Class Distribution (Balanced):")
        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()}:")
            for wound_type, count in summary['classes'][split].items():
                print(f"  {wound_type}: {count}")
        
        print("\n✅ Balanced dataset created successfully!")
        print("📁 Files created:")
        print("  - wound_dataset_balanced/train/")
        print("  - wound_dataset_balanced/val/")
        print("  - wound_dataset_balanced/test/")
        print("  - class_mapping.json")
        print("  - dataset_summary.json")
        
        print("\n🚀 Ready for CNN training!")
        print("Run: python3 wound_cnn_trainer.py")
        
    except Exception as e:
        logger.error(f"Error creating balanced dataset: {e}")
        raise

if __name__ == "__main__":
    main()