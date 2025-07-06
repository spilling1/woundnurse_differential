#!/usr/bin/env python3
"""
Organize training data with proper wound classifications from CSV labels
"""

import pandas as pd
import os
import shutil
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def organize_dataset():
    """Organize the wound dataset with proper classifications"""
    
    # Load the CSV labels
    csv_path = "wound_dataset_pytorch/Train/wound_locations_Labels_AZH_Train.csv"
    df = pd.read_csv(csv_path)
    
    # Create output directory
    output_dir = Path("wound_dataset_final")
    output_dir.mkdir(exist_ok=True)
    
    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Wound type mapping
    wound_type_map = {
        0: "background",      # No wound
        1: "diabetic_ulcer",  # D folder
        2: "neuropathic_ulcer", # N folder  
        3: "pressure_ulcer",  # P folder
        4: "surgical_wound",  # S folder
        5: "venous_ulcer"     # V folder
    }
    
    # Process each image
    annotations = {'train': [], 'val': [], 'test': []}
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for idx, row in df.iterrows():
        image_path = row['index']
        location_id = row['Locations']
        wound_label = row['Labels']
        
        # Get wound type
        wound_type = wound_type_map[wound_label]
        
        # Convert path format (handle both / and \)
        image_path = image_path.replace('\\', '/')
        
        # Find the actual image file
        source_path = Path(f"wound_dataset_pytorch/Train/{image_path}.jpg")
        
        if not source_path.exists():
            logger.warning(f"Image not found: {source_path}")
            continue
        
        # Determine split (70% train, 20% val, 10% test)
        if split_counts['train'] < 487:  # 70% of 696
            split = 'train'
        elif split_counts['val'] < 139:  # 20% of 696
            split = 'val'
        else:
            split = 'test'
        
        split_counts[split] += 1
        
        # Copy image to appropriate split
        dest_path = output_dir / split / "images" / source_path.name
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
        
        annotations[split].append(annotation)
    
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
    
    # Generate dataset summary
    summary = {
        'total_images': sum(split_counts.values()),
        'splits': split_counts,
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
    print("Organizing Wound Training Dataset")
    print("=" * 40)
    
    try:
        summary = organize_dataset()
        
        print("\n📊 Dataset Summary:")
        print(f"Total Images: {summary['total_images']}")
        print(f"Training: {summary['splits']['train']}")
        print(f"Validation: {summary['splits']['val']}")
        print(f"Testing: {summary['splits']['test']}")
        
        print("\n🏷️  Class Distribution:")
        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()}:")
            for wound_type, count in summary['classes'][split].items():
                print(f"  {wound_type}: {count}")
        
        print("\n✅ Dataset organized successfully!")
        print("📁 Files created:")
        print("  - wound_dataset_final/train/")
        print("  - wound_dataset_final/val/")
        print("  - wound_dataset_final/test/")
        print("  - class_mapping.json")
        print("  - dataset_summary.json")
        
        print("\n🚀 Ready for training!")
        print("Run: python3 wound_cnn_trainer.py")
        
    except Exception as e:
        logger.error(f"Error organizing dataset: {e}")
        raise

if __name__ == "__main__":
    main()