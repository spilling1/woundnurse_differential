#!/usr/bin/env python3
"""
Extract and organize uploaded wound images for training
"""

import zipfile
import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_wound_dataset(zip_path: str, output_dir: str = "wound_dataset"):
    """
    Extract uploaded zip file and organize for training
    
    Args:
        zip_path: Path to uploaded zip file
        output_dir: Directory to organize training data
    """
    
    output_path = Path(output_dir)
    
    # Create training structure
    dirs = [
        "train/images", "train/annotations",
        "val/images", "val/annotations", 
        "test/images", "test/annotations",
        "all_images"  # Temporary storage
    ]
    
    for dir_name in dirs:
        (output_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path / "all_images")
    
    logger.info(f"Extracted images to {output_path / 'all_images'}")
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        image_files.extend(list((output_path / "all_images").rglob(f"*{ext}")))
        image_files.extend(list((output_path / "all_images").rglob(f"*{ext.upper()}")))
    
    logger.info(f"Found {len(image_files)} images")
    
    # Split dataset (70% train, 20% val, 10% test)
    import random
    random.shuffle(image_files)
    
    train_size = int(len(image_files) * 0.7)
    val_size = int(len(image_files) * 0.2)
    
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    # Copy to training directories
    for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        for img_file in files:
            dest = output_path / split / "images" / img_file.name
            shutil.copy2(img_file, dest)
    
    logger.info(f"Split dataset: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Create placeholder annotations
    create_placeholder_annotations(output_path, train_files, val_files, test_files)
    
    # Cleanup
    shutil.rmtree(output_path / "all_images")
    
    return {
        'total_images': len(image_files),
        'train_images': len(train_files),
        'val_images': len(val_files),
        'test_images': len(test_files)
    }

def create_placeholder_annotations(output_path: Path, train_files, val_files, test_files):
    """Create placeholder annotation files"""
    
    def create_annotations(files, split):
        annotations = []
        for img_file in files:
            # Placeholder annotation - you'll need to replace with real data
            annotation = {
                "image_name": img_file.name,
                "wound_type": "pressure_ulcer",  # Default - update with real data
                "body_region_id": 220,  # Default sacrum - update with real data
                "bbox": [0.4, 0.4, 0.2, 0.2],  # Default center box - update with real data
                "severity": "moderate"
            }
            annotations.append(annotation)
        
        # Save annotations
        import json
        with open(output_path / split / "annotations.json", 'w') as f:
            json.dump(annotations, f, indent=2)
    
    create_annotations(train_files, 'train')
    create_annotations(val_files, 'val') 
    create_annotations(test_files, 'test')
    
    logger.info("Created placeholder annotation files - update with your real wound data")

def main():
    """Main extraction function"""
    print("Wound Image Dataset Extractor")
    print("=" * 40)
    
    # Look for zip files in current directory
    zip_files = list(Path('.').glob('*.zip'))
    
    if zip_files:
        print(f"Found zip files: {[f.name for f in zip_files]}")
        zip_path = zip_files[0]  # Use first zip file found
        
        print(f"Extracting {zip_path}...")
        results = extract_wound_dataset(str(zip_path))
        
        print("\nExtraction Results:")
        print(f"Total images: {results['total_images']}")
        print(f"Training: {results['train_images']}")
        print(f"Validation: {results['val_images']}")
        print(f"Testing: {results['test_images']}")
        
        print("\nNext steps:")
        print("1. Update annotations in wound_dataset/*/annotations.json with your real wound data")
        print("2. Add body_region_id from your body mapping system")
        print("3. Run: python3 wound_cnn_trainer.py to start training")
        
    else:
        print("No zip files found. Please upload your wound images as a zip file first.")
        print("You can also specify the zip file path manually:")
        print("python3 extract_wound_images.py path/to/your/wounds.zip")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
        results = extract_wound_dataset(zip_path)
        print(f"Extracted {results['total_images']} images successfully")
    else:
        main()