#!/usr/bin/env python3
"""
Add additional test images to the wound detection training system
"""

import zipfile
import shutil
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_test_images(test_zip_path: str, output_dir: str = "extended_test_dataset"):
    """
    Add new test images to extend the validation dataset
    
    Args:
        test_zip_path: Path to ZIP file containing test images
        output_dir: Directory to organize extended test data
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create test image directory
    test_images_dir = output_path / "test_images"
    test_images_dir.mkdir(exist_ok=True)
    
    # Extract test images
    image_count = 0
    
    try:
        with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            for file_name in file_list:
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')) and not file_name.startswith('__MACOSX'):
                    # Extract image
                    zip_ref.extract(file_name, test_images_dir)
                    image_count += 1
                    
                    # Move to flat structure if nested
                    source = test_images_dir / file_name
                    if source.parent != test_images_dir:
                        dest = test_images_dir / source.name
                        shutil.move(source, dest)
        
        # Clean up empty directories
        for item in test_images_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
        
        logger.info(f"Extracted {image_count} test images")
        
        # Create annotation template for test images
        create_test_annotation_template(test_images_dir, output_path)
        
        return image_count
        
    except Exception as e:
        logger.error(f"Error extracting test images: {e}")
        return 0

def create_test_annotation_template(test_images_dir: Path, output_dir: Path):
    """Create annotation template for test images"""
    
    # Get all test images
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.jpeg")) + list(test_images_dir.glob("*.png"))
    
    # Create annotation template
    annotations = []
    
    for img_file in image_files:
        annotation = {
            "image_name": img_file.name,
            "wound_type": "unknown",  # To be classified by trained model
            "body_region_id": -1,     # Unknown region
            "bbox": [0.5, 0.5, 0.4, 0.4],  # Default bounding box
            "confidence": 0.0,        # Will be filled by model prediction
            "notes": "Test image for model validation"
        }
        annotations.append(annotation)
    
    # Save annotation template
    with open(output_dir / "test_annotations_template.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    logger.info(f"Created annotation template for {len(annotations)} test images")

def integrate_with_existing_dataset(test_dir: str = "extended_test_dataset"):
    """Integrate new test images with existing training dataset"""
    
    test_path = Path(test_dir)
    training_path = Path("wound_dataset_balanced")
    
    if not test_path.exists():
        logger.error("Test dataset not found")
        return False
    
    if not training_path.exists():
        logger.error("Training dataset not found")
        return False
    
    # Copy test images to training dataset test folder
    test_images_dir = test_path / "test_images"
    training_test_dir = training_path / "test" / "images"
    
    image_count = 0
    for img_file in test_images_dir.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            shutil.copy2(img_file, training_test_dir)
            image_count += 1
    
    logger.info(f"Added {image_count} test images to training dataset")
    
    # Update test annotations
    test_annotations_file = training_path / "test" / "annotations.json"
    
    # Load existing annotations
    with open(test_annotations_file, 'r') as f:
        existing_annotations = json.load(f)
    
    # Load new test annotation template
    template_file = test_path / "test_annotations_template.json"
    with open(template_file, 'r') as f:
        new_annotations = json.load(f)
    
    # Combine annotations
    combined_annotations = existing_annotations + new_annotations
    
    # Save updated annotations
    with open(test_annotations_file, 'w') as f:
        json.dump(combined_annotations, f, indent=2)
    
    logger.info(f"Updated test annotations with {len(new_annotations)} new entries")
    
    return True

def main():
    """Main function to add test images"""
    print("Test Image Integration Tool")
    print("=" * 35)
    
    # Look for ZIP files in current directory
    zip_files = list(Path('.').glob('*.zip'))
    test_zip_files = [f for f in zip_files if 'test' in f.name.lower()]
    
    if test_zip_files:
        print(f"Found test ZIP files: {[f.name for f in test_zip_files]}")
        
        # Use first test ZIP file
        test_zip = test_zip_files[0]
        print(f"Processing: {test_zip}")
        
        # Extract test images
        image_count = add_test_images(str(test_zip))
        
        if image_count > 0:
            print(f"Successfully extracted {image_count} test images")
            
            # Integrate with existing dataset
            if integrate_with_existing_dataset():
                print("Test images integrated with training dataset")
                
                # Update dataset summary
                print("\nUpdated Dataset Summary:")
                training_path = Path("wound_dataset_balanced")
                
                for split in ['train', 'val', 'test']:
                    images_dir = training_path / split / "images"
                    if images_dir.exists():
                        count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.jpeg"))) + len(list(images_dir.glob("*.png")))
                        print(f"  {split}: {count} images")
                
                print("\nNext steps:")
                print("1. Review test_annotations_template.json")
                print("2. Update wound types for test images if known")
                print("3. Run model evaluation on extended test set")
            
        else:
            print("Failed to extract test images")
    
    else:
        print("No test ZIP files found in current directory")
        print("Available ZIP files:", [f.name for f in zip_files])
        print("\nTo add test images:")
        print("1. Upload test images ZIP file")
        print("2. Rename to include 'test' in filename")
        print("3. Run this script again")

if __name__ == "__main__":
    main()