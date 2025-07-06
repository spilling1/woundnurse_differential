#!/usr/bin/env python3
"""
Batch upload helper for wound images
Handles various upload scenarios for 730 wound images
"""

import os
import requests
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def download_from_url(url: str, output_dir: str = "downloaded_images"):
    """Download images from a shared URL (Google Drive, Dropbox, etc.)"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        filename = url.split('/')[-1] or "wound_images.zip"
        file_path = output_path / filename
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {filename} to {output_dir}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None

def prepare_annotation_template():
    """Create annotation template for your wound data"""
    
    template = {
        "wound_001.jpg": {
            "wound_type": "pressure_ulcer",
            "body_region_id": 220,  # Update with your body map ID
            "bbox": [0.4, 0.4, 0.2, 0.2],  # [x_center, y_center, width, height] normalized
            "severity": "moderate",
            "notes": "Update with actual wound data"
        },
        "wound_002.jpg": {
            "wound_type": "diabetic_foot_ulcer", 
            "body_region_id": 150,  # Right heel
            "bbox": [0.3, 0.8, 0.15, 0.1],
            "severity": "severe",
            "notes": "Example diabetic foot ulcer"
        }
    }
    
    with open("annotation_template.json", 'w') as f:
        json.dump(template, f, indent=2)
    
    print("Created annotation_template.json")
    print("Update this file with your actual wound data before training")

def check_upload_status():
    """Check what images are currently available"""
    
    # Check for zip files
    zip_files = list(Path('.').glob('*.zip'))
    
    # Check for image directories
    image_dirs = ['wound_dataset', 'images', 'training_data']
    existing_dirs = [d for d in image_dirs if Path(d).exists()]
    
    # Count existing images
    image_count = 0
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_count += len(list(Path('.').rglob(f'*{ext}')))
    
    print("Current Upload Status:")
    print("=" * 30)
    print(f"Zip files found: {len(zip_files)}")
    print(f"Image directories: {existing_dirs}")
    print(f"Total images found: {image_count}")
    
    if image_count >= 730:
        print("✅ Sufficient images found for training!")
    elif image_count > 0:
        print(f"⚠️  Found {image_count} images, need {730 - image_count} more")
    else:
        print("❌ No images found - please upload your wound dataset")
    
    return image_count

def main():
    """Main upload helper"""
    print("Wound Image Upload Helper")
    print("=" * 40)
    
    # Check current status
    image_count = check_upload_status()
    
    print("\nUpload Options:")
    print("1. ZIP FILE: Upload wound_images.zip via Replit file manager")
    print("2. CLOUD LINK: Provide Google Drive/Dropbox public link")
    print("3. BATCH FOLDERS: Upload folders of images directly")
    
    # Create helpful templates
    prepare_annotation_template()
    
    print("\nNext Steps:")
    if image_count >= 100:
        print("- Extract images: python3 extract_wound_images.py")
        print("- Update annotations with your body map data")
        print("- Start training: python3 wound_cnn_trainer.py")
    else:
        print("- Upload your 730 wound images using one of the options above")
        print("- Run this script again to check status")

if __name__ == "__main__":
    main()