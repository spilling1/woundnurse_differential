#!/usr/bin/env python3
"""
Setup Dataset Configuration
Creates and validates dataset.yaml for YOLO training
"""

import os
import yaml
import json
from pathlib import Path

def create_dataset_yaml(dataset_dir="wound_dataset_body_context"):
    """Create properly formatted dataset.yaml for YOLO training"""
    print("=== Setting up Dataset Configuration ===")
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"✗ Dataset directory not found: {dataset_dir}")
        return False
    
    # Load existing dataset info
    summary_file = dataset_path / "dataset_summary.json"
    class_file = dataset_path / "class_mapping.json"
    
    if summary_file.exists() and class_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        with open(class_file, 'r') as f:
            classes = json.load(f)
        
        print(f"✓ Found dataset with {summary['total_images']} images")
        print(f"✓ Found {classes['num_classes']} classes")
    else:
        print("✗ Dataset metadata files missing")
        return False
    
    # Create dataset.yaml configuration
    dataset_config = {
        # Core YOLO configuration
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images',
        'nc': classes['num_classes'],
        'names': {int(k): v for k, v in classes['classes'].items()},
        
        # Additional metadata
        'dataset_info': {
            'description': f"Wound detection dataset with {summary['total_images']} images across {classes['num_classes']} categories",
            'total_images': summary['total_images'],
            'train_images': summary['splits']['train'],
            'val_images': summary['splits']['val'],
            'test_images': summary['splits']['test'],
            'created': "2025-07-07",
            'version': "1.0"
        },
        
        # Training recommendations
        'training_config': {
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'patience': 10,
            'optimizer': "AdamW",
            'lr0': 0.01,
            'weight_decay': 0.0005
        }
    }
    
    # Save dataset.yaml
    yaml_file = dataset_path / "dataset.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created dataset.yaml at: {yaml_file}")
    
    # Validate dataset structure
    validate_dataset_structure(dataset_path)
    
    return True

def validate_dataset_structure(dataset_path):
    """Validate that dataset has proper YOLO structure"""
    print("\n=== Validating Dataset Structure ===")
    
    required_dirs = [
        'train/images',
        'train/labels', 
        'val/images',
        'val/labels',
        'test/images',
        'test/labels'
    ]
    
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"✓ {dir_name}: {file_count} files")
        else:
            print(f"✗ {dir_name}: Missing")
    
    # Check if labels exist for images
    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / f"{split}/images"
        labels_dir = dataset_path / f"{split}/labels"
        
        if images_dir.exists() and labels_dir.exists():
            images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            labels = list(labels_dir.glob("*.txt"))
            
            print(f"  {split}: {len(images)} images, {len(labels)} labels")
            
            if len(labels) == 0 and len(images) > 0:
                print(f"  ⚠️  {split}: No label files found - may need YOLO format conversion")

def show_usage_instructions():
    """Show how to use the dataset.yaml file"""
    print("\n=== Usage Instructions ===")
    print("Your dataset.yaml file is now ready for YOLO training!")
    print()
    print("📍 Location: wound_dataset_body_context/dataset.yaml")
    print()
    print("🚀 To train your YOLO model:")
    print("  1. Use this dataset.yaml path in your training script")
    print("  2. Point your YOLO training to: wound_dataset_body_context/dataset.yaml")
    print("  3. The trained model (best.pt) should go in: models/wound_yolo.pt")
    print()
    print("📋 Example training command:")
    print("  yolo train data=wound_dataset_body_context/dataset.yaml model=yolov8n.pt epochs=100")
    print()
    print("🔧 Configuration details:")
    print("  - 6 wound classes configured")
    print("  - Train/val/test splits defined")
    print("  - Absolute paths for compatibility")
    print("  - Training parameters included")

def create_alternative_locations():
    """Create dataset.yaml in alternative locations for flexibility"""
    print("\n=== Creating Alternative Locations ===")
    
    # Main dataset directory (primary location)
    main_location = "wound_dataset_body_context/dataset.yaml"
    
    # Root directory (for easy access)
    root_location = "dataset.yaml"
    
    # Models directory (near model files)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    models_location = "models/dataset.yaml"
    
    if os.path.exists(main_location):
        # Copy to root for easy access
        import shutil
        shutil.copy2(main_location, root_location)
        print(f"✓ Copied to root: {root_location}")
        
        # Copy to models directory
        shutil.copy2(main_location, models_location)
        print(f"✓ Copied to models: {models_location}")
    
    print("\n📍 Your dataset.yaml is now available in multiple locations:")
    print(f"  Primary: {main_location}")
    print(f"  Root: {root_location}")
    print(f"  Models: {models_location}")
    print("\nUse whichever location is most convenient for your training setup!")

if __name__ == "__main__":
    if create_dataset_yaml():
        show_usage_instructions()
        create_alternative_locations()
    else:
        print("❌ Failed to create dataset.yaml")
        print("Please check that the dataset directory and metadata files exist")