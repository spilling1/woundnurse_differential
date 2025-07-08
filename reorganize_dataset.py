#!/usr/bin/env python3
"""
Reorganize Dataset Structure
Adapts current dataset to match your YAML configuration
"""

import os
import shutil
from pathlib import Path

def reorganize_for_yaml():
    """Reorganize dataset to match your YAML structure"""
    print("=== Reorganizing Dataset Structure ===")
    
    base_dir = Path("wound_dataset_body_context")
    
    # Create new structure
    new_structure = {
        "images/train": "train/images",
        "images/val": "val/images", 
        "images/test": "test/images",
        "labels/train": "train/labels",
        "labels/val": "val/labels",
        "labels/test": "test/labels"
    }
    
    for new_path, old_path in new_structure.items():
        new_dir = base_dir / new_path
        old_dir = base_dir / old_path
        
        # Create new directory
        new_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files if old directory exists
        if old_dir.exists():
            files = list(old_dir.glob("*"))
            if files:
                for file in files:
                    if file.is_file():
                        target = new_dir / file.name
                        if not target.exists():
                            shutil.copy2(file, target)
                
                print(f"✓ {new_path}: {len(files)} files")
            else:
                print(f"○ {new_path}: Empty (created)")
        else:
            print(f"○ {new_path}: Created (source missing)")
    
    # Verify structure
    print("\n=== Dataset Structure Verification ===")
    for split in ['train', 'val', 'test']:
        images_dir = base_dir / f"images/{split}"
        labels_dir = base_dir / f"labels/{split}" 
        
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
            
            print(f"  {split}: {len(image_files)} images, {len(label_files)} labels")
    
    print(f"\n✅ Dataset structure now matches your YAML configuration!")
    return True

if __name__ == "__main__":
    reorganize_for_yaml()