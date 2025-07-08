#!/usr/bin/env python3
"""
Training Data Cleanup Strategy
Determines what training data to keep vs archive vs delete
"""

import os
import shutil
from pathlib import Path

def analyze_storage_usage():
    """Analyze current storage usage of training data"""
    print("=== Storage Usage Analysis ===")
    
    directories = {
        "wound_dataset_body_context": "Main dataset (for training/validation)",
        "training_archive": "Archived training scripts and models", 
        "test_images_raw": "Raw test images",
        "to_delete": "Files marked for deletion"
    }
    
    total_size = 0
    for dirname, description in directories.items():
        if os.path.exists(dirname):
            size = get_directory_size(dirname)
            size_mb = size / (1024 * 1024)
            total_size += size
            print(f"  {dirname}: {size_mb:.1f} MB - {description}")
        else:
            print(f"  {dirname}: Not found")
    
    total_mb = total_size / (1024 * 1024)
    print(f"\nTotal training data: {total_mb:.1f} MB")
    
    return total_mb

def get_directory_size(path):
    """Calculate directory size in bytes"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except:
        pass
    return total

def create_cleanup_strategy():
    """Create strategic cleanup plan"""
    print("\n=== Cleanup Strategy ===")
    
    strategies = {
        "KEEP": {
            "description": "Essential for production",
            "items": [
                "models/wound_yolo.pt (your trained model)",
                "dataset.yaml (configuration)", 
                "wound_dataset_body_context/test/ (for validation)",
                "YOLO service files (yolo_smart_service.py)"
            ]
        },
        "ARCHIVE": {
            "description": "Compress and store offline",
            "items": [
                "wound_dataset_body_context/train/ (training images)",
                "wound_dataset_body_context/val/ (validation images)",
                "training_archive/ (training scripts)"
            ]
        },
        "DELETE": {
            "description": "Safe to remove",
            "items": [
                "to_delete/ (already marked for deletion)",
                "test_images_raw/ (duplicates existing test data)",
                "__pycache__/ (Python cache files)",
                "*.log files (training logs)"
            ]
        }
    }
    
    for action, details in strategies.items():
        print(f"\n{action}: {details['description']}")
        for item in details['items']:
            print(f"  - {item}")
    
    return strategies

def show_production_essentials():
    """Show what's needed for production deployment"""
    print("\n=== Production Deployment Essentials ===")
    
    essentials = {
        "CRITICAL": [
            "models/wound_yolo.pt (your trained model)",
            "yolo_smart_service.py (detection service)",
            "dataset.yaml (model configuration)"
        ],
        "USEFUL": [
            "wound_dataset_body_context/test/ (validation images)",
            "test_custom_model.py (testing script)",
            "deploy_custom_model.py (deployment script)"
        ],
        "OPTIONAL": [
            "wound_dataset_body_context/train/ (training images)",
            "wound_dataset_body_context/val/ (validation images)",
            "training_archive/ (training scripts)"
        ]
    }
    
    total_critical_size = 0
    for category, files in essentials.items():
        print(f"\n{category}:")
        for file_desc in files:
            file_path = file_desc.split(" (")[0]
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path) / (1024 * 1024)
                else:
                    size = get_directory_size(file_path) / (1024 * 1024)
                
                if category == "CRITICAL":
                    total_critical_size += size
                    
                print(f"  ✓ {file_desc} - {size:.1f} MB")
            else:
                print(f"  ? {file_desc} - Missing")
    
    print(f"\nCritical files total: {total_critical_size:.1f} MB")
    
    return essentials

def calculate_space_savings():
    """Calculate potential space savings"""
    print("\n=== Space Savings Analysis ===")
    
    # Calculate sizes for different categories
    keep_sizes = []
    archive_sizes = []
    delete_sizes = []
    
    # Items to keep
    keep_items = [
        "models/",
        "yolo_smart_service.py",
        "dataset.yaml",
        "wound_dataset_body_context/test/",
        "wound_dataset_body_context/dataset.yaml",
        "wound_dataset_body_context/class_mapping.json"
    ]
    
    # Items to archive (optional)
    archive_items = [
        "wound_dataset_body_context/train/",
        "wound_dataset_body_context/val/",
        "wound_dataset_body_context/images/train/",
        "wound_dataset_body_context/images/val/",
        "training_archive/"
    ]
    
    # Items to delete
    delete_items = [
        "to_delete/",
        "test_images_raw/",
        "__pycache__/"
    ]
    
    def calculate_category_size(items, category_name):
        total = 0
        print(f"\n{category_name}:")
        for item in items:
            if os.path.exists(item):
                if os.path.isfile(item):
                    size = os.path.getsize(item) / (1024 * 1024)
                else:
                    size = get_directory_size(item) / (1024 * 1024)
                total += size
                print(f"  {item}: {size:.1f} MB")
        return total
    
    keep_total = calculate_category_size(keep_items, "KEEP (Essential)")
    archive_total = calculate_category_size(archive_items, "ARCHIVE (Optional)")
    delete_total = calculate_category_size(delete_items, "DELETE (Safe to remove)")
    
    current_total = keep_total + archive_total + delete_total
    
    print(f"\n=== Summary ===")
    print(f"Current total: {current_total:.1f} MB")
    print(f"Essential only: {keep_total:.1f} MB")
    print(f"Can archive: {archive_total:.1f} MB")
    print(f"Can delete: {delete_total:.1f} MB")
    print(f"Space savings: {archive_total + delete_total:.1f} MB ({((archive_total + delete_total)/current_total)*100:.1f}%)")

def provide_recommendations():
    """Provide specific recommendations"""
    print("\n=== Recommendations ===")
    
    print("For PRODUCTION deployment:")
    print("✓ You can safely DELETE training images after your model is trained")
    print("✓ Keep only test images for validation")
    print("✓ Keep your trained model and configuration files")
    print("✓ Archive training scripts for future reference")
    
    print("\nFor DEVELOPMENT/RETRAINING:")
    print("✓ Keep training images if you plan to retrain")
    print("✓ Keep validation images for model improvement")
    print("✓ Archive old training attempts")
    
    print("\nSAFE TO DELETE immediately:")
    print("- to_delete/ directory")
    print("- __pycache__/ directories")
    print("- .log files")
    print("- test_images_raw/ (duplicates test data)")
    
    print("\nONCE YOUR MODEL IS DEPLOYED:")
    print("- Training images can be archived/deleted")
    print("- Validation images can be archived/deleted")
    print("- Keep only test images for ongoing validation")

if __name__ == "__main__":
    current_size = analyze_storage_usage()
    create_cleanup_strategy()
    show_production_essentials()
    calculate_space_savings()
    provide_recommendations()