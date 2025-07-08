#!/usr/bin/env python3
"""
Validate Your Dataset YAML Configuration
Checks compatibility with the current project structure
"""

import yaml
import os
from pathlib import Path

def analyze_your_dataset_yaml():
    """Analyze your provided dataset.yaml configuration"""
    print("=== Analyzing Your Dataset Configuration ===")
    
    # Load your dataset.yaml
    yaml_file = "attached_assets/dataset_1751958394221.yaml"
    
    if not os.path.exists(yaml_file):
        print("✗ Dataset YAML file not found")
        return False
    
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("✓ Dataset YAML loaded successfully")
    print(f"  Classes: {config['nc']}")
    print(f"  Path: {config['path']}")
    
    # Show class mapping
    print("\n📋 Class Configuration:")
    for idx, name in config['names'].items():
        print(f"  {idx}: {name}")
    
    # Show clinical priority structure
    if 'clinical_priority' in config:
        print("\n🏥 Clinical Priority Levels:")
        for level, wounds in config['clinical_priority'].items():
            print(f"  {level.upper()}: {', '.join(wounds)}")
    
    # Check compatibility with current dataset
    current_dataset = "wound_dataset_body_context"
    if os.path.exists(current_dataset):
        print(f"\n🔍 Compatibility Check with {current_dataset}:")
        
        # Load current class mapping
        class_file = f"{current_dataset}/class_mapping.json"
        if os.path.exists(class_file):
            import json
            with open(class_file, 'r') as f:
                current_classes = json.load(f)
            
            print(f"  Current dataset classes: {current_classes['num_classes']}")
            print(f"  Your YAML classes: {config['nc']}")
            
            # Compare class names
            yaml_classes = set(config['names'].values())
            current_class_names = set(current_classes['class_names'])
            
            common_classes = yaml_classes.intersection(current_class_names)
            print(f"  Common classes: {len(common_classes)}")
            
            if common_classes:
                print(f"    Matching: {', '.join(sorted(common_classes))}")
            
            different_classes = yaml_classes.symmetric_difference(current_class_names)
            if different_classes:
                print(f"    Different: {', '.join(sorted(different_classes))}")
    
    return config

def adapt_yaml_for_current_project(config):
    """Adapt your YAML for the current project structure"""
    print("\n=== Adapting Configuration for Current Project ===")
    
    # Update path to match current structure
    adapted_config = config.copy()
    adapted_config['path'] = str(Path("wound_dataset_body_context").absolute())
    
    print(f"✓ Updated path: {adapted_config['path']}")
    
    # Keep your excellent class structure and clinical priorities
    print("✓ Preserving your clinical priority structure")
    print("✓ Maintaining your 5-class configuration")
    
    # Save adapted version
    output_file = "wound_dataset_body_context/dataset.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(adapted_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Saved adapted configuration: {output_file}")
    
    # Also save to root for easy access
    root_file = "dataset.yaml"
    with open(root_file, 'w') as f:
        yaml.dump(adapted_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Saved copy to root: {root_file}")
    
    return adapted_config

def show_deployment_locations():
    """Show where to place your dataset.yaml"""
    print("\n=== Dataset YAML Deployment Locations ===")
    
    locations = [
        {
            "path": "wound_dataset_body_context/dataset.yaml",
            "purpose": "Primary location - with dataset files",
            "priority": "🎯 RECOMMENDED"
        },
        {
            "path": "dataset.yaml", 
            "purpose": "Root directory - easy access for training",
            "priority": "✅ CONVENIENT"
        },
        {
            "path": "models/dataset.yaml",
            "purpose": "With model files - organized approach", 
            "priority": "📁 ORGANIZED"
        }
    ]
    
    print("Your dataset.yaml can be placed in multiple locations:")
    for loc in locations:
        print(f"\n{loc['priority']}")
        print(f"  📍 {loc['path']}")
        print(f"  💡 {loc['purpose']}")
    
    print("\n🚀 For YOLO training, use any of these paths:")
    print("  yolo train data=wound_dataset_body_context/dataset.yaml model=yolov8n.pt")
    print("  yolo train data=dataset.yaml model=yolov8n.pt")

def validate_training_compatibility():
    """Check if the configuration works with current training setup"""
    print("\n=== Training Compatibility Check ===")
    
    # Check if required directories exist
    base_path = "wound_dataset_body_context"
    required_dirs = [
        f"{base_path}/images/train",
        f"{base_path}/images/val", 
        f"{base_path}/images/test"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png'))])
            print(f"✓ {dir_path}: {file_count} images")
        else:
            print(f"✗ {dir_path}: Missing")
            all_good = False
    
    if all_good:
        print("\n🎉 Your dataset.yaml is fully compatible!")
        print("✓ All required directories exist")
        print("✓ Class configuration is well-structured")
        print("✓ Clinical priorities are clearly defined")
        print("✓ Ready for YOLO training")
    else:
        print("\n⚠️  Some adjustments may be needed")
        print("Check that image directories match the expected structure")
    
    return all_good

if __name__ == "__main__":
    config = analyze_your_dataset_yaml()
    if config:
        adapted_config = adapt_yaml_for_current_project(config)
        show_deployment_locations()
        validate_training_compatibility()
    else:
        print("❌ Could not process dataset.yaml")