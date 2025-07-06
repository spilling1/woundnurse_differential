#!/usr/bin/env python3
"""
Evaluate trained wound CNN model on test images
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """Same architecture as training model"""
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class WoundModelEvaluator:
    """Evaluate trained wound model on new test images"""
    
    def __init__(self, model_path: str):
        """
        Initialize evaluator with trained model
        
        Args:
            model_path: Path to saved model file (.pth)
        """
        self.device = torch.device('cpu')
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load class mapping
        self.class_mapping = checkpoint['class_mapping']
        self.idx_to_class = {int(idx): name for idx, name in self.class_mapping['classes'].items()}
        
        # Initialize and load model
        self.model = SimpleCNN(num_classes=len(self.class_mapping['classes']))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image transform (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model accuracy: {checkpoint.get('val_acc', 'unknown')}%")
    
    def predict_image(self, image_path: str):
        """
        Predict wound type for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Prediction results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.idx_to_class[predicted_idx.item()]
                confidence_score = confidence.item()
                
                # Get top 3 predictions
                top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
                top3_predictions = [
                    {
                        'class': self.idx_to_class[idx.item()],
                        'confidence': prob.item()
                    }
                    for prob, idx in zip(top3_probs[0], top3_indices[0])
                ]
                
                return {
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'confidence': confidence_score,
                    'top3_predictions': top3_predictions,
                    'success': True
                }
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'success': False
            }
    
    def evaluate_directory(self, test_dir: str, output_file: str = "test_results.json"):
        """
        Evaluate all images in a directory
        
        Args:
            test_dir: Directory containing test images
            output_file: JSON file to save results
            
        Returns:
            dict: Evaluation summary
        """
        test_path = Path(test_dir)
        
        if not test_path.exists():
            logger.error(f"Test directory not found: {test_dir}")
            return None
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(test_path.glob(f"*{ext}")))
            image_files.extend(list(test_path.glob(f"*{ext.upper()}")))
        
        logger.info(f"Found {len(image_files)} test images")
        
        # Process each image
        results = []
        class_counts = {}
        
        for img_file in image_files:
            result = self.predict_image(str(img_file))
            results.append(result)
            
            if result['success']:
                predicted_class = result['predicted_class']
                class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                
                logger.info(f"{img_file.name}: {predicted_class} ({result['confidence']:.3f})")
        
        # Create summary
        summary = {
            'total_images': len(image_files),
            'successful_predictions': len([r for r in results if r['success']]),
            'class_distribution': class_counts,
            'model_info': {
                'classes': list(self.class_mapping['classes'].values()),
                'model_accuracy': checkpoint.get('val_acc', 'unknown')
            },
            'detailed_results': results
        }
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        return summary
    
    def print_summary(self, summary: dict):
        """Print evaluation summary"""
        print("\n🔍 Test Evaluation Results")
        print("=" * 30)
        print(f"Total images processed: {summary['total_images']}")
        print(f"Successful predictions: {summary['successful_predictions']}")
        
        print("\n📊 Predicted Class Distribution:")
        for class_name, count in summary['class_distribution'].items():
            percentage = (count / summary['total_images']) * 100
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        print(f"\n🎯 Model Accuracy: {summary['model_info']['model_accuracy']}%")

def find_best_model():
    """Find the best trained model file"""
    model_files = list(Path('.').glob('wound_model_*.pth'))
    
    if not model_files:
        return None
    
    # Sort by accuracy (extract from filename)
    def get_accuracy(filename):
        try:
            acc_str = filename.stem.split('_')[-1].replace('acc', '')
            return float(acc_str)
        except:
            return 0.0
    
    best_model = max(model_files, key=get_accuracy)
    return str(best_model)

def main():
    """Main evaluation function"""
    print("Wound Model Evaluator")
    print("=" * 25)
    
    # Find trained model
    model_path = find_best_model()
    
    if not model_path:
        print("❌ No trained model found")
        print("Please complete training first by running: python3 minimal_wound_trainer.py")
        return
    
    print(f"📁 Using model: {model_path}")
    
    # Initialize evaluator
    try:
        evaluator = WoundModelEvaluator(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Look for test images
    test_directories = ['test_images', 'new_test_images', 'validation_images']
    test_dir = None
    
    for dir_name in test_directories:
        if Path(dir_name).exists():
            test_dir = dir_name
            break
    
    if test_dir:
        print(f"📂 Found test directory: {test_dir}")
        
        # Evaluate test images
        summary = evaluator.evaluate_directory(test_dir)
        
        if summary:
            evaluator.print_summary(summary)
            print(f"\n💾 Detailed results saved to test_results.json")
        
    else:
        print("📂 No test image directory found")
        print("Available options:")
        print("1. Create 'test_images' directory and add your images")
        print("2. Upload test images ZIP file")
        print("3. Run add_test_images.py to process ZIP file")

if __name__ == "__main__":
    main()