#!/usr/bin/env python3
"""
Comprehensive test evaluation for wound CNN using your labeled test dataset
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
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

class ComprehensiveEvaluator:
    """Comprehensive evaluation of wound detection model"""
    
    def __init__(self, model_path: str, test_images_dir: str, test_labels_csv: str):
        self.device = torch.device('cpu')
        self.test_images_dir = Path(test_images_dir)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_mapping = checkpoint['class_mapping']
        self.model_accuracy = checkpoint.get('val_acc', 'unknown')
        
        # Class mappings
        self.label_to_class = {
            0: "background",
            1: "diabetic_ulcer", 
            2: "neuropathic_ulcer",
            3: "pressure_ulcer",
            4: "surgical_wound",
            5: "venous_ulcer"
        }
        
        self.class_to_label = {v: k for k, v in self.label_to_class.items()}
        
        # Load and prepare model
        self.model = SimpleCNN(num_classes=6)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load test labels
        self.test_df = pd.read_csv(test_labels_csv)
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        logger.info(f"Model loaded: {model_path}")
        logger.info(f"Model validation accuracy: {self.model_accuracy}%")
        logger.info(f"Test dataset: {len(self.test_df)} labeled images")
    
    def predict_image(self, image_path: str):
        """Predict single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                return {
                    'predicted_label': predicted_idx.item(),
                    'confidence': confidence.item(),
                    'probabilities': probabilities[0].numpy()
                }
        except Exception as e:
            logger.error(f"Error predicting {image_path}: {e}")
            return None
    
    def evaluate_test_set(self):
        """Evaluate entire test set with ground truth labels"""
        results = []
        y_true = []
        y_pred = []
        confidences = []
        
        for _, row in self.test_df.iterrows():
            # Get image path
            image_name = row['index'].replace('\\', '/').split('/')[-1] + '.jpg'
            image_path = self.test_images_dir / image_name
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Get ground truth
            true_label = row['Labels']
            true_class = self.label_to_class[true_label]
            
            # Make prediction
            prediction = self.predict_image(str(image_path))
            
            if prediction:
                pred_label = prediction['predicted_label']
                pred_class = self.label_to_class[pred_label]
                confidence = prediction['confidence']
                
                results.append({
                    'image_name': image_name,
                    'true_label': true_label,
                    'true_class': true_class,
                    'predicted_label': pred_label,
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'correct': true_label == pred_label,
                    'body_region_id': row['Locations']
                })
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                confidences.append(confidence)
        
        return results, y_true, y_pred, confidences
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate detailed classification report"""
        class_names = [self.label_to_class[i] for i in range(6)]
        
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        return report
    
    def create_confusion_matrix(self, y_true, y_pred, save_path="confusion_matrix.png"):
        """Create and save confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        class_names = [self.label_to_class[i] for i in range(6)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Wound Classification Confusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved: {save_path}")
    
    def analyze_per_class_performance(self, results):
        """Analyze performance per wound class"""
        class_performance = {}
        
        for label in range(6):
            class_name = self.label_to_class[label]
            class_results = [r for r in results if r['true_label'] == label]
            
            if class_results:
                correct = sum(1 for r in class_results if r['correct'])
                total = len(class_results)
                accuracy = correct / total if total > 0 else 0
                avg_confidence = np.mean([r['confidence'] for r in class_results])
                
                class_performance[class_name] = {
                    'total_images': total,
                    'correct_predictions': correct,
                    'accuracy': accuracy,
                    'average_confidence': avg_confidence
                }
        
        return class_performance
    
    def generate_comprehensive_report(self):
        """Generate complete evaluation report"""
        logger.info("Starting comprehensive evaluation...")
        
        # Evaluate test set
        results, y_true, y_pred, confidences = self.evaluate_test_set()
        
        if not results:
            logger.error("No results generated - check image paths")
            return None
        
        # Calculate overall metrics
        total_images = len(results)
        correct_predictions = sum(1 for r in results if r['correct'])
        overall_accuracy = correct_predictions / total_images if total_images > 0 else 0
        average_confidence = np.mean(confidences)
        
        # Generate reports
        classification_rep = self.generate_classification_report(y_true, y_pred)
        per_class_performance = self.analyze_per_class_performance(results)
        
        # Create confusion matrix
        self.create_confusion_matrix(y_true, y_pred)
        
        # Compile comprehensive report
        comprehensive_report = {
            'model_info': {
                'validation_accuracy': self.model_accuracy,
                'test_accuracy': overall_accuracy * 100,
                'total_test_images': total_images,
                'correct_predictions': correct_predictions
            },
            'overall_metrics': {
                'accuracy': overall_accuracy,
                'average_confidence': average_confidence,
                'macro_avg_precision': classification_rep['macro avg']['precision'],
                'macro_avg_recall': classification_rep['macro avg']['recall'],
                'macro_avg_f1': classification_rep['macro avg']['f1-score']
            },
            'per_class_performance': per_class_performance,
            'classification_report': classification_rep,
            'detailed_results': results
        }
        
        # Save report
        with open('comprehensive_test_report.json', 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        return comprehensive_report
    
    def print_summary(self, report):
        """Print evaluation summary"""
        print("\n🎯 COMPREHENSIVE WOUND MODEL EVALUATION")
        print("=" * 50)
        
        model_info = report['model_info']
        print(f"Model Validation Accuracy: {model_info['validation_accuracy']}%")
        print(f"Test Set Accuracy: {model_info['test_accuracy']:.1f}%")
        print(f"Test Images Processed: {model_info['total_test_images']}")
        print(f"Correct Predictions: {model_info['correct_predictions']}")
        
        print(f"\n📊 Overall Performance:")
        overall = report['overall_metrics']
        print(f"  Accuracy: {overall['accuracy']:.3f}")
        print(f"  Average Confidence: {overall['average_confidence']:.3f}")
        print(f"  Macro Avg Precision: {overall['macro_avg_precision']:.3f}")
        print(f"  Macro Avg Recall: {overall['macro_avg_recall']:.3f}")
        print(f"  Macro Avg F1-Score: {overall['macro_avg_f1']:.3f}")
        
        print(f"\n🔍 Per-Class Performance:")
        for class_name, perf in report['per_class_performance'].items():
            print(f"  {class_name}:")
            print(f"    Images: {perf['total_images']}")
            print(f"    Accuracy: {perf['accuracy']:.3f}")
            print(f"    Avg Confidence: {perf['average_confidence']:.3f}")

def find_best_model():
    """Find the best trained model"""
    model_files = list(Path('.').glob('wound_model_*.pth'))
    if not model_files:
        return None
    
    def get_accuracy(filename):
        try:
            acc_str = filename.stem.split('_')[-1].replace('acc', '')
            return float(acc_str)
        except:
            return 0.0
    
    return str(max(model_files, key=get_accuracy))

def main():
    """Main evaluation function"""
    print("Comprehensive Wound Model Test Evaluation")
    print("=" * 45)
    
    # Check for trained model
    model_path = find_best_model()
    if not model_path:
        print("⏳ Waiting for model training to complete...")
        print("Run this script again once training finishes")
        return
    
    print(f"📁 Using model: {model_path}")
    
    # Check for test data
    test_images_dir = "test_images_clean"
    test_labels_csv = "test_images_raw/Test/wound_locations_Labels_AZH_Test.csv"
    
    if not Path(test_images_dir).exists():
        print(f"❌ Test images not found: {test_images_dir}")
        return
    
    if not Path(test_labels_csv).exists():
        print(f"❌ Test labels not found: {test_labels_csv}")
        return
    
    # Run comprehensive evaluation
    try:
        evaluator = ComprehensiveEvaluator(model_path, test_images_dir, test_labels_csv)
        report = evaluator.generate_comprehensive_report()
        
        if report:
            evaluator.print_summary(report)
            print(f"\n💾 Detailed report saved: comprehensive_test_report.json")
            print(f"📊 Confusion matrix saved: confusion_matrix.png")
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()