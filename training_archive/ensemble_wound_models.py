#!/usr/bin/env python3
"""
Ensemble approach for improved wound classification accuracy
Multiple models working together for better performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleWoundDataset(Dataset):
    """Dataset for ensemble training with data augmentation"""
    
    def __init__(self, dataset_path: str, split: str = 'train', img_size: int = 64, augment: bool = False):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.img_size = img_size
        
        # Load annotations
        annotations_file = self.dataset_path / split / "annotations.json"
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Different transforms for ensemble diversity
        if augment and split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        self.class_to_idx = {
            "background": 0,
            "diabetic_ulcer": 1,
            "neuropathic_ulcer": 2,
            "pressure_ulcer": 3,
            "surgical_wound": 4,
            "venous_ulcer": 5
        }
        
        logger.info(f"Ensemble dataset loaded: {len(self.annotations)} {split} samples")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = self.dataset_path / self.split / "images" / annotation["image_name"]
        
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')
                image = self.transform(image)
            
            wound_type = annotation["wound_type"]
            label = self.class_to_idx[wound_type]
            return image, label
            
        except Exception as e:
            logger.warning(f"Error loading {image_path}: {e}")
            return torch.zeros(3, self.img_size, self.img_size), 0

class DeepCNN(nn.Module):
    """Deeper CNN architecture for better feature extraction"""
    
    def __init__(self, num_classes=6):
        super(DeepCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

class ResidualBlock(nn.Module):
    """Residual block for ResNet-style architecture"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = torch.relu(out)
        return out

class WoundResNet(nn.Module):
    """ResNet-style architecture for wound classification"""
    
    def __init__(self, num_classes=6):
        super(WoundResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class EnsemblePredictor:
    """Ensemble predictor combining multiple models"""
    
    def __init__(self, model_paths: list):
        self.models = []
        self.device = torch.device('cpu')
        
        # Load different model architectures
        for i, path in enumerate(model_paths):
            if 'deep' in path.lower():
                model = DeepCNN()
            elif 'resnet' in path.lower():
                model = WoundResNet()
            else:
                from quick_wound_trainer import TinyCNN
                model = TinyCNN()
            
            if Path(path).exists():
                checkpoint = torch.load(path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.models.append(model)
                logger.info(f"Loaded model {i+1}: {path}")
        
        self.class_names = ["background", "diabetic_ulcer", "neuropathic_ulcer", 
                           "pressure_ulcer", "surgical_wound", "venous_ulcer"]
    
    def predict(self, image_tensor):
        """Ensemble prediction using majority voting and confidence averaging"""
        if not self.models:
            logger.error("No models loaded for ensemble prediction")
            return None
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(image_tensor.unsqueeze(0))
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predictions.append(predicted.item())
                confidences.append(probabilities[0].numpy())
        
        # Majority voting
        majority_vote = Counter(predictions).most_common(1)[0][0]
        
        # Average confidence scores
        avg_confidences = np.mean(confidences, axis=0)
        ensemble_confidence = avg_confidences[majority_vote]
        
        return {
            'predicted_class': self.class_names[majority_vote],
            'confidence': ensemble_confidence * 100,
            'individual_predictions': [self.class_names[p] for p in predictions],
            'all_confidences': {self.class_names[i]: conf * 100 for i, conf in enumerate(avg_confidences)}
        }
    
    def evaluate_ensemble(self, test_loader):
        """Evaluate ensemble performance"""
        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(6)}
        class_total = {i: 0 for i in range(6)}
        
        for images, labels in test_loader:
            for i in range(len(images)):
                prediction = self.predict(images[i])
                if prediction:
                    predicted_idx = self.class_names.index(prediction['predicted_class'])
                    true_idx = labels[i].item()
                    
                    total += 1
                    class_total[true_idx] += 1
                    
                    if predicted_idx == true_idx:
                        correct += 1
                        class_correct[true_idx] += 1
        
        overall_accuracy = 100 * correct / total if total > 0 else 0
        
        # Per-class accuracy
        class_accuracies = {}
        for i in range(6):
            if class_total[i] > 0:
                class_accuracies[self.class_names[i]] = 100 * class_correct[i] / class_total[i]
            else:
                class_accuracies[self.class_names[i]] = 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies,
            'total_samples': total
        }

def train_multiple_models():
    """Train multiple different models for ensemble"""
    logger.info("Training multiple models for ensemble approach")
    
    dataset_path = "wound_dataset_balanced"
    if not Path(dataset_path).exists():
        logger.error("Dataset not found!")
        return []
    
    models_to_train = [
        ("deep_cnn", DeepCNN()),
        ("resnet", WoundResNet())
    ]
    
    trained_models = []
    
    for model_name, model in models_to_train:
        logger.info(f"Training {model_name}...")
        
        # Training setup
        device = torch.device('cpu')
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Load datasets with augmentation
        train_dataset = EnsembleWoundDataset(dataset_path, 'train', augment=True)
        val_dataset = EnsembleWoundDataset(dataset_path, 'val', augment=False)
        
        # Reduce dataset size for quick training
        train_dataset.annotations = train_dataset.annotations[:150]
        val_dataset.annotations = val_dataset.annotations[:40]
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Quick training - 3 epochs
        best_accuracy = 0.0
        for epoch in range(3):
            # Training
            model.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 5 == 0:
                    logger.info(f"{model_name} - Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            logger.info(f"{model_name} - Epoch {epoch+1} Accuracy: {accuracy:.2f}%")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_path = f"ensemble_{model_name}_acc_{accuracy:.1f}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'accuracy': accuracy,
                    'model_type': model_name
                }, model_path)
                trained_models.append(model_path)
                logger.info(f"Saved {model_name}: {model_path}")
    
    return trained_models

def main():
    """Main function to demonstrate ensemble approach"""
    # Train multiple models
    model_paths = train_multiple_models()
    
    # Add existing quick model
    existing_models = list(Path('.').glob('quick_wound_model_*.pth'))
    if existing_models:
        model_paths.extend([str(p) for p in existing_models])
    
    if len(model_paths) >= 2:
        logger.info(f"Creating ensemble with {len(model_paths)} models")
        
        # Create ensemble
        ensemble = EnsemblePredictor(model_paths)
        
        # Test ensemble
        test_dataset = EnsembleWoundDataset("wound_dataset_balanced", 'val')
        test_dataset.annotations = test_dataset.annotations[:20]  # Quick test
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        
        results = ensemble.evaluate_ensemble(test_loader)
        
        logger.info("Ensemble Results:")
        logger.info(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        logger.info("Per-class Accuracies:")
        for class_name, accuracy in results['class_accuracies'].items():
            logger.info(f"  {class_name}: {accuracy:.2f}%")
    else:
        logger.warning("Not enough models for ensemble. Need at least 2 models.")

if __name__ == "__main__":
    main()