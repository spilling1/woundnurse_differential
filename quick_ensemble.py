#!/usr/bin/env python3
"""
Quick ensemble training - multiple lightweight models for fast completion
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from quick_wound_trainer import QuickWoundDataset
import numpy as np
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MiniCNN1(nn.Module):
    """First variant - focus on small features"""
    def __init__(self, num_classes=6):
        super(MiniCNN1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

class MiniCNN2(nn.Module):
    """Second variant - focus on larger features"""
    def __init__(self, num_classes=6):
        super(MiniCNN2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

class MiniCNN3(nn.Module):
    """Third variant - deeper but narrow"""
    def __init__(self, num_classes=6):
        super(MiniCNN3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

def quick_train_model(model, model_name, epochs=3):
    """Quick training for a single model"""
    logger.info(f"Training {model_name}...")
    
    device = torch.device('cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Load small dataset
    train_dataset = QuickWoundDataset("wound_dataset_balanced", 'train', img_size=32)
    val_dataset = QuickWoundDataset("wound_dataset_balanced", 'val', img_size=32)
    
    # Even smaller for speed
    train_dataset.annotations = train_dataset.annotations[:60]
    val_dataset.annotations = val_dataset.annotations[:20]
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
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
        
        accuracy = 100 * correct / total if total > 0 else 0
        logger.info(f"{model_name} - Epoch {epoch+1}: {accuracy:.1f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    
    # Save model
    if best_accuracy > 0:
        model_path = f"mini_{model_name}_acc_{best_accuracy:.1f}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': best_accuracy,
            'model_name': model_name
        }, model_path)
        logger.info(f"Saved {model_name}: {best_accuracy:.1f}%")
        return model_path, best_accuracy
    
    return None, 0

def ensemble_predict(models, image_tensor, class_names):
    """Make ensemble prediction"""
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for model in models:
            output = model(image_tensor.unsqueeze(0))
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predictions.append(predicted.item())
            confidences.append(probabilities[0].numpy())
    
    # Majority vote
    majority_vote = Counter(predictions).most_common(1)[0][0]
    
    # Average confidence
    avg_confidences = np.mean(confidences, axis=0)
    ensemble_confidence = avg_confidences[majority_vote]
    
    return majority_vote, ensemble_confidence

def test_ensemble(model_paths):
    """Test ensemble performance"""
    logger.info("Testing ensemble...")
    
    # Load models
    models = []
    class_names = ["background", "diabetic_ulcer", "neuropathic_ulcer", 
                   "pressure_ulcer", "surgical_wound", "venous_ulcer"]
    
    for i, path in enumerate(model_paths):
        if 'mini_cnn1' in path:
            model = MiniCNN1()
        elif 'mini_cnn2' in path:
            model = MiniCNN2()
        elif 'mini_cnn3' in path:
            model = MiniCNN3()
        else:
            continue
        
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        logger.info(f"Loaded model {i+1}: {checkpoint.get('accuracy', 0):.1f}%")
    
    if len(models) < 2:
        logger.warning("Need at least 2 models for ensemble")
        return
    
    # Test on validation set
    test_dataset = QuickWoundDataset("wound_dataset_balanced", 'val', img_size=32)
    test_dataset.annotations = test_dataset.annotations[:15]  # Quick test
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        predicted_class, confidence = ensemble_predict(models, images[0], class_names)
        true_class = labels[0].item()
        
        total += 1
        if predicted_class == true_class:
            correct += 1
        
        logger.info(f"True: {class_names[true_class]}, Predicted: {class_names[predicted_class]}, Confidence: {confidence*100:.1f}%")
    
    ensemble_accuracy = 100 * correct / total if total > 0 else 0
    logger.info(f"Ensemble Accuracy: {ensemble_accuracy:.1f}%")
    return ensemble_accuracy

def main():
    """Train multiple models and create ensemble"""
    logger.info("Quick Ensemble Training")
    logger.info("=" * 40)
    
    models_to_train = [
        (MiniCNN1(), "cnn1"),
        (MiniCNN2(), "cnn2"),
        (MiniCNN3(), "cnn3")
    ]
    
    trained_models = []
    individual_accuracies = []
    
    for model, name in models_to_train:
        model_path, accuracy = quick_train_model(model, name, epochs=2)
        if model_path:
            trained_models.append(model_path)
            individual_accuracies.append(accuracy)
    
    logger.info(f"Trained {len(trained_models)} models")
    logger.info(f"Individual accuracies: {individual_accuracies}")
    
    if len(trained_models) >= 2:
        ensemble_accuracy = test_ensemble(trained_models)
        
        logger.info("\nSUMMARY:")
        logger.info(f"Individual models: {individual_accuracies}")
        logger.info(f"Ensemble accuracy: {ensemble_accuracy:.1f}%")
        
        if ensemble_accuracy > max(individual_accuracies):
            improvement = ensemble_accuracy - max(individual_accuracies)
            logger.info(f"Ensemble improvement: +{improvement:.1f}%")
        
        return trained_models, ensemble_accuracy
    else:
        logger.warning("Not enough models trained successfully")
        return [], 0

if __name__ == "__main__":
    main()