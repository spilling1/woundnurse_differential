#!/usr/bin/env python3
"""
Quick CNN Training for Wound Detection - Optimized for fast completion
Reduced epochs and batch processing to complete before environment interruption
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickWoundDataset(Dataset):
    """Fast-loading dataset for wound detection"""
    
    def __init__(self, dataset_path: str, split: str = 'train', img_size: int = 32):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.img_size = img_size
        
        # Load annotations
        annotations_file = self.dataset_path / split / "annotations.json"
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Reduce dataset size for quick training
        if split == 'train':
            self.annotations = self.annotations[:100]  # Only use 100 samples
        elif split == 'val':
            self.annotations = self.annotations[:30]   # Only use 30 samples
        
        # Define transforms - smaller size for speed
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Class mapping
        self.class_to_idx = {
            "background": 0,
            "diabetic_ulcer": 1,
            "neuropathic_ulcer": 2,
            "pressure_ulcer": 3,
            "surgical_wound": 4,
            "venous_ulcer": 5
        }
        
        logger.info(f"Quick dataset loaded: {len(self.annotations)} {split} samples")
    
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

class TinyCNN(nn.Module):
    """Ultra-lightweight CNN for quick training"""
    
    def __init__(self, num_classes=6):
        super(TinyCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Single conv block
            nn.Conv2d(3, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, stride=4),  # Aggressive pooling
            
            # Second conv block
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

def quick_train():
    """Quick training function - designed to complete in under 5 minutes"""
    logger.info("Starting Quick Wound CNN Training")
    logger.info("=" * 50)
    
    # Check dataset
    dataset_path = "wound_dataset_balanced"
    if not Path(dataset_path).exists():
        logger.error("Dataset not found!")
        return
    
    # Initialize model and training
    device = torch.device('cpu')
    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Higher learning rate
    
    # Load datasets with reduced size
    train_dataset = QuickWoundDataset(dataset_path, 'train', img_size=32)
    val_dataset = QuickWoundDataset(dataset_path, 'val', img_size=32)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Quick training - only 5 epochs
    num_epochs = 5
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 3 == 0:  # Log every 3 batches
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
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
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s")
        logger.info(f"Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = f"quick_wound_model_acc_{accuracy:.1f}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch + 1,
                'class_to_idx': train_dataset.class_to_idx
            }, model_path)
            logger.info(f"New best model saved: {model_path}")
    
    logger.info(f"Quick training completed! Best accuracy: {best_accuracy:.2f}%")
    return model_path if 'model_path' in locals() else None

def test_quick_model(model_path: str):
    """Test the quick model on a few samples"""
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = TinyCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test on a few validation samples
    val_dataset = QuickWoundDataset("wound_dataset_balanced", 'val', img_size=32)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=True)
    
    class_names = ["background", "diabetic_ulcer", "neuropathic_ulcer", 
                   "pressure_ulcer", "surgical_wound", "venous_ulcer"]
    
    logger.info("Testing model on sample images:")
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(min(3, len(images))):  # Test 3 samples
                true_class = class_names[labels[i]]
                pred_class = class_names[predicted[i]]
                confidence = probabilities[i][predicted[i]].item() * 100
                
                logger.info(f"Sample {i+1}: True={true_class}, Predicted={pred_class}, Confidence={confidence:.1f}%")
            break
    
    logger.info(f"Model accuracy: {checkpoint['accuracy']:.2f}%")

if __name__ == "__main__":
    start_time = time.time()
    
    # Run quick training
    model_path = quick_train()
    
    # Test the model if training succeeded
    if model_path:
        test_quick_model(model_path)
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.1f} seconds")