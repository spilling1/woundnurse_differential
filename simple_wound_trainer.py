#!/usr/bin/env python3
"""
Simple CNN Training for Wound Detection - Optimized for CPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import numpy as np
from pathlib import Path
from PIL import Image
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_progress.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleWoundDataset(Dataset):
    """Simplified dataset for wound detection"""
    
    def __init__(self, dataset_path: str, split: str = 'train'):
        self.dataset_path = Path(dataset_path)
        self.split = split
        
        # Load annotations
        annotations_file = self.dataset_path / split / "annotations.json"
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Load class mapping
        class_mapping_file = self.dataset_path / "class_mapping.json"
        with open(class_mapping_file, 'r') as f:
            self.class_mapping = json.load(f)
        
        self.class_to_idx = {name: int(idx) for idx, name in self.class_mapping['classes'].items()}
        
        # Simple transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),  # Smaller size for faster training
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        logger.info(f"Loaded {len(self.annotations)} {split} samples")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = self.dataset_path / self.split / "images" / annotation["image_name"]
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            # Get label
            wound_type = annotation["wound_type"]
            label = self.class_to_idx[wound_type]
            
            return image, label
        except Exception as e:
            logger.warning(f"Error loading {image_path}: {e}")
            # Return a dummy tensor if image fails to load
            return torch.zeros(3, 128, 128), 0

class SimpleWoundCNN(nn.Module):
    """Lightweight CNN for wound classification"""
    
    def __init__(self, num_classes: int = 6):
        super(SimpleWoundCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model():
    """Main training function"""
    logger.info("Starting Simple Wound CNN Training")
    logger.info("=" * 50)
    
    # Check dataset
    if not Path("wound_dataset_balanced").exists():
        logger.error("Dataset not found. Please run balanced_dataset_organizer.py first.")
        return
    
    # Device
    device = torch.device('cpu')  # Force CPU for stability
    logger.info(f"Using device: {device}")
    
    # Load datasets
    train_dataset = SimpleWoundDataset("wound_dataset_balanced", 'train')
    val_dataset = SimpleWoundDataset("wound_dataset_balanced", 'val')
    
    # Data loaders (smaller batch size for CPU)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Model
    model = SimpleWoundCNN(num_classes=6)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info("Starting training...")
    
    # Training loop
    best_val_acc = 0.0
    epochs = 20  # Reduced for faster training
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        epoch_time = time.time() - epoch_start
        
        logger.info(f'Epoch {epoch+1}/{epochs} Results:')
        logger.info(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        logger.info(f'  Time: {epoch_time:.2f}s')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_mapping': train_dataset.class_mapping
            }, f'best_wound_model_{val_acc:.1f}acc.pth')
            logger.info(f'  New best model saved! Validation accuracy: {val_acc:.2f}%')
        
        logger.info("-" * 50)
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Test the model
    test_dataset = SimpleWoundDataset("wound_dataset_balanced", 'test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
    
    test_acc = 100. * test_correct / test_total
    logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
    
    return best_val_acc, test_acc

if __name__ == "__main__":
    train_model()