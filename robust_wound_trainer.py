#!/usr/bin/env python3
"""
Robust CNN Training for Wound Detection - Optimized for Replit Environment
Includes checkpoint saving, memory management, and auto-restart capabilities
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
import os
import gc
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WoundDataset(Dataset):
    """Memory-efficient dataset for wound detection training"""
    
    def __init__(self, dataset_path: str, split: str = 'train', img_size: int = 64):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.img_size = img_size
        
        # Load annotations
        annotations_file = self.dataset_path / split / "annotations.json"
        if not annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
            
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Define transforms
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
        
        logger.info(f"Loaded {len(self.annotations)} {split} samples")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = self.dataset_path / self.split / "images" / annotation["image_name"]
        
        try:
            # Load and process image
            with Image.open(image_path) as img:
                image = img.convert('RGB')
                image = self.transform(image)
            
            # Get label
            wound_type = annotation["wound_type"]
            label = self.class_to_idx[wound_type]
            
            return image, label
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return zero tensor and background label as fallback
            return torch.zeros(3, self.img_size, self.img_size), 0

class SimpleCNN(nn.Module):
    """Lightweight CNN optimized for memory efficiency"""
    
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1 - Reduced channels
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 2
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Block 3
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

class RobustTrainer:
    """Robust training manager with checkpoint saving and memory management"""
    
    def __init__(self, dataset_path: str = "wound_dataset_balanced"):
        self.dataset_path = dataset_path
        self.device = torch.device('cpu')  # Use CPU for stability
        
        # Training parameters
        self.batch_size = 8  # Reduced batch size for memory
        self.learning_rate = 0.001
        self.num_epochs = 20
        self.checkpoint_interval = 5  # Save every 5 epochs
        
        # Initialize model
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
        logger.info(f"Initialized trainer - Device: {self.device}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_accuracy = checkpoint['best_accuracy']
            self.train_losses = checkpoint['train_losses']
            self.val_accuracies = checkpoint['val_accuracies']
            logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            return True
        return False
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint at epoch {self.current_epoch}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch with memory management"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Memory cleanup
            del images, labels, outputs
            if batch_idx % 10 == 0:
                gc.collect()
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Memory cleanup
                del images, labels, outputs
        
        accuracy = 100 * correct / total
        self.val_accuracies.append(accuracy)
        return accuracy
    
    def train(self):
        """Main training loop with robustness features"""
        logger.info("Starting robust wound CNN training")
        logger.info("=" * 50)
        
        try:
            # Load datasets
            train_dataset = WoundDataset(self.dataset_path, 'train')
            val_dataset = WoundDataset(self.dataset_path, 'val')
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            
            # Try to load existing checkpoint
            checkpoint_path = "training_checkpoint.pth"
            if self.load_checkpoint(checkpoint_path):
                logger.info(f"Resuming training from epoch {self.current_epoch}")
            
            # Training loop
            for epoch in range(self.current_epoch, self.num_epochs):
                self.current_epoch = epoch + 1
                
                # Training
                train_loss = self.train_epoch(train_loader)
                
                # Validation
                val_accuracy = self.validate(val_loader)
                
                logger.info(f"Epoch {self.current_epoch}/{self.num_epochs} - Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                
                # Save best model
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    self.save_model(f"best_wound_model_acc_{val_accuracy:.2f}.pth")
                
                # Save checkpoint
                if self.current_epoch % self.checkpoint_interval == 0:
                    self.save_checkpoint(checkpoint_path)
                
                # Memory cleanup
                gc.collect()
                
                # Small pause to prevent overheating
                time.sleep(1)
            
            logger.info(f"Training completed! Best accuracy: {self.best_accuracy:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Save emergency checkpoint
            self.save_checkpoint("emergency_checkpoint.pth")
            return False
    
    def save_model(self, filename: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'accuracy': self.best_accuracy,
            'class_to_idx': {
                "background": 0,
                "diabetic_ulcer": 1,
                "neuropathic_ulcer": 2,
                "pressure_ulcer": 3,
                "surgical_wound": 4,
                "venous_ulcer": 5
            }
        }, filename)
        logger.info(f"Saved model: {filename}")

def main():
    """Main training function"""
    # Set memory optimization
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Check if dataset exists
    if not Path("wound_dataset_balanced").exists():
        logger.error("Dataset not found. Please run balanced_dataset_organizer.py first")
        return
    
    # Start training
    trainer = RobustTrainer()
    success = trainer.train()
    
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed - check logs for details")

if __name__ == "__main__":
    main()