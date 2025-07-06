#!/usr/bin/env python3
"""
Wound Detection CNN Training System
Custom deep learning model for wound detection using your 730 images and body map data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import time
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WoundDataset(Dataset):
    """Dataset for wound detection with body map integration"""
    
    def __init__(self, data_config: Dict, split: str = 'train'):
        self.data_config = data_config
        self.split = split
        self.img_size = data_config.get('img_size', 640)
        
        # Class mappings for enhanced wound types
        self.wound_classes = {
            'pressure_ulcer_heel': 0,
            'pressure_ulcer_sacrum': 1, 
            'pressure_ulcer_shoulder': 2,
            'diabetic_foot_ulcer': 3,
            'venous_leg_ulcer': 4,
            'surgical_wound_chest': 5,
            'surgical_wound_abdomen': 6,
            'arterial_ulcer_leg': 7
        }
        
        self.samples = self.load_samples()
        logger.info(f"Loaded {len(self.samples)} samples for {split}")
    
    def load_samples(self) -> List[Dict]:
        """Load wound samples from your data structure"""
        samples = []
        
        # This will be adapted to your specific data format
        data_dir = Path(self.data_config['data_dir']) / self.split
        
        # Example structure - adapt to your format
        if (data_dir / 'annotations.json').exists():
            with open(data_dir / 'annotations.json', 'r') as f:
                annotations = json.load(f)
            
            for annotation in annotations:
                sample = {
                    'image_path': data_dir / 'images' / annotation['image_name'],
                    'wound_type': annotation.get('wound_type', 'unknown'),
                    'body_region': annotation.get('body_region', 'unknown'),
                    'bbox': annotation.get('bbox', [0.4, 0.4, 0.2, 0.2]),
                    'severity': annotation.get('severity', 'moderate'),
                    'size_mm2': annotation.get('size_mm2', 100)
                }
                
                # Create enhanced class
                enhanced_class = f"{sample['wound_type']}_{sample['body_region']}"
                if enhanced_class in self.wound_classes:
                    sample['class_id'] = self.wound_classes[enhanced_class]
                    samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = cv2.imread(str(sample['image_path']))
        if image is None:
            # Create dummy image if file missing
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Process labels
        class_id = sample.get('class_id', 0)
        bbox = torch.tensor(sample['bbox'], dtype=torch.float32)
        
        # Additional features from body map
        severity_score = {'mild': 0.3, 'moderate': 0.6, 'severe': 0.9}.get(
            sample.get('severity', 'moderate'), 0.6
        )
        
        return {
            'image': image,
            'class_id': torch.tensor(class_id, dtype=torch.long),
            'bbox': bbox,
            'severity': torch.tensor(severity_score, dtype=torch.float32)
        }

class WoundDetectionCNN(nn.Module):
    """Custom CNN for wound detection and classification"""
    
    def __init__(self, num_classes: int = 8, img_size: int = 640):
        super(WoundDetectionCNN, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 320x320
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 160x160
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 80x80
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 40x40
            
            # Block 5
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))  # 8x8
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # x, y, width, height
        )
        
        # Severity regression head
        self.severity_regressor = nn.Sequential(
            nn.Linear(512 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features_flat = features.view(features.size(0), -1)
        
        # Multi-task outputs
        classification = self.classifier(features_flat)
        bbox = self.bbox_regressor(features_flat)
        severity = self.severity_regressor(features_flat)
        
        return {
            'classification': classification,
            'bbox': bbox,
            'severity': severity
        }

class WoundCNNTrainer:
    """Training manager for wound detection CNN"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training on device: {self.device}")
        
        # Initialize model
        self.model = WoundDetectionCNN(
            num_classes=config['num_classes'],
            img_size=config['img_size']
        )
        self.model.to(self.device)
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.MSELoss()
        self.severity_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.get('step_size', 20), 
            gamma=config.get('gamma', 0.5)
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
    
    def prepare_data(self):
        """Prepare training and validation datasets"""
        
        # Create datasets
        train_dataset = WoundDataset(self.config, split='train')
        val_dataset = WoundDataset(self.config, split='val')
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            class_targets = batch['class_id'].to(self.device)
            bbox_targets = batch['bbox'].to(self.device)
            severity_targets = batch['severity'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate losses
            cls_loss = self.classification_loss(outputs['classification'], class_targets)
            bbox_loss = self.bbox_loss(outputs['bbox'], bbox_targets)
            sev_loss = self.severity_loss(outputs['severity'].squeeze(), severity_targets)
            
            # Combined loss
            total_batch_loss = cls_loss + 0.5 * bbox_loss + 0.3 * sev_loss
            
            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += total_batch_loss.item()
            _, predicted = outputs['classification'].max(1)
            total += class_targets.size(0)
            correct += predicted.eq(class_targets).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                          f'Loss: {total_batch_loss.item():.4f}, '
                          f'Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                class_targets = batch['class_id'].to(self.device)
                bbox_targets = batch['bbox'].to(self.device)
                severity_targets = batch['severity'].to(self.device)
                
                outputs = self.model(images)
                
                # Calculate losses
                cls_loss = self.classification_loss(outputs['classification'], class_targets)
                bbox_loss = self.bbox_loss(outputs['bbox'], bbox_targets)
                sev_loss = self.severity_loss(outputs['severity'].squeeze(), severity_targets)
                
                total_batch_loss = cls_loss + 0.5 * bbox_loss + 0.3 * sev_loss
                total_loss += total_batch_loss.item()
                
                # Accuracy
                _, predicted = outputs['classification'].max(1)
                total += class_targets.size(0)
                correct += predicted.eq(class_targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs: int):
        """Main training loop"""
        logger.info(f"Starting training for {epochs} epochs")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Epoch Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"best_wound_cnn_acc_{val_acc:.2f}.pth")
                logger.info(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pth")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Plot training history
        self.plot_training_history()
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        save_path = Path("models") / filename
        save_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, save_path)
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy curves
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        logger.info("Training history plot saved as training_history.png")

def setup_training_environment():
    """Set up directories and configuration for training"""
    
    # Create directory structure
    dirs = [
        "wound_dataset/train/images",
        "wound_dataset/train/annotations", 
        "wound_dataset/val/images",
        "wound_dataset/val/annotations",
        "models",
        "results"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Default configuration
    config = {
        'data_dir': 'wound_dataset',
        'num_classes': 8,  # Enhanced wound-location combinations
        'img_size': 640,
        'batch_size': 8,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'step_size': 20,
        'gamma': 0.5,
        'epochs': 50
    }
    
    return config

def main():
    """Main training function"""
    print("Wound Detection CNN Training System")
    print("=" * 50)
    
    # Setup environment
    config = setup_training_environment()
    
    print(f"Configuration: {config}")
    print("\nDataset structure created for your 730 wound images")
    print("Place your data in:")
    print("- wound_dataset/train/images/ (training images)")
    print("- wound_dataset/train/annotations.json (training labels)")
    print("- wound_dataset/val/images/ (validation images)")
    print("- wound_dataset/val/annotations.json (validation labels)")
    
    # Initialize trainer
    trainer = WoundCNNTrainer(config)
    
    print(f"\nModel architecture: Custom CNN with {config['num_classes']} wound classes")
    print("Ready to train on your body map integrated dataset")
    
    return trainer, config

if __name__ == "__main__":
    trainer, config = main()