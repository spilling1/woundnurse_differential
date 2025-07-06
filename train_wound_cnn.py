#!/usr/bin/env python3
"""
Custom CNN Training for Wound Detection
Trains on your 696 wound images with 6 classes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WoundDataset(Dataset):
    """Dataset for wound detection training"""
    
    def __init__(self, dataset_path: str, split: str = 'train', img_size: int = 224):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.img_size = img_size
        
        # Load annotations
        annotations_file = self.dataset_path / split / "annotations.json"
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Load class mapping
        class_mapping_file = self.dataset_path / "class_mapping.json"
        with open(class_mapping_file, 'r') as f:
            self.class_mapping = json.load(f)
        
        self.num_classes = self.class_mapping['num_classes']
        self.class_to_idx = {name: idx for idx, name in self.class_mapping['classes'].items()}
        
        # Define transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        logger.info(f"Loaded {len(self.annotations)} {split} samples")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = self.dataset_path / self.split / "images" / annotation["image_name"]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        
        # Get label
        wound_type = annotation["wound_type"]
        label = self.class_to_idx[wound_type]
        
        return image, label

class WoundCNN(nn.Module):
    """Custom CNN for wound classification"""
    
    def __init__(self, num_classes: int = 6):
        super(WoundCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class WoundTrainer:
    """Training manager for wound CNN"""
    
    def __init__(self, dataset_path: str = "wound_dataset_balanced"):
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load datasets
        self.train_dataset = WoundDataset(dataset_path, 'train')
        self.val_dataset = WoundDataset(dataset_path, 'val')
        self.test_dataset = WoundDataset(dataset_path, 'test')
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        # Initialize model
        self.model = WoundCNN(num_classes=self.train_dataset.num_classes)
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        logger.info(f"Model initialized with {self.train_dataset.num_classes} classes")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, epochs: int = 50):
        """Main training loop"""
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f'Epoch {epoch+1}/{epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info(f'  Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f'best_wound_model_acc_{val_acc:.2f}.pth')
                logger.info(f'  New best model saved! Validation accuracy: {val_acc:.2f}%')
        
        total_time = time.time() - start_time
        logger.info(f'Training completed in {total_time:.2f}s')
        logger.info(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        # Final test
        self.test()
    
    def test(self):
        """Test the model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0] * self.train_dataset.num_classes
        class_total = [0] * self.train_dataset.num_classes
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Per-class accuracy
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        test_acc = 100. * correct / total
        logger.info(f'Test Accuracy: {test_acc:.2f}%')
        
        # Per-class accuracy
        class_names = self.train_dataset.class_mapping['class_names']
        for i in range(self.train_dataset.num_classes):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                logger.info(f'  {class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_mapping': self.train_dataset.class_mapping,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filename)
        logger.info(f'Model saved as {filename}')
    
    def plot_training_history(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        logger.info('Training history saved as training_history.png')

def main():
    """Main training function"""
    logger.info("Starting Wound CNN Training")
    logger.info("=" * 50)
    
    # Check if dataset exists
    if not Path("wound_dataset_balanced").exists():
        logger.error("Balanced dataset not found. Please run balanced_dataset_organizer.py first.")
        return
    
    # Initialize trainer
    trainer = WoundTrainer()
    
    # Start training
    trainer.train(epochs=30)  # Reduced epochs for faster training
    
    # Plot results
    trainer.plot_training_history()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()