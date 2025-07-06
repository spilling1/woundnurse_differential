#!/usr/bin/env python3
"""
Minimal CNN trainer for wound classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import time
from pathlib import Path
from PIL import Image

class WoundDataset(Dataset):
    def __init__(self, dataset_path, split='train'):
        self.dataset_path = Path(dataset_path)
        self.split = split
        
        # Load annotations
        with open(self.dataset_path / split / "annotations.json", 'r') as f:
            self.annotations = json.load(f)
        
        # Load class mapping
        with open(self.dataset_path / "class_mapping.json", 'r') as f:
            self.class_mapping = json.load(f)
        
        self.class_to_idx = {name: int(idx) for idx, name in self.class_mapping['classes'].items()}
        
        # Simple transforms
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Small size for fast training
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = self.dataset_path / self.split / "images" / annotation["image_name"]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            wound_type = annotation["wound_type"]
            label = self.class_to_idx[wound_type]
            return image, label
        except:
            return torch.zeros(3, 64, 64), 0

class SimpleCNN(nn.Module):
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

def train_model():
    print("Starting Minimal Wound CNN Training")
    print("==================================")
    
    # Check dataset
    if not Path("wound_dataset_balanced").exists():
        print("Error: Dataset not found")
        return
    
    # Load data
    train_dataset = WoundDataset("wound_dataset_balanced", 'train')
    val_dataset = WoundDataset("wound_dataset_balanced", 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Model setup
    model = SimpleCNN(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    
    # Training loop
    for epoch in range(10):  # Reduced epochs
        print(f"\nEpoch {epoch+1}/10")
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        epoch_time = time.time() - start_time
        
        print(f"  Train: Loss {train_loss/len(train_loader):.4f}, Acc {train_acc:.1f}%")
        print(f"  Val: Loss {val_loss/len(val_loader):.4f}, Acc {val_acc:.1f}%")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_mapping': train_dataset.class_mapping
            }, f'wound_model_{val_acc:.1f}acc.pth')
            print(f"  NEW BEST MODEL SAVED: {val_acc:.1f}% accuracy")
    
    print(f"\nTraining Complete! Best accuracy: {best_acc:.1f}%")
    return best_acc

if __name__ == "__main__":
    try:
        best_acc = train_model()
        if best_acc > 0:
            print("SUCCESS: Model trained and saved")
        else:
            print("FAILED: Training unsuccessful")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()