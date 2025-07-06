"""
Quick Ensemble Training - Efficient approach for Replit environment
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
from collections import Counter

class QuickDataset(Dataset):
    def __init__(self, dataset_path, split='train'):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.class_names = [
            "background", "diabetic_ulcer", "neuropathic_ulcer", 
            "pressure_ulcer", "surgical_wound", "venous_ulcer"
        ]
        
        self.images = []
        self.labels = []
        self.load_dataset(dataset_path, split)
    
    def load_dataset(self, dataset_path, split):
        split_dir = os.path.join(dataset_path, split)
        if os.path.exists(split_dir):
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.exists(class_dir):
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.images.append(os.path.join(class_dir, img_file))
                            self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images for {split}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            image = self.transform(image)
            return image, self.labels[idx]
        except:
            return torch.zeros(3, 64, 64), self.labels[idx]

class CompactCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CompactCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

def train_quick_model(model_name, dataset_path, epochs=10):
    """Train a model quickly with the new data"""
    print(f"Training {model_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_dataset = QuickDataset(dataset_path, 'train')
    if len(train_dataset) == 0:
        print("No training data found!")
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model
    model = CompactCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}: {accuracy:.1f}%')
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'class_names': train_dataset.class_names
            }, f'{model_name}_acc_{accuracy:.1f}.pth')
    
    print(f'Final accuracy: {best_acc:.1f}%')
    return f'{model_name}_acc_{best_acc:.1f}.pth'

def main():
    """Quick training with body context data"""
    print("=== Quick Ensemble Training ===")
    
    # Use the body context dataset
    dataset_path = "wound_dataset_body_context"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
    
    # Train multiple quick models
    models = [
        "body_context_v1",
        "body_context_v2",
        "body_context_v3"
    ]
    
    for model_name in models:
        train_quick_model(model_name, dataset_path, epochs=8)
    
    # Show results
    print("\n=== Training Results ===")
    all_models = glob.glob("*_acc_*.pth")
    
    model_scores = []
    for model in all_models:
        try:
            acc = float(model.split('_acc_')[1].split('.pth')[0])
            model_scores.append((acc, model))
        except:
            continue
    
    print(f"All models ({len(model_scores)}):")
    for acc, model in sorted(model_scores, reverse=True):
        print(f"  {model} - {acc:.1f}%")
    
    # Update the CNN classifier to use the best model
    if model_scores:
        best_model = sorted(model_scores, reverse=True)[0][1]
        print(f"\nBest model: {best_model}")
        print(f"This model will be available for the wound detection system.")

if __name__ == "__main__":
    main()