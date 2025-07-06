"""
Efficient Ensemble Training for Enhanced Wound Detection
Optimized for Replit environment with your additional data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from collections import Counter
import zipfile
import shutil
import glob

class WoundDataset(Dataset):
    def __init__(self, dataset_path, split='train', img_size=64):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.class_names = [
            "background", "diabetic_ulcer", "neuropathic_ulcer", 
            "pressure_ulcer", "surgical_wound", "venous_ulcer"
        ]
        
        self.load_dataset(split)
    
    def load_dataset(self, split):
        split_dir = os.path.join(self.dataset_path, split)
        self.images = []
        self.labels = []
        
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
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except:
            # Fallback for corrupted images
            image = torch.zeros(3, self.img_size, self.img_size)
            return image, label

class DeepCNN(nn.Module):
    """Deeper CNN for better accuracy"""
    def __init__(self, num_classes=6):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

class WideCNN(nn.Module):
    """Wide CNN for different perspective"""
    def __init__(self, num_classes=6):
        super(WideCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

class CompactCNN(nn.Module):
    """Compact CNN for speed"""
    def __init__(self, num_classes=6):
        super(CompactCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

def integrate_test_data():
    """Integrate new test data into training"""
    print("Integrating new test data...")
    
    test_zip = "attached_assets/Test_1751778753808.zip"
    if not os.path.exists(test_zip):
        print("Test data not found, using existing dataset")
        return "wound_dataset_final"
    
    # Extract test data
    with zipfile.ZipFile(test_zip, 'r') as zip_ref:
        zip_ref.extractall("new_test_data")
    
    # Create enhanced dataset
    enhanced_dir = "wound_dataset_enhanced"
    if not os.path.exists(enhanced_dir):
        shutil.copytree("wound_dataset_final", enhanced_dir)
    
    # Add new images to training
    train_dir = os.path.join(enhanced_dir, 'train')
    
    # Process new images
    new_count = 0
    for root, dirs, files in os.walk("new_test_data"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                
                # Simple categorization based on filename
                if 'diabetic' in file.lower() or 'foot' in file.lower():
                    category = 'diabetic_ulcer'
                elif 'pressure' in file.lower():
                    category = 'pressure_ulcer'
                elif 'venous' in file.lower():
                    category = 'venous_ulcer'
                elif 'surgical' in file.lower():
                    category = 'surgical_wound'
                elif 'neuro' in file.lower():
                    category = 'neuropathic_ulcer'
                else:
                    category = 'background'
                
                dst_path = os.path.join(train_dir, category, f"enhanced_{new_count}_{file}")
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                new_count += 1
    
    print(f"Added {new_count} new images to training dataset")
    return enhanced_dir

def train_single_model(model_class, model_name, dataset_path, epochs=20):
    """Train a single model efficiently"""
    print(f"\nTraining {model_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_dataset = WoundDataset(dataset_path, 'train')
    if len(train_dataset) == 0:
        print("No training data found!")
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        print(f'{model_name} - Epoch {epoch+1}/{epochs}: Acc {accuracy:.1f}%')
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'class_names': train_dataset.class_names
            }, f'{model_name}_acc_{accuracy:.1f}.pth')
    
    print(f'{model_name} final accuracy: {best_acc:.1f}%')
    return f'{model_name}_acc_{best_acc:.1f}.pth'

def main():
    """Main training function"""
    print("=== Enhanced Ensemble Training ===")
    
    # Integrate new data
    dataset_path = integrate_test_data()
    
    # Train diverse models
    models = [
        (DeepCNN, "deep_cnn_v2"),
        (WideCNN, "wide_cnn_v2"), 
        (CompactCNN, "compact_cnn_v2")
    ]
    
    trained_models = []
    for model_class, model_name in models:
        result = train_single_model(model_class, model_name, dataset_path, epochs=15)
        if result:
            trained_models.append(result)
    
    # List all models
    print(f"\n=== Training Complete ===")
    all_models = glob.glob("*_acc_*.pth")
    print(f"Available models ({len(all_models)}):")
    
    # Sort by accuracy
    model_accs = []
    for model in all_models:
        try:
            acc = float(model.split('_acc_')[1].split('.pth')[0])
            model_accs.append((acc, model))
        except:
            model_accs.append((0, model))
    
    for acc, model in sorted(model_accs, reverse=True):
        print(f"  {model} ({acc:.1f}% accuracy)")
    
    print(f"\nBest ensemble candidates:")
    top_models = sorted(model_accs, reverse=True)[:3]
    for acc, model in top_models:
        print(f"  {model}")

if __name__ == "__main__":
    main()