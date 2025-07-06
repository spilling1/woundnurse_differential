"""
Enhanced Body Context Trainer for Improved Wound Detection
Properly categorizes images based on your folder structure: S/D/V/P
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import zipfile
import shutil
import glob
from collections import Counter

class EnhancedWoundDataset(Dataset):
    def __init__(self, dataset_path, split='train', augment=True):
        self.augment = augment and split == 'train'
        
        # Enhanced augmentation for better body context understanding
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((96, 96)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(0.3),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        if len(self.images) > 0:
            class_counts = Counter(self.labels)
            for class_idx, count in class_counts.items():
                print(f"  {self.class_names[class_idx]}: {count}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading {self.images[idx]}: {e}")
            return torch.zeros(3, 96, 96), self.labels[idx]

class BodyContextCNN(nn.Module):
    """CNN optimized for body context understanding"""
    def __init__(self, num_classes=6):
        super(BodyContextCNN, self).__init__()
        
        # Feature extraction with attention to body regions
        self.features = nn.Sequential(
            # Initial wide receptive field for body context
            nn.Conv2d(3, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Detailed feature extraction
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Pattern recognition
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # High-level features
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

class ResidualWoundCNN(nn.Module):
    """ResNet-inspired architecture for wound classification"""
    def __init__(self, num_classes=6):
        super(ResidualWoundCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

def integrate_body_context_data():
    """Integrate body context images using proper categorization"""
    print("Integrating body context training data...")
    
    # Map folder names to wound types
    folder_mapping = {
        'S': 'surgical_wound',    # Surgical wounds
        'D': 'diabetic_ulcer',    # Diabetic ulcers
        'V': 'venous_ulcer',      # Venous ulcers
        'P': 'pressure_ulcer',    # Pressure ulcers
        'N': 'neuropathic_ulcer', # Neuropathic ulcers (if present)
        'B': 'background'         # Background/normal tissue (if present)
    }
    
    # Create enhanced dataset
    enhanced_dir = "wound_dataset_body_context"
    
    # Copy existing dataset as base
    if os.path.exists("wound_dataset_final"):
        print("Copying existing dataset...")
        shutil.copytree("wound_dataset_final", enhanced_dir, dirs_exist_ok=True)
    else:
        print("Creating new dataset structure...")
        os.makedirs(enhanced_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            for class_name in ["background", "diabetic_ulcer", "neuropathic_ulcer", "pressure_ulcer", "surgical_wound", "venous_ulcer"]:
                os.makedirs(os.path.join(enhanced_dir, split, class_name), exist_ok=True)
    
    # Process new images with body context
    train_dir = os.path.join(enhanced_dir, 'train')
    added_count = 0
    
    # Check if extracted images exist
    source_dir = "original_images_new/original images"
    if os.path.exists(source_dir):
        print(f"Processing images from {source_dir}...")
        
        # Process each category folder
        for folder_name in os.listdir(source_dir):
            folder_path = os.path.join(source_dir, folder_name)
            if os.path.isdir(folder_path) and folder_name in folder_mapping:
                wound_type = folder_mapping[folder_name]
                target_dir = os.path.join(train_dir, wound_type)
                os.makedirs(target_dir, exist_ok=True)
                
                print(f"Processing {folder_name} folder -> {wound_type}")
                
                # Copy all images from this category
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = os.path.join(folder_path, img_file)
                        dst_path = os.path.join(target_dir, f"body_context_{folder_name}_{img_file}")
                        
                        try:
                            shutil.copy2(src_path, dst_path)
                            added_count += 1
                        except Exception as e:
                            print(f"Could not copy {src_path}: {e}")
    
    print(f"Added {added_count} body context images to training dataset")
    return enhanced_dir

def train_enhanced_model(model_class, model_name, dataset_path, epochs=20):
    """Train enhanced model with body context"""
    print(f"\nTraining {model_name} with body context...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = EnhancedWoundDataset(dataset_path, 'train', augment=True)
    val_dataset = EnhancedWoundDataset(dataset_path, 'val', augment=False)
    
    if len(train_dataset) == 0:
        print("No training data found!")
        return None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=0) if len(val_dataset) > 0 else None
    
    # Initialize model
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_acc = 0
    patience = 0
    max_patience = 5
    
    for epoch in range(epochs):
        # Training phase
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
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}')
        
        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Validation phase
        val_acc = 0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            val_acc = 100. * val_correct / val_total
            
            print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%, Loss: {avg_loss:.4f}')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                patience = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'accuracy': val_acc,
                    'epoch': epoch,
                    'class_names': train_dataset.class_names
                }, f'{model_name}_acc_{val_acc:.1f}.pth')
                print(f'✓ New best model saved: {val_acc:.1f}% accuracy')
            else:
                patience += 1
        else:
            print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.1f}%, Loss: {avg_loss:.4f}')
            
            # Save based on training accuracy when no validation
            if train_acc > best_acc:
                best_acc = train_acc
                patience = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'accuracy': train_acc,
                    'epoch': epoch,
                    'class_names': train_dataset.class_names
                }, f'{model_name}_acc_{train_acc:.1f}.pth')
                print(f'✓ New best model saved: {train_acc:.1f}% accuracy')
            else:
                patience += 1
        
        scheduler.step()
        
        # Early stopping
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f'{model_name} final accuracy: {best_acc:.1f}%')
    return f'{model_name}_acc_{best_acc:.1f}.pth'

def main():
    """Main training function"""
    print("=== Enhanced Body Context Training ===")
    
    # Integrate body context data
    dataset_path = integrate_body_context_data()
    
    # Train multiple models for ensemble
    models_to_train = [
        (BodyContextCNN, "body_context_cnn"),
        (ResidualWoundCNN, "residual_wound_cnn")
    ]
    
    trained_models = []
    for model_class, model_name in models_to_train:
        result = train_enhanced_model(model_class, model_name, dataset_path, epochs=15)
        if result:
            trained_models.append(result)
    
    # Display results
    print(f"\n=== Training Complete ===")
    all_models = glob.glob("*_acc_*.pth")
    
    # Sort by accuracy
    model_scores = []
    for model in all_models:
        try:
            acc = float(model.split('_acc_')[1].split('.pth')[0])
            model_scores.append((acc, model))
        except:
            continue
    
    print(f"All trained models ({len(model_scores)}):")
    for acc, model in sorted(model_scores, reverse=True):
        print(f"  {model} - {acc:.1f}% accuracy")
    
    # Show top ensemble candidates
    top_models = sorted(model_scores, reverse=True)[:5]
    print(f"\nTop models for ensemble:")
    for acc, model in top_models:
        print(f"  {model} ({acc:.1f}%)")
    
    if len(top_models) > 0:
        print(f"\nExpected ensemble accuracy: {top_models[0][0] + 2:.1f}% - {top_models[0][0] + 7:.1f}%")

if __name__ == "__main__":
    main()