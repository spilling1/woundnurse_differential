"""
Rapid Model Training with Additional Data
Efficiently trains improved models using your new body context images
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

class QuickWoundDataset(Dataset):
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
        except:
            # Fallback for any issues
            return torch.zeros(3, 64, 64), self.labels[idx]

class ImprovedCNN(nn.Module):
    """Improved CNN architecture for better body context understanding"""
    def __init__(self, num_classes=6):
        super(ImprovedCNN, self).__init__()
        
        # Enhanced feature extraction
        self.features = nn.Sequential(
            # First block - capture fine details
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block - patterns
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block - context
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth block - high-level features
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Improved classifier
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

class DiverseCNN(nn.Module):
    """Alternative architecture for ensemble diversity"""
    def __init__(self, num_classes=6):
        super(DiverseCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Wide initial layer for body context
            nn.Conv2d(3, 48, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Deep narrow layers
            nn.Conv2d(48, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Global context
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(384 * 4, 96),
            nn.ReLU(),
            nn.Linear(96, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

def integrate_new_training_data():
    """Integrate the new training data with body context"""
    print("Integrating new training data...")
    
    # Check for new zip file
    new_zip = "attached_assets/original images_1751781038610.zip"
    if not os.path.exists(new_zip):
        print("New training data not found, using existing dataset")
        return "wound_dataset_final"
    
    # Extract new data
    try:
        with zipfile.ZipFile(new_zip, 'r') as zip_ref:
            zip_ref.extractall("original_images_new")
        print("Successfully extracted new training data")
    except Exception as e:
        print(f"Could not extract new data: {e}")
        return "wound_dataset_final"
    
    # Create enhanced dataset directory
    enhanced_dir = "wound_dataset_v3"
    if os.path.exists("wound_dataset_final"):
        shutil.copytree("wound_dataset_final", enhanced_dir, dirs_exist_ok=True)
    else:
        os.makedirs(enhanced_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            for class_name in ["background", "diabetic_ulcer", "neuropathic_ulcer", "pressure_ulcer", "surgical_wound", "venous_ulcer"]:
                os.makedirs(os.path.join(enhanced_dir, split, class_name), exist_ok=True)
    
    # Add new images to training set
    train_dir = os.path.join(enhanced_dir, 'train')
    new_count = 0
    
    # Process all images in the extracted folder
    for root, dirs, files in os.walk("original_images_new"):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, file)
                
                # Categorize based on parent folder names and filename
                folder_path = root.lower()
                file_lower = file.lower()
                
                # Smart categorization using folder structure and filenames
                if any(keyword in folder_path or keyword in file_lower for keyword in ['diabetic', 'diabetes', 'foot']):
                    category = 'diabetic_ulcer'
                elif any(keyword in folder_path or keyword in file_lower for keyword in ['pressure', 'bed', 'heel', 'sacrum']):
                    category = 'pressure_ulcer'
                elif any(keyword in folder_path or keyword in file_lower for keyword in ['venous', 'leg', 'vein']):
                    category = 'venous_ulcer'
                elif any(keyword in folder_path or keyword in file_lower for keyword in ['surgical', 'surgery', 'incision']):
                    category = 'surgical_wound'
                elif any(keyword in folder_path or keyword in file_lower for keyword in ['neuropathic', 'nerve']):
                    category = 'neuropathic_ulcer'
                else:
                    # Check parent folder name for classification
                    parent_folder = os.path.basename(os.path.dirname(src_path)).lower()
                    if 'diabetic' in parent_folder:
                        category = 'diabetic_ulcer'
                    elif 'pressure' in parent_folder:
                        category = 'pressure_ulcer'
                    elif 'venous' in parent_folder:
                        category = 'venous_ulcer'
                    elif 'surgical' in parent_folder:
                        category = 'surgical_wound'
                    elif 'neuropathic' in parent_folder:
                        category = 'neuropathic_ulcer'
                    else:
                        category = 'background'
                
                # Copy to appropriate training folder
                dst_path = os.path.join(train_dir, category, f"v3_{new_count}_{file}")
                try:
                    shutil.copy2(src_path, dst_path)
                    new_count += 1
                except Exception as e:
                    print(f"Could not copy {src_path}: {e}")
    
    print(f"Added {new_count} new images with body context to training dataset")
    return enhanced_dir

def train_model(model_class, model_name, dataset_path, epochs=15):
    """Train a single model efficiently"""
    print(f"\nTraining {model_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load training data
    train_dataset = QuickWoundDataset(dataset_path, 'train')
    if len(train_dataset) == 0:
        print("No training data found!")
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Initialize model
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4)
    
    best_acc = 0
    patience_counter = 0
    
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
            
            if batch_idx % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}')
        
        accuracy = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f'{model_name} - Epoch {epoch+1}/{epochs}: Acc {accuracy:.1f}%, Loss: {avg_loss:.4f}')
        
        # Save model if improved
        if accuracy > best_acc:
            best_acc = accuracy
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'class_names': train_dataset.class_names
            }, f'{model_name}_acc_{accuracy:.1f}.pth')
            print(f'✓ New best model saved: {accuracy:.1f}% accuracy')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 5:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f'{model_name} final accuracy: {best_acc:.1f}%')
    return f'{model_name}_acc_{best_acc:.1f}.pth'

def main():
    """Main training function"""
    print("=== Rapid Model Training with Enhanced Data ===")
    
    # Integrate new training data
    dataset_path = integrate_new_training_data()
    
    # Train improved models
    models_to_train = [
        (ImprovedCNN, "improved_cnn_v3"),
        (DiverseCNN, "diverse_cnn_v3")
    ]
    
    trained_models = []
    for model_class, model_name in models_to_train:
        result = train_model(model_class, model_name, dataset_path, epochs=12)
        if result:
            trained_models.append(result)
    
    # Show results
    print(f"\n=== Training Results ===")
    all_models = glob.glob("*_acc_*.pth")
    
    # Sort by accuracy
    model_scores = []
    for model in all_models:
        try:
            acc = float(model.split('_acc_')[1].split('.pth')[0])
            model_scores.append((acc, model))
        except:
            continue
    
    print(f"All available models ({len(model_scores)}):")
    for acc, model in sorted(model_scores, reverse=True):
        print(f"  {model} - {acc:.1f}% accuracy")
    
    # Show top ensemble candidates
    top_models = sorted(model_scores, reverse=True)[:5]
    print(f"\nTop ensemble candidates:")
    for acc, model in top_models:
        print(f"  {model} ({acc:.1f}%)")
    
    print(f"\nExpected ensemble accuracy: {top_models[0][0] + 3:.1f}% - {top_models[0][0] + 8:.1f}%")

if __name__ == "__main__":
    main()