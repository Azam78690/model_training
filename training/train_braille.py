"""
Training script for Braille CNN model
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.braille_model import BrailleCNNClassifier
from versioning.registry_manager import ModelRegistry

class BrailleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Braille Dataset
        
        Args:
            data_dir: Directory containing class folders with image files
            transform: Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load data from directory structure"""
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist")
            return
        
        # Get class directories
        class_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        class_dirs.sort()
        
        # Create class mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_dirs)}
        self.classes = class_dirs
        
        # Load samples
        for class_name in class_dirs:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Find all image files in class directory
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                image_files.extend([f for f in os.listdir(class_dir) if f.lower().endswith(ext)])
            
            for image_file in image_files:
                file_path = os.path.join(class_dir, image_file)
                self.samples.append((file_path, class_idx))
        
        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, class_idx = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(file_path).convert('L')  # Convert to grayscale
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            # Return a black image as fallback
            image = Image.new('L', (64, 64), 0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        class_idx = torch.LongTensor([class_idx])
        
        return image, class_idx.squeeze()

def train_model(data_dir, model_dir, epochs=20, batch_size=16, learning_rate=0.001):
    """
    Train Braille model
    
    Args:
        data_dir: Directory containing training data
        model_dir: Directory to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    # Load dataset
    print("Loading dataset...")
    dataset = BrailleDataset(data_dir, transform=transform)
    
    if len(dataset) == 0:
        print("No data found. Please collect some data first.")
        return None
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    num_classes = len(dataset.classes)
    model = BrailleCNNClassifier(num_classes=num_classes)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"Training model with {num_classes} classes...")
    print(f"Model info: {model.get_model_info()}")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in progress_bar:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"braille_model_{timestamp}.pt"
    model_path = os.path.join(model_dir, model_filename)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_info': model.get_model_info(),
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx,
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Register model
    registry = ModelRegistry()
    version = f"v{len(registry.list_models('braille')) + 1}"
    
    registry.register_model(
        model_type='braille',
        version=version,
        model_path=model_path,
        accuracy=epoch_acc / 100,
        metadata={
            'classes': dataset.classes,
            'num_samples': len(dataset),
            'training_config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        }
    )
    
    print(f"Model registered as version: {version}")
    print(f"Final accuracy: {epoch_acc:.2f}%")
    
    return model_path

def main():
    parser = argparse.ArgumentParser(description='Train Braille Model')
    parser.add_argument('--data_dir', type=str, default='datasets/braille',
                       help='Directory containing training data')
    parser.add_argument('--model_dir', type=str, default='models/braille',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Train model
    model_path = train_model(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    if model_path:
        print(f"Training completed successfully!")
        print(f"Model saved to: {model_path}")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()
