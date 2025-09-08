"""
Training script for Sign Language GRU model
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sign_model import SignGRUClassifier
from versioning.registry_manager import ModelRegistry

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, max_seq_length=16):
        """
        Sign Language Dataset
        
        Args:
            data_dir: Directory containing class folders with .npy files
            max_seq_length: Maximum sequence length for padding/truncation
        """
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
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
            
            # Find all .npy files in class directory
            npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            
            for npy_file in npy_files:
                file_path = os.path.join(class_dir, npy_file)
                self.samples.append((file_path, class_idx))
        
        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, class_idx = self.samples[idx]
        
        # Load sequence data
        sequence = np.load(file_path)  # Shape: (seq_len, 63)
        
        # Pad or truncate to max_seq_length
        if len(sequence) < self.max_seq_length:
            # Pad with zeros
            padding = np.zeros((self.max_seq_length - len(sequence), 63))
            sequence = np.vstack([sequence, padding])
        else:
            # Truncate
            sequence = sequence[:self.max_seq_length]
        
        # Convert to tensor
        sequence = torch.FloatTensor(sequence)
        class_idx = torch.LongTensor([class_idx])
        
        return sequence, class_idx.squeeze()

def train_model(data_dir, model_dir, epochs=20, batch_size=8, learning_rate=0.001, 
                max_seq_length=16, hidden_size=128, num_layers=2):
    """
    Train Sign Language model
    
    Args:
        data_dir: Directory containing training data
        model_dir: Directory to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        hidden_size: GRU hidden size
        num_layers: Number of GRU layers
    """
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = SignLanguageDataset(data_dir, max_seq_length)
    
    if len(dataset) == 0:
        print("No data found. Please collect some data first.")
        return None
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    num_classes = len(dataset.classes)
    model = SignGRUClassifier(
        input_size=63,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers
    )
    
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
        
        for sequences, labels in progress_bar:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences)
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
    model_filename = f"sign_model_{timestamp}.pt"
    model_path = os.path.join(model_dir, model_filename)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_info': model.get_model_info(),
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx,
        'max_seq_length': max_seq_length,
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_size': hidden_size,
            'num_layers': num_layers
        }
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Register model
    registry = ModelRegistry()
    version = f"v{len(registry.list_models('sign_language')) + 1}"
    
    registry.register_model(
        model_type='sign_language',
        version=version,
        model_path=model_path,
        accuracy=epoch_acc / 100,
        metadata={
            'classes': dataset.classes,
            'num_samples': len(dataset),
            'training_config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'hidden_size': hidden_size,
                'num_layers': num_layers
            }
        }
    )
    
    print(f"Model registered as version: {version}")
    print(f"Final accuracy: {epoch_acc:.2f}%")
    
    return model_path

def main():
    parser = argparse.ArgumentParser(description='Train Sign Language Model')
    parser.add_argument('--data_dir', type=str, default='datasets/sign_language',
                       help='Directory containing training data')
    parser.add_argument('--model_dir', type=str, default='models/sign_language',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=16,
                       help='Maximum sequence length')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='GRU hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GRU layers')
    
    args = parser.parse_args()
    
    # Train model
    model_path = train_model(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )
    
    if model_path:
        print(f"Training completed successfully!")
        print(f"Model saved to: {model_path}")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()
