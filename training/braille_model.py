"""
Braille Model - Lightweight CNN for image-based braille recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BrailleCNNClassifier(nn.Module):
    def __init__(self, num_classes=26):
        """
        Braille CNN Classifier
        
        Args:
            num_classes: Number of output classes
        """
        super(BrailleCNNClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 16x16 -> 8x8
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, 64, 64)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x
    
    def predict(self, x):
        """
        Make prediction with softmax
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            return probabilities
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_shape': (1, 64, 64)
        }
