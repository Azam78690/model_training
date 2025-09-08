"""
Sign Language Model - GRU-based classifier for dynamic and static gestures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SignGRUClassifier(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=26, num_layers=2, dropout=0.2):
        """
        Sign Language GRU Classifier
        
        Args:
            input_size: Number of input features (21 landmarks × 3 coordinates = 63)
            hidden_size: Hidden state size
            num_classes: Number of output classes
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super(SignGRUClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Optional sequence lengths for packed sequences
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # GRU forward pass
        if lengths is not None:
            # Pack padded sequences
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            gru_out, h_n = self.gru(x)
            # Unpack
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        else:
            gru_out, h_n = self.gru(x)
        
        # Use the last hidden state from the last layer
        last_hidden = h_n[-1]  # (batch_size, hidden_size)
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Classify
        out = self.classifier(last_hidden)
        
        return out
    
    def predict(self, x, lengths=None):
        """
        Make prediction with softmax
        
        Args:
            x: Input tensor
            lengths: Optional sequence lengths
            
        Returns:
            Predicted class probabilities
        """
        with torch.no_grad():
            logits = self.forward(x, lengths)
            probabilities = F.softmax(logits, dim=1)
            return probabilities
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
