"""
Position-wise Feed-Forward Network implementation.

Mathematical formulation:
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    where the inner layer has dimension d_ff (typically 4 * d_model)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    A two-layer fully-connected network applied independently to each position.
    Uses ReLU activation and dropout.
    
    Args:
        d_model: Dimension of model embeddings
        d_ff: Dimension of feed-forward network (inner layer)
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # First linear transformation: d_model -> d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        
        # Second linear transformation: d_ff -> d_model
        self.fc2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First layer with ReLU activation
        # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        hidden = F.relu(self.fc1(x))
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Second layer
        # Shape: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        output = self.fc2(hidden)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output

