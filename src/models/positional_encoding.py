"""
Positional Encoding implementation.

Mathematical formulation:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
where pos is the position and i is the dimension index.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    Injects position information into token embeddings using sine and cosine
    functions of different frequencies.
    
    Args:
        d_model: Dimension of model embeddings
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        
        # Position indices: shape (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Division term: 10000^(2i/d_model) for i in [0, d_model/2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices (2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: shape (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding to input
        # self.pe[:, :x.size(1)] selects positional encodings up to current sequence length
        x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)

