"""
Utility layers: Layer Normalization and Residual Connections.
"""
import torch
import torch.nn as nn


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer normalization.
    
    Implements: LayerNorm(x + Sublayer(x))
    Note: This follows the "post-norm" configuration for stability.
    
    Args:
        d_model: Dimension of model embeddings
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Apply residual connection and layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            sublayer: The sublayer to apply (e.g., attention or feed-forward)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Apply sublayer, dropout, add residual, then normalize
        return self.norm(x + self.dropout(sublayer(x)))


class PreNormSublayerConnection(nn.Module):
    """
    Layer normalization followed by residual connection.
    
    Implements: x + Sublayer(LayerNorm(x))
    This is the "pre-norm" configuration, which can be more stable for deep models.
    
    Args:
        d_model: Dimension of model embeddings
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Apply layer normalization and residual connection.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            sublayer: The sublayer to apply
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Apply normalization, sublayer, dropout, then add residual
        return x + self.dropout(sublayer(self.norm(x)))

