"""
Transformer Encoder implementation.
"""
import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .ffn import PositionwiseFeedForward
from .layers import PreNormSublayerConnection


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Consists of:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    Each sublayer has a residual connection and layer normalization.
    
    Args:
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position-wise feed-forward network
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Sublayer connections (pre-norm)
        self.sublayer1 = PreNormSublayerConnection(d_model, dropout)
        self.sublayer2 = PreNormSublayerConnection(d_model, dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional padding mask of shape (batch_size, 1, 1, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention sublayer
        # Lambda is used to pass the mask to self-attention
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask)[0])
        
        # Feed-forward sublayer
        x = self.sublayer2(x, self.ffn)
        
        return x


class Encoder(nn.Module):
    """
    Transformer Encoder: Stack of N encoder layers.
    
    Args:
        n_layers: Number of encoder layers
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout probability
    """
    
    def __init__(
        self, 
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all encoder layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional padding mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer normalization
        return self.norm(x)

