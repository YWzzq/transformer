"""
Transformer Decoder implementation.
"""
import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .ffn import PositionwiseFeedForward
from .layers import PreNormSublayerConnection


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.
    
    Consists of:
    1. Masked multi-head self-attention
    2. Multi-head cross-attention (encoder-decoder attention)
    3. Position-wise feed-forward network
    Each sublayer has a residual connection and layer normalization.
    
    Args:
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Masked multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Multi-head cross-attention (attends to encoder output)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position-wise feed-forward network
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Sublayer connections (pre-norm)
        self.sublayer1 = PreNormSublayerConnection(d_model, dropout)
        self.sublayer2 = PreNormSublayerConnection(d_model, dropout)
        self.sublayer3 = PreNormSublayerConnection(d_model, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of decoder layer.
        
        Args:
            x: Target input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Source padding mask of shape (batch_size, 1, 1, src_seq_len)
            tgt_mask: Target mask (future + padding) of shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
        
        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention sublayer (look only at past positions)
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        
        # Cross-attention sublayer (attend to encoder output)
        x = self.sublayer2(
            x, 
            lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask)[0]
        )
        
        # Feed-forward sublayer
        x = self.sublayer3(x, self.ffn)
        
        return x


class Decoder(nn.Module):
    """
    Transformer Decoder: Stack of N decoder layers.
    
    Args:
        n_layers: Number of decoder layers
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
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all decoder layers.
        
        Args:
            x: Target input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output tensor
            src_mask: Source padding mask
            tgt_mask: Target mask (future + padding)
        
        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Final layer normalization
        return self.norm(x)

