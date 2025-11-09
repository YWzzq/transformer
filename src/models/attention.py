"""
Multi-Head Attention mechanism implementation from scratch.

Mathematical formulation:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    
    Computes attention weights using scaled dot product of queries and keys,
    then uses these weights to compute weighted sum of values.
    
    Args:
        dropout: Dropout probability applied to attention weights
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            Q: Query tensor of shape (batch_size, n_heads, seq_len_q, d_k)
            K: Key tensor of shape (batch_size, n_heads, seq_len_k, d_k)
            V: Value tensor of shape (batch_size, n_heads, seq_len_v, d_v)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
                  or (batch_size, 1, 1, seq_len_k) for padding mask
                  Values should be 0 for positions to mask, 1 otherwise
        
        Returns:
            output: Attention output of shape (batch_size, n_heads, seq_len_q, d_v)
            attention_weights: Attention weights of shape (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        # d_k: dimension of key/query vectors
        d_k = Q.size(-1)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        # Shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        # Shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum of values
        # Shape: (batch_size, n_heads, seq_len_q, d_v)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Splits the embedding dimension into multiple heads, applies scaled dot-product
    attention in parallel, then concatenates and projects the results.
    
    Args:
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (n_heads, d_k).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Reshaped tensor of shape (batch_size, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        # Reshape to (batch_size, seq_len, n_heads, d_k)
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        # Transpose to (batch_size, n_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_heads.
        
        Args:
            x: Input tensor of shape (batch_size, n_heads, seq_len, d_k)
        
        Returns:
            Reshaped tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, n_heads, seq_len, d_k = x.size()
        # Transpose to (batch_size, seq_len, n_heads, d_k)
        x = x.transpose(1, 2).contiguous()
        # Reshape to (batch_size, seq_len, d_model)
        return x.view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor
        
        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Attention weights of shape (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)
        
        # 1. Linear projections
        Q = self.W_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.W_v(value)  # (batch_size, seq_len_v, d_model)
        
        # 2. Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, n_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch_size, n_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch_size, n_heads, seq_len_v, d_k)
        
        # 3. Apply scaled dot-product attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        # attn_output: (batch_size, n_heads, seq_len_q, d_k)
        
        # 4. Concatenate heads
        attn_output = self.combine_heads(attn_output)
        # (batch_size, seq_len_q, d_model)
        
        # 5. Final linear projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights

