"""
Complete Transformer model for sequence-to-sequence tasks.
"""
import torch
import torch.nn as nn
from typing import Optional
from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    Full Transformer model for sequence-to-sequence tasks (e.g., machine translation).
    
    Architecture:
    1. Source embedding + positional encoding → Encoder
    2. Target embedding + positional encoding → Decoder (with encoder output)
    3. Linear projection to vocabulary
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads
        n_encoder_layers: Number of encoder layers
        n_decoder_layers: Number of decoder layers
        d_ff: Dimension of feed-forward network
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        pad_idx: Padding token index
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
        use_pos_encoding: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.use_pos_encoding = use_pos_encoding
        
        # Source and target embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding (shared between encoder and decoder)
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        else:
            self.pos_encoding = nn.Dropout(dropout)  # Only dropout, no positional info
        
        # Encoder and decoder
        self.encoder = Encoder(n_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_decoder_layers, d_model, n_heads, d_ff, dropout)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for source sequence.
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
        
        Returns:
            Mask tensor of shape (batch_size, 1, 1, src_seq_len)
        """
        # Create mask where padding tokens are 0, others are 1
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create combined padding and future mask for target sequence.
        
        Args:
            tgt: Target tensor of shape (batch_size, tgt_seq_len)
        
        Returns:
            Mask tensor of shape (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        batch_size, tgt_len = tgt.size()
        
        # Padding mask: shape (batch_size, 1, 1, tgt_len)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Future mask (causal mask): shape (1, 1, tgt_len, tgt_len)
        # Lower triangular matrix (including diagonal)
        tgt_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device)
        ).bool().unsqueeze(0).unsqueeze(0)
        
        # Combine both masks: shape (batch_size, 1, tgt_len, tgt_len)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        return tgt_mask
    
    def encode(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            src_mask: Optional source mask
        
        Returns:
            Encoder output of shape (batch_size, src_seq_len, d_model)
        """
        # Embedding + scaling + positional encoding
        src_emb = self.src_embedding(src) * (self.d_model ** 0.5)
        src_emb = self.pos_encoding(src_emb)
        
        # Encode
        encoder_output = self.encoder(src_emb, src_mask)
        
        return encoder_output
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence given encoder output.
        
        Args:
            tgt: Target tensor of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output tensor
            src_mask: Optional source mask
            tgt_mask: Optional target mask
        
        Returns:
            Decoder output of shape (batch_size, tgt_seq_len, d_model)
        """
        # Embedding + scaling + positional encoding
        tgt_emb = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Decode
        decoder_output = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
        
        return decoder_output
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer.
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            tgt: Target tensor of shape (batch_size, tgt_seq_len)
            src_mask: Optional source mask
            tgt_mask: Optional target mask
        
        Returns:
            Logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        # Decode target
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        bos_idx: int,
        eos_idx: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate target sequence using greedy decoding.
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            max_len: Maximum generation length
            bos_idx: Begin-of-sequence token index
            eos_idx: End-of-sequence token index
            temperature: Sampling temperature (1.0 = greedy)
        
        Returns:
            Generated sequence of shape (batch_size, gen_len)
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        src_mask = self.make_src_mask(src)
        encoder_output = self.encode(src, src_mask)
        
        # Initialize target with BOS token
        tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        
        # Generate tokens one by one
        for _ in range(max_len - 1):
            # Create target mask
            tgt_mask = self.make_tgt_mask(tgt)
            
            # Decode
            decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
            
            # Get logits for last position
            logits = self.output_projection(decoder_output[:, -1, :])  # (batch_size, tgt_vocab_size)
            
            # Sample next token (greedy if temperature=1.0)
            if temperature == 1.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if all sequences have generated EOS
            if (next_token == eos_idx).all():
                break
        
        return tgt

