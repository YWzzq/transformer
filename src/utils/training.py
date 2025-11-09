"""
Training utilities: optimizer, learning rate scheduler, and training loop.
"""
import math
import torch
import torch.nn as nn
from typing import Optional
import time


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization.
    
    Args:
        vocab_size: Size of vocabulary
        pad_idx: Padding token index
        smoothing: Smoothing factor (0.0 = no smoothing, 0.1 is common)
    """
    
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            logits: Model output of shape (batch_size, seq_len, vocab_size)
            target: Target indices of shape (batch_size, seq_len)
        
        Returns:
            Loss scalar
        """
        # Reshape for cross entropy
        logits = logits.contiguous().view(-1, self.vocab_size)  # (batch_size * seq_len, vocab_size)
        target = target.contiguous().view(-1)  # (batch_size * seq_len)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # Distribute smoothing
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)  # Assign confidence to true label
            true_dist[:, self.pad_idx] = 0  # Zero out padding
            
            # Mask padding positions
            mask = (target == self.pad_idx)
            true_dist = true_dist.masked_fill(mask.unsqueeze(1), 0)
        
        # Compute KL divergence loss
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        
        # Mask padding and average
        loss = loss.masked_fill(mask, 0)
        loss = loss.sum() / (~mask).sum()
        
        return loss


class NoamOptimizer:
    """
    Optimizer with Noam learning rate scheduling (from "Attention Is All You Need").
    
    Learning rate schedule:
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension
        warmup_steps: Number of warmup steps
        factor: Scaling factor for learning rate
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0
    
    def step(self):
        """Update parameters and learning rate."""
        self._step += 1
        rate = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def get_lr(self):
        """Compute current learning rate."""
        step = self._step
        if step == 0:
            step = 1
        
        lr = self.factor * (self.d_model ** (-0.5)) * min(
            step ** (-0.5),
            step * (self.warmup_steps ** (-1.5))
        )
        return lr
    
    def zero_grad(self):
        """Zero out gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """Return state dictionary."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'step': self._step,
            'rate': self._rate
        }
    
    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self._step = state_dict['step']
        self._rate = state_dict['rate']


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
    clip_grad: float = 1.0,
    log_interval: int = 100
):
    """
    Train for one epoch.
    
    Args:
        model: Transformer model
        dataloader: Training dataloader
        optimizer: Optimizer (NoamOptimizer or standard)
        criterion: Loss function
        device: Device to train on
        clip_grad: Gradient clipping threshold
        log_interval: Print loss every N batches
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        src = batch['src'].to(device)  # (batch_size, src_seq_len)
        tgt = batch['tgt'].to(device)  # (batch_size, tgt_seq_len)
        
        # Target input (remove last token) and output (remove first token)
        tgt_input = tgt[:, :-1]  # Remove EOS for input
        tgt_output = tgt[:, 1:]   # Remove BOS for target
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(src, tgt_input)  # (batch_size, tgt_seq_len-1, vocab_size)
        
        # Compute loss
        loss = criterion(logits, tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # Update parameters
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item()
        total_tokens += (tgt_output != 0).sum().item()  # Count non-padding tokens
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            lr = optimizer._rate if hasattr(optimizer, '_rate') else optimizer.param_groups[0]['lr']
            
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Time: {elapsed:.2f}s")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device
):
    """
    Evaluate model on validation set.
    
    Args:
        model: Transformer model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        Average loss for the dataset
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # Target input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            logits = model(src, tgt_input)
            
            # Compute loss
            loss = criterion(logits, tgt_output)
            
            total_loss += loss.item()
            total_tokens += (tgt_output != 0).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer,
    path: str,
    device: torch.device
):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        path: Path to checkpoint
        device: Device to load to
    
    Returns:
        Dictionary with epoch, loss, and other saved items
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    return checkpoint

