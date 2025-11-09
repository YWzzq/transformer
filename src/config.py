"""Configuration for Transformer training."""
import torch

class Config:
    """Hyperparameters and settings for Transformer model."""
    
    # Model Architecture
    d_model = 384              # Embedding dimension (medium capacity)
    n_heads = 8                # Number of attention heads
    d_ff = 1536                # Feed-forward dimension (medium capacity)
    n_encoder_layers = 4       # Number of encoder layers
    n_decoder_layers = 4       # Number of decoder layers
    dropout = 0.25             # Dropout rate (balanced regularization)
    max_seq_len = 128          # Maximum sequence length
    
    # Vocabulary
    vocab_size = 32000         # Vocabulary size (will be determined from data)
    pad_idx = 0                # Padding token index
    bos_idx = 1                # Begin-of-sequence token index
    eos_idx = 2                # End-of-sequence token index
    
    # Training
    batch_size = 40            # Balanced for medium model
    num_epochs = 100           # Early stopping will control actual epochs
    learning_rate = 5e-3       # Not used (NoamOptimizer controls LR)
    warmup_steps = 1500        # Balanced for medium model (~2 epochs)
    gradient_clip = 1.0
    label_smoothing = 0.15     # Increased for regularization
    
    # Early Stopping
    early_stopping_patience = 5  # Stop if no improvement for 5 epochs
    min_delta = 0.001            # Minimum change to qualify as improvement
    
    # Optimizer
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    adam_eps = 1e-9
    
    # Paths
    data_dir = "data/datasets"  # Multi30K dataset
    save_dir = "results/checkpoints"
    log_dir = "results/logs"
    
    # Device
    #使用gpu1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use cuda (defaults to cuda:0)
    
    # Reproducibility
    seed = 42
    
    # Logging
    log_interval = 100         # Print loss every N batches
    save_interval = 1          # Save checkpoint every N epochs
    
    def __repr__(self):
        attrs = {k: v for k, v in self.__class__.__dict__.items() 
                 if not k.startswith('_') and not callable(v)}
        return '\n'.join([f"{k}: {v}" for k, v in attrs.items()])

