"""
Main training script for Transformer model on IWSLT2017 EN-DE translation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import random
import numpy as np
import argparse
from src.models.transformer import Transformer
from src.data.dataset import (
    load_iwslt_data,
    load_multi30k_data,
    build_vocabularies, 
    create_dataloaders,
    Vocabulary
)
from src.utils.training import (
    LabelSmoothingLoss,
    NoamOptimizer,
    train_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint
)
from src.utils.visualization import plot_training_curves, save_training_log
from src.config import Config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    # Set random seed
    set_seed(Config.seed)
    
    # Device
    device = Config.device
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(Config.save_dir, exist_ok=True)
    os.makedirs(Config.log_dir, exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    if args.dataset == 'multi30k':
        train_pairs, val_pairs = load_multi30k_data(
            Config.data_dir,
            max_samples=args.max_samples,
            max_len=Config.max_seq_len
        )
    else:
        train_pairs, val_pairs = load_iwslt_data(
            Config.data_dir,
            max_samples=args.max_samples,
            max_len=Config.max_seq_len
        )
    
    # Build or load vocabularies
    vocab_dir = Config.save_dir
    src_vocab_path = os.path.join(vocab_dir, 'src_vocab.pkl')
    tgt_vocab_path = os.path.join(vocab_dir, 'tgt_vocab.pkl')
    
    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path) and not args.rebuild_vocab:
        print("Loading existing vocabularies...")
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
        print(f"Source vocabulary size: {len(src_vocab)}")
        print(f"Target vocabulary size: {len(tgt_vocab)}")
    else:
        src_vocab, tgt_vocab = build_vocabularies(
            train_pairs,
            max_vocab_size=Config.vocab_size,
            min_freq=2,
            save_dir=vocab_dir
        )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_pairs,
        val_pairs,
        src_vocab,
        tgt_vocab,
        batch_size=Config.batch_size,
        max_len=Config.max_seq_len
    )
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Build model
    print("\n" + "="*60)
    print("Building Model")
    print("="*60)
    
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=Config.d_model,
        n_heads=Config.n_heads,
        n_encoder_layers=Config.n_encoder_layers,
        n_decoder_layers=Config.n_decoder_layers,
        d_ff=Config.d_ff,
        max_seq_len=Config.max_seq_len,
        dropout=Config.dropout,
        pad_idx=Config.pad_idx
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=Config.pad_idx,
        smoothing=Config.label_smoothing
    )
    
    base_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # Will be controlled by NoamOptimizer
        betas=(Config.adam_beta1, Config.adam_beta2),
        eps=Config.adam_eps
    )
    
    optimizer = NoamOptimizer(
        base_optimizer,
        d_model=Config.d_model,
        warmup_steps=Config.warmup_steps,
        factor=1.0
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0  # Early stopping counter
    
    start_epoch = 1
    if args.resume:
        checkpoint_path = os.path.join(Config.save_dir, 'checkpoint_last.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(model, optimizer, checkpoint_path, device)
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {checkpoint['epoch']}")
    
    for epoch in range(start_epoch, Config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{Config.num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            clip_grad=Config.gradient_clip,
            log_interval=Config.log_interval
        )
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % Config.save_interval == 0:
            config_dict = {k: v for k, v in vars(Config).items() 
                          if not k.startswith('_') and not callable(v) and k != 'device'}
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                os.path.join(Config.save_dir, f'checkpoint_epoch_{epoch}.pth'),
                train_losses=train_losses,
                val_losses=val_losses,
                best_val_loss=best_val_loss,
                config=config_dict
            )
        
        # Save best model and check for improvement
        if val_loss < best_val_loss - Config.min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset counter
            config_dict = {k: v for k, v in vars(Config).items() 
                          if not k.startswith('_') and not callable(v) and k != 'device'}
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                os.path.join(Config.save_dir, 'checkpoint_best.pth'),
                train_losses=train_losses,
                val_losses=val_losses,
                best_val_loss=best_val_loss,
                config=config_dict
            )
            print(f"  → Best model saved (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  → No improvement for {epochs_no_improve} epoch(s)")
        
        # Early stopping check
        if epochs_no_improve >= Config.early_stopping_patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"{'='*60}")
            break
        
        # Save last checkpoint
        config_dict = {k: v for k, v in vars(Config).items() 
                      if not k.startswith('_') and not callable(v) and k != 'device'}
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            os.path.join(Config.save_dir, 'checkpoint_last.pth'),
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val_loss,
            config=config_dict
        )
    
    # Plot training curves
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)
    
    plot_training_curves(
        train_losses,
        val_losses,
        'results/figures/training_curves.png'
    )
    
    # Save training log
    log_data = {
        'Model': 'Transformer (Encoder-Decoder)',
        'Dataset': 'IWSLT2017 EN-DE',
        'Training samples': len(train_pairs),
        'Validation samples': len(val_pairs),
        'Source vocab size': len(src_vocab),
        'Target vocab size': len(tgt_vocab),
        'Model parameters': f"{count_parameters(model):,}",
        'Configuration': '',
        f'  d_model': Config.d_model,
        f'  n_heads': Config.n_heads,
        f'  n_encoder_layers': Config.n_encoder_layers,
        f'  n_decoder_layers': Config.n_decoder_layers,
        f'  d_ff': Config.d_ff,
        f'  dropout': Config.dropout,
        f'  batch_size': Config.batch_size,
        f'  learning_rate': Config.learning_rate,
        f'  warmup_steps': Config.warmup_steps,
        f'  num_epochs': Config.num_epochs,
        'Best validation loss': f"{best_val_loss:.4f}",
        'Final training loss': f"{train_losses[-1]:.4f}",
        'Final validation loss': f"{val_losses[-1]:.4f}",
    }
    
    save_training_log(log_data, 'results/logs/training_log.txt')
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer for EN-DE translation')
    parser.add_argument('--dataset', type=str, default='iwslt',
                       choices=['iwslt', 'multi30k'],
                       help='Dataset to use (iwslt or multi30k)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of training samples (for quick testing)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--rebuild-vocab', action='store_true',
                       help='Rebuild vocabulary even if it exists')
    
    args = parser.parse_args()
    main(args)

