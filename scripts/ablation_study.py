"""
Ablation study script to analyze different model configurations.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import random
import numpy as np
import argparse
from src.models.transformer import Transformer
from src.data.dataset import (
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
    save_checkpoint
)
from src.utils.visualization import plot_ablation_study, plot_comparison_bar
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


def train_configuration(
    config_name: str,
    model_kwargs: dict,
    train_loader,
    val_loader,
    src_vocab,
    tgt_vocab,
    device,
    num_epochs: int = 10
):
    """
    Train model with specific configuration.
    
    Returns:
        List of validation losses per epoch
    """
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"{'='*60}")
    
    # Build model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        pad_idx=Config.pad_idx,
        **model_kwargs
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=Config.pad_idx,
        smoothing=Config.label_smoothing
    )
    
    base_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(Config.adam_beta1, Config.adam_beta2),
        eps=Config.adam_eps
    )
    
    optimizer = NoamOptimizer(
        base_optimizer,
        d_model=model_kwargs.get('d_model', Config.d_model),
        warmup_steps=Config.warmup_steps,
        factor=1.0
    )
    
    # Training
    val_losses = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            clip_grad=Config.gradient_clip,
            log_interval=50
        )
        
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save final model
    save_checkpoint(
        model,
        optimizer,
        num_epochs,
        val_losses[-1],
        os.path.join(Config.save_dir, f'ablation_{config_name.replace(" ", "_")}.pth')
    )
    
    return val_losses


def main(args):
    set_seed(Config.seed)
    device = Config.device
    
    # Create directories
    os.makedirs(Config.save_dir, exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Load data (use subset for ablation study)
    print("Loading Multi30k data...")
    train_pairs, val_pairs = load_multi30k_data(
        data_dir=Config.data_dir,
        max_samples=args.max_samples,
        max_len=Config.max_seq_len
    )
    
    # Load vocabularies
    src_vocab = Vocabulary.load(os.path.join(Config.save_dir, 'src_vocab.pkl'))
    tgt_vocab = Vocabulary.load(os.path.join(Config.save_dir, 'tgt_vocab.pkl'))
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_pairs,
        val_pairs,
        src_vocab,
        tgt_vocab,
        batch_size=Config.batch_size,
        max_len=Config.max_seq_len
    )
    
    # Define ablation configurations
    configurations = {}
    
    # Baseline
    if args.run_baseline or args.run_all:
        configurations['Baseline'] = {
            'd_model': Config.d_model,
            'n_heads': Config.n_heads,
            'n_encoder_layers': Config.n_encoder_layers,
            'n_decoder_layers': Config.n_decoder_layers,
            'd_ff': Config.d_ff,
            'dropout': Config.dropout,
            'max_seq_len': Config.max_seq_len
        }
    
    # Ablation 1: Different number of attention heads
    if args.run_heads or args.run_all:
        for n_heads in args.heads_config:
            configurations[f'{n_heads} Heads'] = {
                'd_model': Config.d_model,
                'n_heads': n_heads,
                'n_encoder_layers': Config.n_encoder_layers,
                'n_decoder_layers': Config.n_decoder_layers,
                'd_ff': Config.d_ff,
                'dropout': Config.dropout,
                'max_seq_len': Config.max_seq_len
            }
    
    # Ablation 2: Different model depth
    if args.run_depth or args.run_all:
        configurations['2 Layers'] = {
            'd_model': Config.d_model,
            'n_heads': Config.n_heads,
            'n_encoder_layers': 2,
            'n_decoder_layers': 2,
            'd_ff': Config.d_ff,
            'dropout': Config.dropout,
            'max_seq_len': Config.max_seq_len
        }
    
    # Ablation 3: Different FFN dimension
    if args.run_ffn or args.run_all:
        configurations['FFN 512'] = {
            'd_model': Config.d_model,
            'n_heads': Config.n_heads,
            'n_encoder_layers': Config.n_encoder_layers,
            'n_decoder_layers': Config.n_decoder_layers,
            'd_ff': 512,
            'dropout': Config.dropout,
            'max_seq_len': Config.max_seq_len
        }
    
    # Ablation 4: No dropout
    if args.run_dropout or args.run_all:
        configurations['No Dropout'] = {
            'd_model': Config.d_model,
            'n_heads': Config.n_heads,
            'n_encoder_layers': Config.n_encoder_layers,
            'n_decoder_layers': Config.n_decoder_layers,
            'd_ff': Config.d_ff,
            'dropout': 0.0,
            'max_seq_len': Config.max_seq_len
        }
    
    # Ablation 5: No positional encoding (CRITICAL!)
    if args.run_no_pos or args.run_all:
        configurations['No Pos Encoding'] = {
            'd_model': Config.d_model,
            'n_heads': Config.n_heads,
            'n_encoder_layers': Config.n_encoder_layers,
            'n_decoder_layers': Config.n_decoder_layers,
            'd_ff': Config.d_ff,
            'dropout': Config.dropout,
            'max_seq_len': Config.max_seq_len,
            'use_pos_encoding': False  # Disable positional encoding
        }
    
    # Train all configurations
    results = {}
    
    for config_name, model_kwargs in configurations.items():
        val_losses = train_configuration(
            config_name,
            model_kwargs,
            train_loader,
            val_loader,
            src_vocab,
            tgt_vocab,
            device,
            num_epochs=args.epochs
        )
        results[config_name] = val_losses
    
    # Plot results
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)
    
    plot_ablation_study(
        results,
        'results/figures/ablation_study.png',
        metric_name='Validation Loss',
        title='Ablation Study: Model Configurations'
    )
    
    # Plot final comparison
    final_losses = {name: losses[-1] for name, losses in results.items()}
    plot_comparison_bar(
        list(final_losses.keys()),
        list(final_losses.values()),
        'results/figures/ablation_comparison.png',
        ylabel='Final Validation Loss',
        title='Ablation Study: Final Performance Comparison'
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Ablation Study Summary")
    print("="*60)
    for name, losses in results.items():
        print(f"{name:20s} | Final Val Loss: {losses[-1]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation study for Transformer')
    parser.add_argument('--max-samples', type=int, default=20000,
                       help='Maximum training samples for ablation study')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs for each configuration')
    
    # Select which ablations to run
    parser.add_argument('--run-all', action='store_true',
                       help='Run all ablation experiments')
    parser.add_argument('--run-baseline', action='store_true',
                       help='Run baseline configuration')
    parser.add_argument('--run-heads', action='store_true',
                       help='Run attention heads ablation')
    parser.add_argument('--heads-config', type=int, nargs='+', default=[2, 4],
                       help='Number of attention heads to test (default: 2, 4)')
    parser.add_argument('--run-depth', action='store_true',
                       help='Run model depth ablation')
    parser.add_argument('--run-ffn', action='store_true',
                       help='Run FFN dimension ablation')
    parser.add_argument('--run-dropout', action='store_true',
                       help='Run dropout ablation')
    parser.add_argument('--run-no-pos', action='store_true',
                       help='Run no positional encoding ablation (CRITICAL!)')
    
    args = parser.parse_args()
    
    # If no specific ablation selected, run all
    if not any([args.run_baseline, args.run_heads, args.run_depth, 
                args.run_ffn, args.run_dropout, args.run_no_pos]):
        args.run_all = True
    
    main(args)

