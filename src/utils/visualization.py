"""
Visualization utilities for training curves and results.
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from typing import List, Dict
import os


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
    title: str = "Training and Validation Loss"
):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}")


def plot_learning_rate(
    learning_rates: List[float],
    save_path: str,
    title: str = "Learning Rate Schedule"
):
    """
    Plot learning rate schedule.
    
    Args:
        learning_rates: List of learning rates per step
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    steps = range(1, len(learning_rates) + 1)
    
    plt.plot(steps, learning_rates, 'g-', linewidth=2)
    
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning rate plot saved to {save_path}")


def plot_ablation_study(
    results: Dict[str, List[float]],
    save_path: str,
    metric_name: str = "Validation Loss",
    title: str = "Ablation Study Results"
):
    """
    Plot ablation study results comparing different configurations.
    
    Args:
        results: Dictionary mapping configuration names to loss values per epoch
        save_path: Path to save the plot
        metric_name: Name of the metric being plotted
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, (config_name, values) in enumerate(results.items()):
        epochs = range(1, len(values) + 1)
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        plt.plot(epochs, values, color=color, marker=marker, linestyle='-',
                label=config_name, linewidth=2, markersize=6)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ablation study plot saved to {save_path}")


def plot_comparison_bar(
    config_names: List[str],
    values: List[float],
    save_path: str,
    ylabel: str = "Final Validation Loss",
    title: str = "Model Configuration Comparison"
):
    """
    Create bar chart comparing final metrics across configurations.
    
    Args:
        config_names: List of configuration names
        values: List of corresponding metric values
        save_path: Path to save the plot
        ylabel: Y-axis label
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'orchid', 'chocolate']
    bar_colors = [colors[i % len(colors)] for i in range(len(config_names))]
    
    bars = plt.bar(range(len(config_names)), values, color=bar_colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11)
    
    plt.xticks(range(len(config_names)), config_names, rotation=15, ha='right')
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison bar chart saved to {save_path}")


def save_training_log(
    log_data: Dict,
    save_path: str
):
    """
    Save training log to text file.
    
    Args:
        log_data: Dictionary with training information
        save_path: Path to save the log
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Training Log\n")
        f.write("=" * 60 + "\n\n")
        
        for key, value in log_data.items():
            if isinstance(value, (list, tuple)):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"  {item}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Training log saved to {save_path}")

