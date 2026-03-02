"""
Visualize training history and loss curves.
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def plot_training_history(history_path, output_path=None):
    """
    Plot training curves from history JSON file.
    
    Args:
        history_path: Path to training_history.json
        output_path: Optional path to save plot (if None, displays interactively)
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = list(range(1, len(history['train_d_loss']) + 1))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GAN Training History', fontsize=16, fontweight='bold')
    
    # Plot 1: Discriminator Loss
    axes[0, 0].plot(epochs, history['train_d_loss'], label='Train D Loss', color='blue', linewidth=2)
    axes[0, 0].plot(epochs, history['val_d_loss'], label='Val D Loss', color='orange', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Discriminator Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Generator Loss
    axes[0, 1].plot(epochs, history['train_g_loss'], label='Train G Loss', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Generator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy
    axes[1, 0].plot(epochs, history['val_acc'], label='Val Accuracy', color='red', linewidth=2)
    axes[1, 0].axhline(y=history['best_val_acc'], color='green', linestyle='--', label=f'Best: {history["best_val_acc"]:.4f}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 4: Combined Losses
    axes[1, 1].plot(epochs, history['train_d_loss'], label='D Loss', color='blue', linewidth=2, alpha=0.7)
    axes[1, 1].plot(epochs, history['train_g_loss'], label='G Loss', color='green', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Combined Training Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total epochs: {len(epochs)}")
    print(f"Best validation accuracy: {history['best_val_acc']:.4f}")
    print(f"Final train D loss: {history['train_d_loss'][-1]:.4f}")
    print(f"Final train G loss: {history['train_g_loss'][-1]:.4f}")
    print(f"Final val D loss: {history['val_d_loss'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize training history')
    parser.add_argument('--history', type=str, default='checkpoints/training_history.json',
                        help='Path to training_history.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for plot (if not specified, shows interactively)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.history).exists():
        print(f"Error: History file not found at {args.history}")
        print("Please train the model first.")
        return
    
    # Plot
    plot_training_history(args.history, args.output)


if __name__ == "__main__":
    main()
