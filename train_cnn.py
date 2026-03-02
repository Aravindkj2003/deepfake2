"""
Train Discriminator directly as CNN classifier (supervised learning).
Much faster and more stable than full GAN training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from models.gan_model import Discriminator
from models.dataset import get_data_loaders


class CNNTrainer:
    """
    Train discriminator as a supervised binary classifier.
    Faster and more stable than GAN training.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create discriminator model
        self.model = Discriminator(
            channels=1,
            dropout_rate=config['dropout_rate']
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Data loaders
        self.train_loader, self.val_loader = get_data_loaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            val_split=config['val_split'],
            num_workers=config['num_workers']
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'epochs_no_improve': 0
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nModel Configuration:")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['lr']}")
        print(f"  Dropout rate: {config['dropout_rate']}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for specs, labels in pbar:
            specs = specs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(specs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            acc = correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate model."""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for specs, labels in self.val_loader:
                specs = specs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(specs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'discriminator_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'cnn_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'cnn_best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_acc: {self.history['val_acc'][-1]:.4f})")
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting CNN Discriminator Training")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            
            # Check for improvement
            is_best = val_acc > self.history['best_val_acc']
            if is_best:
                self.history['best_val_acc'] = val_acc
                self.history['epochs_no_improve'] = 0
            else:
                self.history['epochs_no_improve'] += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.history['epochs_no_improve'] >= self.config['early_stopping_patience']:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                print(f"  No improvement for {self.config['early_stopping_patience']} epochs")
                break
        
        # Save final history
        history_path = self.checkpoint_dir / 'cnn_training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Best validation accuracy: {self.history['best_val_acc']:.4f} ({self.history['best_val_acc']*100:.2f}%)")
        print(f"Model saved to: {self.checkpoint_dir / 'cnn_best_model.pth'}")


def main():
    parser = argparse.ArgumentParser(description='Train CNN discriminator for audio deepfake detection')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to augmented dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--early-stopping-patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    config = vars(args)
    
    # Train
    trainer = CNNTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
