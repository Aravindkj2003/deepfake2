import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from models.gan_model import create_gan
from models.dataset import get_data_loaders


class GANTrainer:
    """
    Trainer for audio deepfake detection GAN.
    Implements proper adversarial training with anti-overfitting measures.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create models
        self.generator, self.discriminator = create_gan(
            latent_dim=config['latent_dim'],
            channels=1,
            dropout_rate=config['dropout_rate'],
            device=self.device
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers with weight decay for regularization
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config['lr_g'],
            betas=(0.5, 0.999),
            weight_decay=config['weight_decay']
        )
        
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr_d'],
            betas=(0.5, 0.999),
            weight_decay=config['weight_decay']
        )
        
        # Data loaders
        self.train_loader, self.val_loader = get_data_loaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            val_split=config['val_split'],
            num_workers=config['num_workers']
        )
        
        # Training tracking
        self.history = {
            'train_d_loss': [],
            'train_g_loss': [],
            'val_d_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'epochs_no_improve': 0
        }
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nModel Configuration:")
        print(f"  Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"  Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rates: G={config['lr_g']}, D={config['lr_d']}")
        print(f"  Dropout rate: {config['dropout_rate']}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        d_losses = []
        g_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for real_specs, real_labels in pbar:
            batch_size = real_specs.size(0)
            real_specs = real_specs.to(self.device)
            real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9  # Label smoothing
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # =====================
            # Train Discriminator
            # =====================
            self.optimizer_D.zero_grad()
            
            # Real samples
            real_preds = self.discriminator(real_specs)
            d_loss_real = self.criterion(real_preds, real_labels)
            
            # Fake samples
            z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
            fake_specs = self.generator(z).detach()
            fake_preds = self.discriminator(fake_specs)
            d_loss_fake = self.criterion(fake_preds, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            self.optimizer_D.step()
            
            # ==================
            # Train Generator
            # ==================
            self.optimizer_G.zero_grad()
            
            # Generate fake samples and try to fool discriminator
            z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
            fake_specs = self.generator(z)
            fake_preds = self.discriminator(fake_specs)
            
            # Generator wants discriminator to think fakes are real
            g_loss = self.criterion(fake_preds, torch.ones(batch_size, 1).to(self.device))
            g_loss.backward()
            self.optimizer_G.step()
            
            # Track losses
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })
        
        return sum(d_losses) / len(d_losses), sum(g_losses) / len(g_losses)
    
    def validate(self):
        """Validate discriminator on validation set."""
        self.discriminator.eval()
        
        val_losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for specs, labels in self.val_loader:
                specs = specs.to(self.device)
                labels = labels.to(self.device)
                
                preds = self.discriminator(specs)
                loss = self.criterion(preds, labels)
                val_losses.append(loss.item())
                
                # Calculate accuracy
                predicted = (preds > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_loss = sum(val_losses) / len(val_losses)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_acc: {self.history['val_acc'][-1]:.4f})")
    
    def train(self):
        """Main training loop with early stopping."""
        print(f"\n{'='*60}")
        print(f"Starting GAN Training")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config['epochs']):
            # Train
            train_d_loss, train_g_loss = self.train_epoch(epoch)
            
            # Validate
            val_d_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_d_loss'].append(train_d_loss)
            self.history['train_g_loss'].append(train_g_loss)
            self.history['val_d_loss'].append(val_d_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"  Train - D_loss: {train_d_loss:.4f} | G_loss: {train_g_loss:.4f}")
            print(f"  Val   - D_loss: {val_d_loss:.4f} | Accuracy: {val_acc:.4f}")
            
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
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Best validation accuracy: {self.history['best_val_acc']:.4f}")
        print(f"Model saved to: {self.checkpoint_dir / 'best_model.pth'}")


def main():
    parser = argparse.ArgumentParser(description='Train GAN for audio deepfake detection')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to augmented dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr-g', type=float, default=0.0002, help='Generator learning rate')
    parser.add_argument('--lr-d', type=float, default=0.0002, help='Discriminator learning rate')
    parser.add_argument('--latent-dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Create config dict
    config = vars(args)
    
    # Initialize trainer and start training
    trainer = GANTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
