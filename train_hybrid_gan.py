"""
Hybrid GAN Training: Load pre-trained discriminator, then do adversarial training.
Best of both worlds - CNN stability + GAN theory.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from models.gan_model import Generator, Discriminator
from models.dataset import get_data_loaders


class HybridGANTrainer:
    """
    Train GAN with pre-trained discriminator.
    Discriminator already knows real/fake features → more stable adversarial training.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create generator
        self.generator = Generator(
            latent_dim=config['latent_dim'],
            channels=1
        ).to(self.device)
        
        # Load PRE-TRAINED discriminator
        print(f"\nLoading pre-trained discriminator from {config['pretrained_checkpoint']}...")
        checkpoint = torch.load(config['pretrained_checkpoint'], map_location=self.device, weights_only=False)
        
        self.discriminator = Discriminator(
            channels=1,
            dropout_rate=config['dropout_rate']
        ).to(self.device)
        
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        pretrain_acc = checkpoint['history']['best_val_acc']
        print(f"✓ Loaded discriminator with {pretrain_acc:.4f} ({pretrain_acc*100:.2f}%) pre-trained accuracy")
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers - lower learning rates for fine-tuning
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config['lr_g'],
            betas=(0.5, 0.999)
        )
        
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr_d'] * 0.1,  # Much lower LR for pre-trained model
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
        
        # Training history
        self.history = {
            'train_d_loss': [],
            'train_g_loss': [],
            'val_d_loss': [],
            'val_acc': [],
            'best_val_acc': pretrain_acc,
            'pretrain_acc': pretrain_acc,
            'epochs_no_improve': 0
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nModel Configuration:")
        print(f"  Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"  Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        print(f"  Pre-trained accuracy: {pretrain_acc:.4f}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rates: G={config['lr_g']}, D={config['lr_d']*0.1} (fine-tune)")
    
    def train_epoch(self, epoch):
        """Train for one epoch with adversarial training."""
        self.generator.train()
        self.discriminator.train()
        
        d_losses = []
        g_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for real_specs, real_labels in pbar:
            batch_size = real_specs.size(0)
            real_specs = real_specs.to(self.device)
            
            # Soft labels for stability
            real_labels_smooth = torch.ones(batch_size, 1).to(self.device) * 0.9
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # =====================
            # Train Discriminator (less frequently to avoid overpowering)
            # =====================
            if epoch % 2 == 0:  # Train D every other iteration
                self.optimizer_D.zero_grad()
                
                # Real samples
                real_preds = self.discriminator(real_specs)
                d_loss_real = self.criterion(real_preds, real_labels_smooth)
                
                # Fake samples
                z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
                fake_specs = self.generator(z).detach()
                fake_preds = self.discriminator(fake_specs)
                d_loss_fake = self.criterion(fake_preds, fake_labels)
                
                # Total discriminator loss
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.optimizer_D.step()
                
                d_losses.append(d_loss.item())
            
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.optimizer_G.step()
            
            g_losses.append(g_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{d_loss.item() if d_losses else 0:.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })
        
        avg_d_loss = sum(d_losses) / len(d_losses) if d_losses else 0
        avg_g_loss = sum(g_losses) / len(g_losses)
        
        return avg_d_loss, avg_g_loss
    
    def validate(self):
        """Validate discriminator on real detection task."""
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
        latest_path = self.checkpoint_dir / 'hybrid_gan_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'hybrid_gan_best.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best hybrid model (val_acc: {self.history['val_acc'][-1]:.4f})")
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting Hybrid GAN Training")
        print(f"(Pre-trained Discriminator + Adversarial Fine-tuning)")
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
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"  Train - D_loss: {train_d_loss:.4f} | G_loss: {train_g_loss:.4f}")
            print(f"  Val   - D_loss: {val_d_loss:.4f} | Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
            
            # Check for improvement
            is_best = val_acc > self.history['best_val_acc']
            if is_best:
                improvement = val_acc - self.history['best_val_acc']
                print(f"  ✓ Improved by {improvement:.4f} ({improvement*100:.2f}%)")
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
        history_path = self.checkpoint_dir / 'hybrid_gan_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Hybrid GAN Training Complete!")
        print(f"{'='*60}")
        print(f"Pre-trained accuracy: {self.history['pretrain_acc']:.4f} ({self.history['pretrain_acc']*100:.2f}%)")
        print(f"Final accuracy: {self.history['best_val_acc']:.4f} ({self.history['best_val_acc']*100:.2f}%)")
        improvement = self.history['best_val_acc'] - self.history['pretrain_acc']
        print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        print(f"Model saved to: {self.checkpoint_dir / 'hybrid_gan_best.pth'}")


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid GAN with pre-trained discriminator')
    parser.add_argument('--pretrained-checkpoint', type=str, required=True,
                        help='Path to pre-trained discriminator checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to augmented dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr-g', type=float, default=0.0002, help='Generator learning rate')
    parser.add_argument('--lr-d', type=float, default=0.001, help='Discriminator base learning rate (will be reduced)')
    parser.add_argument('--latent-dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--early-stopping-patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Check if pre-trained checkpoint exists
    if not Path(args.pretrained_checkpoint).exists():
        print(f"Error: Pre-trained checkpoint not found at {args.pretrained_checkpoint}")
        print("Please train CNN discriminator first using train_cnn.py")
        return
    
    config = vars(args)
    
    # Initialize trainer and start training
    trainer = HybridGANTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
