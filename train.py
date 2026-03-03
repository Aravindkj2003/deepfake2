"""
Unified Training Script for Deepfake Audio Detection Models.

Supports:
- CNN: Simple convolutional neural network
- CNN+LSTM: Hybrid model combining CNN and LSTM
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.dataset import get_data_loaders
from models.gan_model import Discriminator
from models.cnn_lstm_model import create_cnn_lstm_model
from models.advanced_models import create_advanced_model


class ModelTrainer:
    """Trainer for deepfake detection models."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=0.001,
        checkpoint_dir="./checkpoints",
        model_name="model"
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=True)
        for spectrograms, labels in pbar:
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device).float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(spectrograms)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy calculation
            predictions = (outputs > 0.0).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(correct / total):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = correct / total
        
        return avg_loss, avg_accuracy
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for spectrograms, labels in tqdm(self.val_loader, desc="Validating", leave=True):
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device).float()
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                
                outputs = self.model(spectrograms)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                predictions = (outputs > 0.0).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = correct / total
        
        return avg_loss, avg_accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history
        }
        
        # Save latest
        latest_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f'{self.model_name}_best.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model: {best_path}")
    
    def train(self, num_epochs=20, early_stopping_patience=10):
        """Main training loop."""
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"Starting Training: {self.model_name.upper()}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print results
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Early stopping and checkpoint
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_checkpoint(epoch, is_best=True)
                patience_counter = 0
                print(f"✓ Best validation accuracy: {self.best_accuracy:.4f}")
            else:
                patience_counter += 1
                self.save_checkpoint(epoch, is_best=False)
                print(f"No improvement ({patience_counter}/{early_stopping_patience})")
                
                if patience_counter >= early_stopping_patience:
                    print(f"\n✗ Early stopping at epoch {epoch}")
                    break
        
        # Save training history
        history_path = os.path.join(
            self.checkpoint_dir, 
            f'{self.model_name}_training_history.json'
        )
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)")
        print(f"Checkpoint saved: {self.checkpoint_dir}/{self.model_name}_best.pth")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train deepfake audio detection models"
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        choices=['cnn', 'cnn_lstm', 'resnet18', 'resnet50', 'efficientnet_b0'],
        default='resnet18',
        help='Model architecture to train'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Use pretrained ImageNet weights (for advanced models)'
    )
    parser.add_argument(
        '--no-pretrained',
        dest='pretrained',
        action='store_false',
        help='Disable pretrained weights'
    )
    parser.set_defaults(pretrained=True)
    
    # Data
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/for2sec/aug',
        help='Path to augmented data directory'
    )
    
    # Training
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    
    # LSTM specific
    parser.add_argument('--lstm-hidden', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--lstm-layers', type=int, default=2, help='Number of LSTM layers')
    
    # Checkpoints
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory for saving checkpoints'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")
    
    # Create model
    if args.model == 'cnn':
        print("Creating CNN model...")
        model = Discriminator(dropout_rate=args.dropout_rate)
        model_name = 'cnn'
    elif args.model == 'cnn_lstm':
        print("Creating CNN+LSTM model...")
        model = create_cnn_lstm_model(
            lstm_hidden_size=args.lstm_hidden,
            lstm_num_layers=args.lstm_layers,
            dropout_rate=args.dropout_rate
        )
        model_name = 'cnn_lstm'
    else:
        print(f"Creating advanced model: {args.model} (pretrained={args.pretrained})...")
        model = create_advanced_model(args.model, pretrained=args.pretrained)
        model_name = args.model
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")
    
    # Train
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        model_name=model_name
    )
    
    trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience
    )


if __name__ == '__main__':
    main()
