import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network that creates fake audio spectrograms from random noise.
    Uses transposed convolutions to upsample from latent vector.
    """
    
    def __init__(self, latent_dim=100, channels=1):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.channels = channels
        
        # Starting feature map size: 8x8
        self.init_size = 8
        self.fc = nn.Linear(latent_dim, 512 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, channels, 4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, z):
        """
        Args:
            z: Random noise tensor [batch_size, latent_dim]
        Returns:
            Generated spectrogram [batch_size, channels, 128, 128]
        """
        x = self.fc(z)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator network that classifies spectrograms as real or fake.
    This is what we'll use for final predictions in the web app.
    Includes dropout for regularization to prevent overfitting.
    """
    
    def __init__(self, channels=1, dropout_rate=0.3):
        super(Discriminator, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            # Input: 128x128 -> 64x64
            nn.Conv2d(channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # 16x16 -> 8x8
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
        )
        
        # Calculate flattened size: 512 * 8 * 8
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: Spectrogram tensor [batch_size, channels, 128, 128]
        Returns:
            Probability of being real [batch_size, 1]
        """
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


def initialize_weights(model):
    """Initialize model weights using Xavier initialization."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def create_gan(latent_dim=100, channels=1, dropout_rate=0.3, device='cuda'):
    """
    Factory function to create Generator and Discriminator.
    
    Args:
        latent_dim: Size of random noise vector
        channels: Number of channels in spectrogram (1 for grayscale)
        dropout_rate: Dropout probability for regularization
        device: 'cuda' or 'cpu'
    
    Returns:
        (generator, discriminator) tuple ready for training
    """
    generator = Generator(latent_dim=latent_dim, channels=channels).to(device)
    discriminator = Discriminator(channels=channels, dropout_rate=dropout_rate).to(device)
    
    initialize_weights(generator)
    initialize_weights(discriminator)
    
    return generator, discriminator


if __name__ == "__main__":
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    G, D = create_gan(device=device)
    
    # Test forward pass
    z = torch.randn(4, 100).to(device)
    fake_spec = G(z)
    print(f"Generator output shape: {fake_spec.shape}")
    
    real_spec = torch.randn(4, 1, 128, 128).to(device)
    pred = D(real_spec)
    print(f"Discriminator output shape: {pred.shape}")
    print(f"Discriminator predictions: {pred.squeeze()}")
    
    # Count parameters
    g_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"\nGenerator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
