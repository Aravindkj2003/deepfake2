import csv
import torch
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class AudioSpectrogramDataset(Dataset):
    """
    Dataset for loading audio files and converting to Mel-spectrograms.
    Reads from manifest.csv created by augmentation script.
    """
    
    def __init__(self, data_dir, sample_rate=16000, n_mels=128, max_len=128, transform=None):
        """
        Args:
            data_dir: Path to augmented dataset (contains manifest.csv)
            sample_rate: Audio sample rate
            n_mels: Number of Mel frequency bins
            max_len: Maximum time frames (width of spectrogram)
            transform: Optional torchvision transforms
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_len = max_len
        self.transform = transform
        
        # Load manifest
        manifest_path = self.data_dir / "manifest.csv"
        self.samples = []
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = 1 if row['label'] == 'real' else 0
                self.samples.append({
                    'path': row['path'],
                    'label': label,
                    'augmentation': row['augmentation']
                })
        
        print(f"Loaded {len(self.samples)} samples from {manifest_path}")
        real_count = sum(1 for s in self.samples if s['label'] == 1)
        fake_count = len(self.samples) - real_count
        print(f"  Real: {real_count} | Fake: {fake_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def audio_to_melspec(self, audio_path):
        """Convert audio file to Mel-spectrogram."""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Compute Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            fmax=8000
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        mel_spec_db = mel_spec_db * 2.0 - 1.0
        
        # Pad or truncate to fixed width
        if mel_spec_db.shape[1] < self.max_len:
            pad_width = self.max_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1)
        else:
            mel_spec_db = mel_spec_db[:, :self.max_len]
        
        return mel_spec_db
    
    def __getitem__(self, idx):
        """Get spectrogram and label."""
        sample = self.samples[idx]
        
        # Convert audio to spectrogram
        mel_spec = self.audio_to_melspec(sample['path'])
        
        # Convert to tensor [1, n_mels, max_len]
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        label = torch.FloatTensor([sample['label']])
        
        return mel_spec, label


def get_data_loaders(data_dir, batch_size=32, val_split=0.2, num_workers=4):
    """
    Create train and validation data loaders.
    
    Args:
        data_dir: Path to augmented dataset
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        num_workers: Number of worker processes
    
    Returns:
        (train_loader, val_loader)
    """
    # Optional: Add data augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),  # Time reversal
    ])
    
    # Full dataset
    full_dataset = AudioSpectrogramDataset(data_dir, transform=train_transform)
    
    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {train_size} | Val samples: {val_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test
    data_dir = Path("./data/for2sec/aug")
    
    if data_dir.exists():
        dataset = AudioSpectrogramDataset(data_dir)
        print(f"\nDataset size: {len(dataset)}")
        
        # Test one sample
        spec, label = dataset[0]
        print(f"Spectrogram shape: {spec.shape}")
        print(f"Label: {label.item()} ({'real' if label.item() == 1 else 'fake'})")
        print(f"Value range: [{spec.min():.3f}, {spec.max():.3f}]")
    else:
        print(f"Dataset not found at {data_dir}")
        print("Run augmentation first!")
