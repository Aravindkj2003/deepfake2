"""
Quick system check before training.
Verifies dependencies, GPU availability, and data setup.
"""
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  Python {version.major}.{version.minor}.{version.micro} ✓")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\n✓ Checking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'librosa': 'Librosa',
        'numpy': 'NumPy',
        'flask': 'Flask',
        'tqdm': 'tqdm',
        'soundfile': 'SoundFile'
    }
    
    all_installed = True
    
    for package, name in required.items():
        try:
            __import__(package)
            print(f"  {name} ✓")
        except ImportError:
            print(f"  {name} ✗ (not installed)")
            all_installed = False
    
    return all_installed


def check_gpu():
    """Check GPU availability."""
    print("\n✓ Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  GPU available: {device_name} ✓")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print(f"  No GPU detected (will use CPU)")
            print(f"  ⚠ Training will be slower on CPU")
            return False
    except ImportError:
        print(f"  Cannot check (PyTorch not installed)")
        return False


def check_project_structure():
    """Check if project structure is correct."""
    print("\n✓ Checking project structure...")
    
    required_files = [
        'augment_dataset.py',
        'train.py',
        'app.py',
        'evaluate.py',
        'requirements.txt',
        'models/gan_model.py',
        'models/dataset.py',
        'templates/index.html'
    ]
    
    all_exist = True
    
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  {file} ✓")
        else:
            print(f"  {file} ✗ (missing)")
            all_exist = False
    
    return all_exist


def check_dataset():
    """Check if dataset is available."""
    print("\n✓ Checking dataset...")
    
    dataset_path = Path("data/for2sec/aug")
    manifest_path = dataset_path / "manifest.csv"
    
    if manifest_path.exists():
        print(f"  Augmented dataset found ✓")
        
        # Count samples
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
            sample_count = len(lines) - 1  # Minus header
        
        print(f"  Total samples: {sample_count}")
        return True
    else:
        print(f"  Augmented dataset not found ✗")
        print(f"  Run augmentation first:")
        print(f'    python augment_dataset.py --input-dir "path/to/raw" --output-dir "data/for2sec/aug"')
        return False


def check_model_architecture():
    """Test model creation."""
    print("\n✓ Testing model architecture...")
    
    try:
        from models.gan_model import create_gan
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        G, D = create_gan(device=device)
        
        # Test forward pass
        z = torch.randn(1, 100).to(device)
        fake = G(z)
        
        real = torch.randn(1, 1, 128, 128).to(device)
        pred = D(real)
        
        print(f"  Generator output shape: {fake.shape} ✓")
        print(f"  Discriminator output shape: {pred.shape} ✓")
        
        return True
    except Exception as e:
        print(f"  Model test failed: {e}")
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("SYSTEM CHECK FOR AUDIO DEEPFAKE DETECTION")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("GPU Support", check_gpu),
        ("Project Structure", check_project_structure),
        ("Dataset", check_dataset),
        ("Model Architecture", check_model_architecture)
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<40} {status}")
    
    print("="*60)
    
    # Recommendations
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Run augmentation (if not done):")
        print('     python augment_dataset.py --input-dir "raw" --output-dir "data/for2sec/aug"')
        print("  2. Start training:")
        print('     python train.py --data-dir "data/for2sec/aug"')
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        
        if not results["Dependencies"]:
            print("\nTo install dependencies:")
            print("  pip install -r requirements.txt")
        
        if not results["Dataset"]:
            print("\nTo prepare dataset:")
            print('  python augment_dataset.py --input-dir "path/to/raw" --output-dir "data/for2sec/aug"')
    
    print()


if __name__ == "__main__":
    main()
