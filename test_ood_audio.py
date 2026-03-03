#!/usr/bin/env python
"""
Test models on out-of-distribution real audio samples.
These are not from training/validation set.
"""
import torch
import librosa
import numpy as np
from pathlib import Path
from models.gan_model import Discriminator
from models.advanced_models import create_advanced_model

# Audio files to test
TEST_AUDIO_FILES = [
    r"c:\Users\Aravind KJ\Downloads\6.opus",
    r"c:\Users\Aravind KJ\Downloads\8.opus",
    r"c:\Users\Aravind KJ\Downloads\2.mp3",
]

# Model configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEL_SPEC_PARAMS = {
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
}

def load_and_preprocess_audio(filepath):
    """Load audio and convert to 128x128 mel-spectrogram."""
    try:
        # Load audio
        y, sr = librosa.load(filepath, sr=16000, duration=2.0, mono=True)
        
        # Convert to mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_mels=MEL_SPEC_PARAMS['n_mels'],
            n_fft=MEL_SPEC_PARAMS['n_fft'],
            hop_length=MEL_SPEC_PARAMS['hop_length'],
        )
        
        # Convert to dB scale
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Normalize to [-1, 1]
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        S_norm = 2 * S_norm - 1
        
        # Ensure exactly 128x128 size (pad or crop as needed)
        target_freq = 128
        target_time = 128
        current_freq, current_time = S_norm.shape
        
        # Handle frequency dimension (should be 128)
        if current_freq < target_freq:
            S_norm = np.pad(S_norm, ((0, target_freq - current_freq), (0, 0)), mode='constant', constant_values=-1)
        elif current_freq > target_freq:
            S_norm = S_norm[:target_freq, :]
        
        # Handle time dimension
        if current_time < target_time:
            S_norm = np.pad(S_norm, ((0, 0), (0, target_time - current_time)), mode='constant', constant_values=-1)
        elif current_time > target_time:
            S_norm = S_norm[:, :target_time]
        
        # Convert to tensor: shape (1, 1, 128, 128)
        spec_tensor = torch.from_numpy(S_norm).float().unsqueeze(0).unsqueeze(0)
        
        return spec_tensor, True, None
    except Exception as e:
        return None, False, str(e)

def main():
    print("\n" + "="*80)
    print("TESTING ON OUT-OF-DISTRIBUTION REAL AUDIOS")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    
    # CNN
    cnn_model = Discriminator(dropout_rate=0.3).to(DEVICE)
    cnn_ckpt = torch.load('./checkpoints/cnn_best_model.pth', map_location=DEVICE)
    cnn_model.load_state_dict(cnn_ckpt['discriminator_state_dict'])
    cnn_model.eval()
    print("✓ CNN loaded (98.45% accuracy)")
    
    # ResNet18
    resnet_model = create_advanced_model('resnet18', pretrained=False).to(DEVICE)
    resnet_ckpt = torch.load('./checkpoints/resnet18_best.pth', map_location=DEVICE)
    resnet_model.load_state_dict(resnet_ckpt['model_state_dict'])
    resnet_model.eval()
    print("✓ ResNet18 loaded (99.50% accuracy)")
    
    print("\n" + "-"*80)
    print(f"Testing {len(TEST_AUDIO_FILES)} real audio samples (out-of-distribution)")
    print("-"*80)
    
    results = {
        'total': len(TEST_AUDIO_FILES),
        'cnn_correct': 0,
        'resnet_correct': 0,
        'both_correct': 0,
        'details': []
    }
    
    for idx, audio_file in enumerate(TEST_AUDIO_FILES, 1):
        filepath = Path(audio_file)
        
        if not filepath.exists():
            print(f"\n❌ File {idx}: {filepath.name} - NOT FOUND")
            results['details'].append({
                'file': filepath.name,
                'status': 'not_found',
                'error': 'File does not exist'
            })
            continue
        
        print(f"\n[{idx}/{len(TEST_AUDIO_FILES)}] {filepath.name}")
        print(f"  Path: {filepath}")
        print(f"  Size: {filepath.stat().st_size / 1024:.1f} KB")
        
        # Load and preprocess
        spec, success, error = load_and_preprocess_audio(str(filepath))
        
        if not success:
            print(f"  ❌ Load Failed: {error}")
            results['details'].append({
                'file': filepath.name,
                'status': 'load_failed',
                'error': error
            })
            continue
        
        print(f"  Spectrogram shape: {spec.shape}")
        
        # Move to device
        spec = spec.to(DEVICE)
        
        # Test on both models
        with torch.no_grad():
            # CNN (output already has sigmoid)
            cnn_output = cnn_model(spec)
            cnn_prob = cnn_output.item()
            cnn_pred = "Real" if cnn_prob >= 0.5 else "Fake"
            
            # ResNet18 (raw logits)
            resnet_logit = resnet_model(spec)
            resnet_prob = torch.sigmoid(resnet_logit).item()
            resnet_pred = "Real" if resnet_prob >= 0.5 else "Fake"
        
        # Ground truth: these are REAL audios
        truth = "Real"
        cnn_correct = cnn_pred == truth
        resnet_correct = resnet_pred == truth
        both_correct = cnn_correct and resnet_correct
        
        # Update results
        results['cnn_correct'] += int(cnn_correct)
        results['resnet_correct'] += int(resnet_correct)
        results['both_correct'] += int(both_correct)
        
        # Print results
        cnn_status = "✓" if cnn_correct else "✗"
        resnet_status = "✓" if resnet_correct else "✗"
        
        print(f"  CNN:      {cnn_status} Pred={cnn_pred:4s} (prob={cnn_prob:.4f})")
        print(f"  ResNet18: {resnet_status} Pred={resnet_pred:4s} (prob={resnet_prob:.4f})")
        print(f"  Truth:    {truth:4s} (should be REAL)")
        
        results['details'].append({
            'file': filepath.name,
            'status': 'tested',
            'ground_truth': truth,
            'cnn': {
                'prediction': cnn_pred,
                'probability': float(cnn_prob),
                'correct': cnn_correct
            },
            'resnet': {
                'prediction': resnet_pred,
                'probability': float(resnet_prob),
                'correct': resnet_correct
            }
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - OUT-OF-DISTRIBUTION REAL AUDIO TESTS")
    print("="*80)
    tested = sum(1 for d in results['details'] if d['status'] == 'tested')
    
    if tested > 0:
        print(f"\nTotal samples tested: {tested}")
        print(f"CNN Accuracy:       {100*results['cnn_correct']/tested:.1f}% ({results['cnn_correct']}/{tested})")
        print(f"ResNet18 Accuracy:  {100*results['resnet_correct']/tested:.1f}% ({results['resnet_correct']}/{tested})")
        print(f"Both Correct:       {100*results['both_correct']/tested:.1f}% ({results['both_correct']}/{tested})")
        
        # Interpretation
        print("\n" + "-"*80)
        print("INTERPRETATION:")
        print("-"*80)
        
        if results['cnn_correct'] == tested and results['resnet_correct'] == tested:
            print("✓ EXCELLENT: Both models correctly identified all real audios!")
            print("  This shows excellent generalization to out-of-distribution real audio.")
        elif results['cnn_correct'] == tested:
            print("✓ CNN: 100% accuracy on real audio")
            print(f"⚠ ResNet: {100*results['resnet_correct']/tested:.0f}% accuracy - may have false positives")
        elif results['resnet_correct'] == tested:
            print(f"⚠ CNN: {100*results['cnn_correct']/tested:.0f}% accuracy - may have false positives")
            print("✓ ResNet: 100% accuracy on real audio")
        else:
            print(f"⚠ WARNING: Models struggled with out-of-distribution real audio")
            print(f"   CNN: {100*results['cnn_correct']/tested:.0f}%, ResNet: {100*results['resnet_correct']/tested:.0f}%")
    else:
        print("⚠ No samples were successfully tested")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
