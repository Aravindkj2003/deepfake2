import torch
from pathlib import Path

checkpoint_path = Path('checkpoints/best_model.pth')

if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"{'='*60}")
    print(f"CHECKPOINT INFORMATION")
    print(f"{'='*60}")
    print(f"Epoch: {checkpoint['epoch'] + 1}")
    
    history = checkpoint['history']
    epochs_done = len(history['val_acc'])
    best_acc = history['best_val_acc']
    
    print(f"Epochs completed: {epochs_done}")
    print(f"Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    if epochs_done > 0:
        print(f"Last validation accuracy: {history['val_acc'][-1]:.4f}")
        print(f"\nLast 5 epochs:")
        for i in range(max(0, epochs_done-5), epochs_done):
            print(f"  Epoch {i+1}: Val Acc = {history['val_acc'][i]:.4f}")
    
    print(f"{'='*60}")
    
    # Check if model is ready
    if best_acc >= 0.85:
        print("\n✓ Model is performing GREAT! (>85% accuracy)")
        print("  You can use this model now or continue training.")
    elif best_acc >= 0.75:
        print("\n⚠ Model is DECENT (75-85% accuracy)")
        print("  Recommend continuing training for better results.")
    elif best_acc >= 0.60:
        print("\n⚠ Model is OKAY (60-75% accuracy)")
        print("  Should continue training.")
    else:
        print("\n✗ Model needs MORE training (<60% accuracy)")
else:
    print("No checkpoint found.")
