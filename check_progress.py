import json
from pathlib import Path

history_path = Path('checkpoints/training_history.json')

if history_path.exists():
    with open(history_path) as f:
        h = json.load(f)
    
    epochs_done = len(h['val_acc'])
    best_acc = h['best_val_acc']
    last_acc = h['val_acc'][-1]
    
    print(f"{'='*60}")
    print(f"TRAINING PROGRESS REPORT")
    print(f"{'='*60}")
    print(f"Epochs completed: {epochs_done}/50")
    print(f"Best validation accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Last validation accuracy: {last_acc:.4f} ({last_acc*100:.2f}%)")
    print(f"\nLast 5 epochs:")
    for i in range(max(0, epochs_done-5), epochs_done):
        print(f"  Epoch {i+1}: Val Acc = {h['val_acc'][i]:.4f}")
    print(f"{'='*60}")
    
    # Check if model is ready
    if best_acc >= 0.85:
        print("\n✓ Model is performing well! (>85% accuracy)")
        print("  You can use this model now or continue training.")
    elif best_acc >= 0.75:
        print("\n⚠ Model is decent (75-85% accuracy)")
        print("  Recommend continuing training for better results.")
    else:
        print("\n✗ Model needs more training (<75% accuracy)")
else:
    print("No training history found. Training hasn't started yet.")
