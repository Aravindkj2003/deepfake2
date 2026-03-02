import torch
from pathlib import Path

# Check CNN model
cnn_path = Path('checkpoints/cnn_best_model.pth')

if cnn_path.exists():
    checkpoint = torch.load(cnn_path, map_location='cpu', weights_only=False)
    history = checkpoint['history']
    
    print(f"{'='*60}")
    print(f"CNN DISCRIMINATOR TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"Epochs trained: {len(history['val_acc'])}")
    print(f"Best validation accuracy: {history['best_val_acc']:.4f} ({history['best_val_acc']*100:.2f}%)")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
    
    print(f"\nTraining Progress (last 5 epochs):")
    for i in range(max(0, len(history['val_acc'])-5), len(history['val_acc'])):
        acc = history['val_acc'][i]
        loss = history['val_loss'][i]
        print(f"  Epoch {i+1}: Loss = {loss:.4f} | Accuracy = {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"{'='*60}")
    
    # Assessment
    best_acc = history['best_val_acc']
    if best_acc >= 0.90:
        print("\n✅ EXCELLENT! (90%+ accuracy)")
        print("   Ready for hybrid GAN fine-tuning or deployment!")
    elif best_acc >= 0.85:
        print("\n✅ VERY GOOD! (85-90% accuracy)")
        print("   Can proceed with hybrid GAN training.")
    elif best_acc >= 0.80:
        print("\n✅ GOOD! (80-85% accuracy)")
        print("   Can proceed with hybrid GAN training.")
    elif best_acc >= 0.75:
        print("\n⚠️  ACCEPTABLE (75-80% accuracy)")
        print("   Can use, but could be better. Hybrid GAN may improve it.")
    elif best_acc >= 0.70:
        print("\n⚠️  OKAY (70-75% accuracy)")
        print("   Functional but could use improvement.")
    else:
        print("\n❌ NEEDS IMPROVEMENT (<70% accuracy)")
        print("   Consider retraining with different hyperparameters.")
    
    print(f"{'='*60}\n")
    
else:
    print("CNN checkpoint not found at checkpoints/cnn_best_model.pth")
    print("CNN training may not have completed successfully.")
