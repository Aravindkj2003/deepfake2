"""
Evaluate trained discriminator model on test samples.
"""
import torch
import argparse
from pathlib import Path
import json
from tqdm import tqdm

from models.gan_model import Discriminator
from models.dataset import AudioSpectrogramDataset


def evaluate_model(checkpoint_path, data_dir, device='cuda'):
    """
    Evaluate discriminator on validation data.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Path to augmented dataset
        device: 'cuda' or 'cpu'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = Discriminator(channels=1, dropout_rate=0.0).to(device)  # No dropout for eval
    model.load_state_dict(checkpoint['discriminator_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Best validation accuracy: {checkpoint['history']['best_val_acc']:.4f}")
    print(f"  Training epoch: {checkpoint['epoch'] + 1}")
    print()
    
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = AudioSpectrogramDataset(data_dir)
    
    # Evaluate
    print("\nEvaluating...")
    correct = 0
    total = 0
    
    true_positives = 0  # Correctly predicted real
    true_negatives = 0  # Correctly predicted fake
    false_positives = 0  # Predicted real but was fake
    false_negatives = 0  # Predicted fake but was real
    
    with torch.no_grad():
        for spec, label in tqdm(dataset, desc="Evaluating"):
            spec = spec.unsqueeze(0).to(device)  # Add batch dimension
            label = label.to(device)
            
            # Predict
            pred = model(spec)
            predicted = (pred > 0.5).float()
            
            # Update counts
            is_correct = (predicted == label).item()
            correct += is_correct
            total += 1
            
            # Confusion matrix
            if label.item() == 1.0:  # True label is real
                if predicted.item() == 1.0:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:  # True label is fake
                if predicted.item() == 0.0:
                    true_negatives += 1
                else:
                    false_positives += 1
    
    # Calculate metrics
    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (Real → Real): {true_positives}")
    print(f"  True Negatives (Fake → Fake): {true_negatives}")
    print(f"  False Positives (Fake → Real): {false_positives}")
    print(f"  False Negatives (Real → Fake): {false_negatives}")
    print(f"{'='*60}\n")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'total_samples': total,
        'correct': correct,
        'confusion_matrix': {
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    }
    
    results_path = Path(checkpoint_path).parent / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained discriminator model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data/for2sec/aug',
                        help='Path to augmented dataset')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first or provide correct checkpoint path.")
        return
    
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found at {args.data_dir}")
        print("Please run augmentation first or provide correct data path.")
        return
    
    # Evaluate
    evaluate_model(args.checkpoint, args.data_dir, args.device)


if __name__ == "__main__":
    main()
