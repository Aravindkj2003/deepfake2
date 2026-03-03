"""
Side-by-side comparison of CNN vs ResNet18 models.
Tests both models on a variety of audio samples and reports differences.
"""

import torch
import torch.nn as nn
import csv
import json
from pathlib import Path
from collections import defaultdict

from models.dataset import AudioSpectrogramDataset
from models.gan_model import Discriminator
from models.advanced_models import create_advanced_model


class ModelTester:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load CNN model
        print("Loading CNN model...")
        self.cnn_model = Discriminator(dropout_rate=0.3).to(device)
        cnn_checkpoint = torch.load('./checkpoints/cnn_best_model.pth', map_location=device)
        self.cnn_model.load_state_dict(cnn_checkpoint['discriminator_state_dict'])
        self.cnn_model.eval()
        
        # Load ResNet18 model
        print("Loading ResNet18 model...")
        self.resnet_model = create_advanced_model('resnet18', pretrained=False).to(device)
        resnet_checkpoint = torch.load('./checkpoints/resnet18_best.pth', map_location=device)
        self.resnet_model.load_state_dict(resnet_checkpoint['model_state_dict'])
        self.resnet_model.eval()
        
        # Load dataset for samples
        print("Loading dataset...")
        self.dataset = AudioSpectrogramDataset('./data/for2sec/aug')
        
        # Read manifest to organize by augmentation type
        self.manifest = self._read_manifest()
    
    def _read_manifest(self):
        """Read manifest.csv and organize by augmentation type."""
        manifest = defaultdict(list)
        with open('./data/for2sec/aug/manifest.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest[row['augmentation']].append({
                    'path': row['path'],
                    'label': row['label'],
                    'augmentation': row['augmentation']
                })
        return manifest
    
    def test_sample(self, spec, label):
        """Test a single spectrogram with both models."""
        spec = spec.unsqueeze(0).to(self.device)  # Add batch dimension
        
        with torch.no_grad():
            # CNN prediction (output is already sigmoid, range [0, 1])
            cnn_output = self.cnn_model(spec)
            cnn_pred = cnn_output.item()
            cnn_class = "Fake" if cnn_pred < 0.5 else "Real"
            
            # ResNet prediction (raw logits, need sigmoid)
            resnet_logit = self.resnet_model(spec)
            resnet_pred = torch.sigmoid(resnet_logit).item()
            resnet_class = "Fake" if resnet_pred < 0.5 else "Real"

        
        return {
            'cnn_pred': cnn_pred,
            'cnn_class': cnn_class,
            'resnet_pred': resnet_pred,
            'resnet_class': resnet_class,
            'truth': "Real" if label.item() == 1 else "Fake"
        }
    
    def run_tests(self):
        """Test both models on samples from each augmentation type."""
        results = defaultdict(lambda: {
            'samples_tested': 0,
            'cnn_correct': 0,
            'resnet_correct': 0,
            'both_correct': 0,
            'disagreement': 0,
            'details': []
        })
        
        print("\n" + "="*80)
        print("TESTING BOTH MODELS ON VARIOUS AUDIO TYPES")
        print("="*80)
        
        # Test 3 samples from each augmentation type (real and fake)
        for aug_type in sorted(self.manifest.keys()):
            print(f"\n--- Testing {aug_type.upper()} ---")
            
            samples = self.manifest[aug_type]
            if len(samples) == 0:
                continue
            
            # Test up to 3 samples per augmentation type
            test_samples = samples[:3]
            
            for sample in test_samples:
                spec, label = self.dataset[samples.index(sample)]
                result = self.test_sample(spec, label)
                
                # Record result
                results[aug_type]['samples_tested'] += 1
                
                cnn_correct = (result['cnn_class'] == result['truth'])
                resnet_correct = (result['resnet_class'] == result['truth'])
                
                if cnn_correct:
                    results[aug_type]['cnn_correct'] += 1
                if resnet_correct:
                    results[aug_type]['resnet_correct'] += 1
                if cnn_correct and resnet_correct:
                    results[aug_type]['both_correct'] += 1
                if result['cnn_class'] != result['resnet_class']:
                    results[aug_type]['disagreement'] += 1
                
                # Store details
                results[aug_type]['details'].append({
                    'label': result['truth'],
                    'cnn': result['cnn_class'],
                    'cnn_confidence': f"{result['cnn_pred']:.4f}",
                    'resnet': result['resnet_class'],
                    'resnet_confidence': f"{result['resnet_pred']:.4f}",
                    'match': '✓' if cnn_correct and resnet_correct else '✗'
                })
                
                # Print result
                match_str = '✓' if cnn_correct and resnet_correct else '✗'
                print(f"  {match_str} {result['truth']:5} → CNN:{result['cnn_class']:5}({result['cnn_pred']:.2%}) | ResNet:{result['resnet_class']:5}({result['resnet_pred']:.2%})")
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY BY AUDIO TYPE")
        print("="*80)
        
        total_cnn_correct = 0
        total_resnet_correct = 0
        total_samples = 0
        
        for aug_type in sorted(results.keys()):
            r = results[aug_type]
            cnn_acc = (r['cnn_correct'] / r['samples_tested'] * 100) if r['samples_tested'] > 0 else 0
            resnet_acc = (r['resnet_correct'] / r['samples_tested'] * 100) if r['samples_tested'] > 0 else 0
            
            print(f"\n{aug_type:25} | Samples: {r['samples_tested']}")
            print(f"  CNN Accuracy:    {cnn_acc:6.1f}% ({r['cnn_correct']}/{r['samples_tested']})")
            print(f"  ResNet Accuracy: {resnet_acc:6.1f}% ({r['resnet_correct']}/{r['samples_tested']})")
            print(f"  Both Correct:    {r['both_correct']}/{r['samples_tested']}")
            print(f"  Disagreement:    {r['disagreement']}/{r['samples_tested']}")
            
            total_cnn_correct += r['cnn_correct']
            total_resnet_correct += r['resnet_correct']
            total_samples += r['samples_tested']
        
        # Overall summary
        print("\n" + "="*80)
        print("OVERALL RESULTS")
        print("="*80)
        cnn_overall = (total_cnn_correct / total_samples * 100) if total_samples > 0 else 0
        resnet_overall = (total_resnet_correct / total_samples * 100) if total_samples > 0 else 0
        
        print(f"Total samples tested:    {total_samples}")
        print(f"CNN Overall Accuracy:    {cnn_overall:6.2f}% ({total_cnn_correct}/{total_samples})")
        print(f"ResNet18 Overall Acc:    {resnet_overall:6.2f}% ({total_resnet_correct}/{total_samples})")
        print(f"\n{'ResNet18 WINS' if resnet_overall > cnn_overall else 'CNN WINS'} by {abs(resnet_overall - cnn_overall):.2f}%")
        print("="*80)
        
        # Save detailed results
        with open('./test_results.json', 'w') as f:
            json.dump(dict(results), f, indent=2)
        print("\nDetailed results saved to test_results.json")
        
        return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    tester = ModelTester(device)
    results = tester.run_tests()


if __name__ == '__main__':
    main()
