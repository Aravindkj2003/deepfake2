#!/usr/bin/env python
"""Generate augmentation breakdown statistics from test results."""
import json

with open('test_results.json') as f:
    results = json.load(f)

print('\n=== DETAILED AUGMENTATION BREAKDOWN ===\n')
print(f"{'Type':<25} {'CNN Acc':<12} {'ResNet Acc':<12} {'Agreement':<15}")
print('-' * 65)

for aug_type in sorted(results.keys()):
    data = results[aug_type]
    total = data['samples_tested']
    cnn_acc = 100 * data['cnn_correct'] / total if total > 0 else 0
    resnet_acc = 100 * data['resnet_correct'] / total if total > 0 else 0
    agreement = 100 * data['both_correct'] / total if total > 0 else 0
    
    print(f"{aug_type:<25} {cnn_acc:>6.1f}% ({data['cnn_correct']}/{total:<2})  {resnet_acc:>6.1f}% ({data['resnet_correct']}/{total:<2})  {agreement:>6.1f}% ({data['both_correct']}/{total})")

print('\n=== SUMMARY STATISTICS ===\n')
total_samples = sum(r['samples_tested'] for r in results.values())
total_cnn_correct = sum(r['cnn_correct'] for r in results.values())
total_resnet_correct = sum(r['resnet_correct'] for r in results.values())
total_both_correct = sum(r['both_correct'] for r in results.values())

print(f"Total Samples Tested:     {total_samples}")
print(f"CNN Total Correct:        {total_cnn_correct}/{total_samples} ({100*total_cnn_correct/total_samples:.1f}%)")
print(f"ResNet18 Total Correct:   {total_resnet_correct}/{total_samples} ({100*total_resnet_correct/total_samples:.1f}%)")
print(f"Both Models Agree:        {total_both_correct}/{total_samples} ({100*total_both_correct/total_samples:.1f}%)")
print(f"Disagreements:            {total_samples - total_both_correct}/{total_samples}")
