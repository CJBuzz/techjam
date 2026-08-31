import json
import numpy as np

# Read all_candidates_validation
with open('artifacts/all_candidates_validation.json') as f:
    all_candidates = json.load(f)

# Extract results
checkpoints = list(all_candidates['checkpoints'].keys())
conditions = all_candidates['_metadata']['conditions']

print("=" * 80)
print("CHECKPOINT COMPARISON - VALIDATION SET")
print("=" * 80)

# Collect results by checkpoint
results = {}
for ckpt_path in checkpoints:
    ckpt_name = ckpt_path.split('\\')[-1]
    ckpt_data = all_candidates['checkpoints'][ckpt_path]
    
    results[ckpt_name] = {}
    
    # Get clean metrics
    clean_metrics = ckpt_data['clean']['overall']
    results[ckpt_name]['clean_accuracy'] = clean_metrics['accuracy']
    results[ckpt_name]['clean_roc_auc'] = clean_metrics['roc_auc']
    results[ckpt_name]['clean_f1'] = clean_metrics['f1']
    
    # Calculate mean and worst on all conditions
    all_accuracies = []
    for condition in conditions:
        if condition in ckpt_data:
            acc = ckpt_data[condition]['overall']['accuracy']
            all_accuracies.append(acc)
    
    results[ckpt_name]['mean_accuracy_all'] = np.mean(all_accuracies)
    results[ckpt_name]['min_accuracy_all'] = np.min(all_accuracies)
    results[ckpt_name]['worst_condition'] = conditions[np.argmin(all_accuracies)]
    
    # Get metrics per transformation type
    jpeg_accs = [ckpt_data[c]['overall']['accuracy'] for c in conditions if c.startswith('jpeg')]
    blur_accs = [ckpt_data[c]['overall']['accuracy'] for c in conditions if c.startswith('blur')]
    resize_accs = [ckpt_data[c]['overall']['accuracy'] for c in conditions if c.startswith('resize')]
    noise_accs = [ckpt_data[c]['overall']['accuracy'] for c in conditions if c.startswith('noise')]
    color_accs = [ckpt_data[c]['overall']['accuracy'] for c in conditions if c.startswith('color')]
    crop_accs = [ckpt_data[c]['overall']['accuracy'] for c in conditions if c.startswith('crop')]
    
    results[ckpt_name]['jpeg_mean'] = np.mean(jpeg_accs) if jpeg_accs else 0
    results[ckpt_name]['blur_mean'] = np.mean(blur_accs) if blur_accs else 0
    results[ckpt_name]['resize_mean'] = np.mean(resize_accs) if resize_accs else 0
    results[ckpt_name]['noise_mean'] = np.mean(noise_accs) if noise_accs else 0
    results[ckpt_name]['color_mean'] = np.mean(color_accs) if color_accs else 0
    results[ckpt_name]['crop_mean'] = np.mean(crop_accs) if crop_accs else 0

# Print summary table
print("\n### OVERALL METRICS")
print(f"{'Checkpoint':<30} {'Clean Acc':<12} {'Mean All':<12} {'Worst':<12} {'Min Acc':<12}")
print("-" * 80)
for name in sorted(results.keys()):
    r = results[name]
    print(f"{name:<30} {r['clean_accuracy']:.4f}      {r['mean_accuracy_all']:.4f}      {r['min_accuracy_all']:.4f}      ({r['worst_condition']})")

print("\n### BY TRANSFORMATION TYPE")
print(f"{'Checkpoint':<30} {'JPEG':<10} {'Blur':<10} {'Resize':<10} {'Noise':<10} {'Color':<10} {'Crop':<10}")
print("-" * 80)
for name in sorted(results.keys()):
    r = results[name]
    print(f"{name:<30} {r['jpeg_mean']:.4f}    {r['blur_mean']:.4f}    {r['resize_mean']:.4f}    {r['noise_mean']:.4f}    {r['color_mean']:.4f}    {r['crop_mean']:.4f}")

print("\n### PER-CONDITION DETAILED RESULTS")
for name in sorted(results.keys()):
    print(f"\n{name}:")
    ckpt_data = all_candidates['checkpoints'][list(all_candidates['checkpoints'].keys())[list(results.keys()).index(name)]]
    print(f"  {'Condition':<20} {'Accuracy':<12} {'ROC-AUC':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("  " + "-" * 78)
    for condition in conditions:
        if condition in ckpt_data:
            metrics = ckpt_data[condition]['overall']
            print(f"  {condition:<20} {metrics['accuracy']:.4f}       {metrics['roc_auc']:.4f}       {metrics['f1']:.4f}       {metrics['precision']:.4f}       {metrics['recall']:.4f}")
