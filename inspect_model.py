#!/usr/bin/env python3
"""Inspect the trained LSTM model"""
import torch
import json

checkpoint = torch.load('models/lstm_multi_targets_best.pt', map_location='cpu', weights_only=False)

print("=" * 70)
print("TRAINED LSTM MODEL INFORMATION")
print("=" * 70)

if isinstance(checkpoint, dict):
    print(f"\n✓ Checkpoint is a dictionary with {len(checkpoint)} keys:\n")
    for key in checkpoint.keys():
        val = checkpoint[key]
        if isinstance(val, torch.Tensor):
            print(f"  • {key:30s} -> Tensor {tuple(val.shape)}")
        elif isinstance(val, (int, float, str)):
            print(f"  • {key:30s} -> {type(val).__name__}: {val}")
        elif isinstance(val, dict):
            print(f"  • {key:30s} -> Dict with {len(val)} items")
        elif isinstance(val, list):
            print(f"  • {key:30s} -> List with {len(val)} items")
        else:
            print(f"  • {key:30s} -> {type(val).__name__}")
    
    # Print metrics if available
    if 'metrics' in checkpoint:
        print(f"\n📊 MODEL METRICS:")
        metrics = checkpoint['metrics']
        for target, values in metrics.items():
            print(f"\n  {target}:")
            for metric, val in values.items():
                print(f"    - {metric}: {val:.4f}")
    
    # Print architecture info
    if 'model_config' in checkpoint:
        print(f"\n🏗️  MODEL ARCHITECTURE:")
        config = checkpoint['model_config']
        for key, val in config.items():
            print(f"  - {key}: {val}")
    
    # Print state dict sample
    if 'model_state_dict' in checkpoint:
        print(f"\n⚙️  MODEL PARAMETERS ({len(checkpoint['model_state_dict'])} layers):")
        for i, (name, param) in enumerate(list(checkpoint['model_state_dict'].items())[:10]):
            print(f"  {i+1}. {name:45s} {tuple(param.shape)}")
        if len(checkpoint['model_state_dict']) > 10:
            print(f"  ... and {len(checkpoint['model_state_dict']) - 10} more layers")

else:
    print(f"\n✗ Checkpoint is a {type(checkpoint).__name__}, not a dict")

print("\n" + "=" * 70)
print("SUMMARY: LSTM model trained for 3 targets (goals, cards, corners)")
print("=" * 70)
