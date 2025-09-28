#!/usr/bin/env python3
"""
Quick Coverage Control Ablation Study
Tests the impact of coverage control on near-pole learning
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

sys.path.append('/home/zsemed/ZeroProofML')

from zeroproof.model import TRRationalLayer
from zeroproof.training import TRTrainer
from zeroproof.core import TRTensor

def load_dataset(path):
    """Load the small IK dataset"""
    with open(path, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    n_total = len(samples)
    
    # Split 80/20
    n_train = int(0.8 * n_total)
    
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]
    
    # Extract inputs and targets
    train_x = torch.tensor([s['target_position'] for s in train_samples], dtype=torch.float32)
    train_y = torch.tensor([s['joint_angles'] for s in train_samples], dtype=torch.float32)
    test_x = torch.tensor([s['target_position'] for s in test_samples], dtype=torch.float32)
    test_y = torch.tensor([s['joint_angles'] for s in test_samples], dtype=torch.float32)
    
    # Calculate det(J) for bucketing
    train_det = torch.tensor([abs(s['det_jacobian']) for s in train_samples], dtype=torch.float32)
    test_det = torch.tensor([abs(s['det_jacobian']) for s in test_samples], dtype=torch.float32)
    
    return (train_x, train_y, train_det), (test_x, test_y, test_det)

def create_model(input_dim=4, output_dim=2, use_coverage_control=True):
    """Create a simple TR model"""
    class SimpleTRModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tr_layer = TRRationalLayer(
                in_features=input_dim,
                out_features=output_dim,
                degree_p=3,
                degree_q=2,
                shared_q=True
            )
            self.use_coverage = use_coverage_control
            
        def forward(self, x):
            return self.tr_layer(x)
    
    return SimpleTRModel()

def compute_bucket_mse(pred, target, det_j):
    """Compute MSE for each bucket"""
    buckets = {
        'B0': (0, 1e-5),
        'B1': (1e-5, 1e-4),
        'B2': (1e-4, 1e-3),
        'B3': (1e-3, 1e-2),
        'B4': (1e-2, float('inf'))
    }
    
    results = {}
    for name, (low, high) in buckets.items():
        mask = (det_j > low) & (det_j <= high)
        if mask.sum() > 0:
            bucket_pred = pred[mask]
            bucket_target = target[mask]
            mse = ((bucket_pred - bucket_target) ** 2).mean().item()
            results[name] = {'mse': mse, 'count': mask.sum().item()}
        else:
            results[name] = {'mse': 0.0, 'count': 0}
    
    return results

def train_with_config(use_coverage, epochs=10, lr=0.01):
    """Train model with or without coverage control"""
    
    # Load data
    train_data, test_data = load_dataset('/home/zsemed/ZeroProofML/data/rr_ik_dataset_small.json')
    train_x, train_y, train_det = train_data
    test_x, test_y, test_det = test_data
    
    # Create model
    model = create_model(use_coverage_control=use_coverage)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training metrics
    train_losses = []
    coverage_history = []
    bucket_history = []
    
    print(f"\n{'='*60}")
    print(f"Training {'WITH' if use_coverage else 'WITHOUT'} Coverage Control")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        pred = model(train_x)
        
        # Basic MSE loss
        mse_loss = ((pred - train_y) ** 2).mean()
        
        # Coverage control (if enabled)
        if use_coverage:
            # Compute coverage statistics
            with torch.no_grad():
                # Check how many samples are in near-pole regions
                near_pole_mask = train_det < 1e-3
                near_pole_coverage = near_pole_mask.float().mean().item()
                
                # Add coverage penalty if too few near-pole samples
                target_coverage = 0.15  # Want at least 15% near-pole
                if near_pole_coverage < target_coverage:
                    # Upweight near-pole samples
                    weights = torch.ones_like(train_det)
                    weights[near_pole_mask] *= 2.0  # Double weight for near-pole
                    weighted_loss = (weights * ((pred - train_y) ** 2).mean(dim=1)).mean()
                    loss = weighted_loss
                else:
                    loss = mse_loss
        else:
            loss = mse_loss
            near_pole_coverage = 0.0
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Log metrics
        train_losses.append(loss.item())
        coverage_history.append(near_pole_coverage)
        
        # Evaluate on test set
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(test_x)
                test_mse = ((test_pred - test_y) ** 2).mean().item()
                bucket_mse = compute_bucket_mse(test_pred, test_y, test_det)
                bucket_history.append(bucket_mse)
                
                print(f"Epoch {epoch+1:3d}: Train Loss={loss.item():.4f}, Test MSE={test_mse:.4f}, "
                      f"Near-pole coverage={near_pole_coverage:.1%}")
                if epoch == epochs - 1:
                    print("\nFinal Bucket MSE:")
                    for name, stats in bucket_mse.items():
                        if stats['count'] > 0:
                            print(f"  {name}: {stats['mse']:.6f} (n={stats['count']})")
    
    train_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x)
        final_mse = ((test_pred - test_y) ** 2).mean().item()
        final_buckets = compute_bucket_mse(test_pred, test_y, test_det)
    
    results = {
        'use_coverage': use_coverage,
        'final_test_mse': final_mse,
        'bucket_mse': final_buckets,
        'train_time': train_time,
        'train_losses': train_losses,
        'coverage_history': coverage_history,
        'epochs': epochs
    }
    
    return results

def main():
    """Run ablation study"""
    
    print("\n" + "="*70)
    print(" COVERAGE CONTROL ABLATION STUDY")
    print("="*70)
    
    # Run experiments
    results_with = train_with_config(use_coverage=True, epochs=15)
    results_without = train_with_config(use_coverage=False, epochs=15)
    
    # Analysis
    print("\n" + "="*70)
    print(" ABLATION RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'WITH Coverage':<20} {'WITHOUT Coverage':<20}")
    print("-" * 70)
    
    # Overall MSE
    print(f"{'Overall Test MSE':<30} {results_with['final_test_mse']:<20.6f} {results_without['final_test_mse']:<20.6f}")
    
    # Near-pole buckets (B0, B1)
    for bucket in ['B0', 'B1', 'B2']:
        with_mse = results_with['bucket_mse'].get(bucket, {}).get('mse', 0)
        without_mse = results_without['bucket_mse'].get(bucket, {}).get('mse', 0)
        
        if with_mse > 0 and without_mse > 0:
            improvement = (without_mse - with_mse) / without_mse * 100
            print(f"{bucket + ' MSE':<30} {with_mse:<20.6f} {without_mse:<20.6f} ({improvement:+.1f}%)")
        else:
            print(f"{bucket + ' MSE':<30} {with_mse:<20.6f} {without_mse:<20.6f}")
    
    # Training time
    print(f"{'Training Time (s)':<30} {results_with['train_time']:<20.2f} {results_without['train_time']:<20.2f}")
    
    # Average coverage
    avg_cov_with = np.mean(results_with['coverage_history']) * 100
    avg_cov_without = np.mean(results_without['coverage_history']) * 100
    print(f"{'Avg Near-pole Coverage (%)':<30} {avg_cov_with:<20.1f} {avg_cov_without:<20.1f}")
    
    # Save results
    output_dir = Path('/home/zsemed/ZeroProofML/results/robotics/ablation_coverage')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ablation_results = {
        'with_coverage': results_with,
        'without_coverage': results_without,
        'analysis': {
            'b0_improvement': (results_without['bucket_mse'].get('B0', {}).get('mse', 0) - 
                              results_with['bucket_mse'].get('B0', {}).get('mse', 0)),
            'b1_improvement': (results_without['bucket_mse'].get('B1', {}).get('mse', 0) - 
                              results_with['bucket_mse'].get('B1', {}).get('mse', 0)),
            'coverage_impact': avg_cov_with - avg_cov_without
        }
    }
    
    with open(output_dir / 'coverage_ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/coverage_ablation_results.json")
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Panel 1: Training loss
        ax1 = axes[0]
        ax1.plot(results_with['train_losses'], label='With Coverage', color='green', linewidth=2)
        ax1.plot(results_without['train_losses'], label='Without Coverage', color='red', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Convergence')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Panel 2: Near-pole coverage over time
        ax2 = axes[1]
        epochs = np.arange(len(results_with['coverage_history']))
        ax2.plot(epochs, np.array(results_with['coverage_history']) * 100, 
                marker='o', label='With Controller', color='green')
        ax2.plot(epochs, np.array(results_without['coverage_history']) * 100,
                marker='s', label='Without Controller', color='red')
        ax2.axhline(y=15, color='k', linestyle='--', alpha=0.5, label='Target (15%)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Near-pole Coverage (%)')
        ax2.set_title('Coverage Evolution')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Panel 3: Bucket MSE comparison
        ax3 = axes[2]
        buckets = ['B0', 'B1', 'B2']
        x = np.arange(len(buckets))
        width = 0.35
        
        with_mse = [results_with['bucket_mse'].get(b, {}).get('mse', 0) for b in buckets]
        without_mse = [results_without['bucket_mse'].get(b, {}).get('mse', 0) for b in buckets]
        
        bars1 = ax3.bar(x - width/2, with_mse, width, label='With Coverage', color='green', alpha=0.7)
        bars2 = ax3.bar(x + width/2, without_mse, width, label='Without Coverage', color='red', alpha=0.7)
        
        ax3.set_xlabel('Bucket')
        ax3.set_ylabel('MSE')
        ax3.set_title('Near-pole Performance')
        ax3.set_xticks(x)
        ax3.set_xticklabels(buckets)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Coverage Control Ablation Study', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(output_dir / 'coverage_ablation.png', dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / 'coverage_ablation.pdf', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved to coverage_ablation.png/pdf")
        
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    return ablation_results

if __name__ == '__main__':
    results = main()
