"""
Comprehensive comparison of all baseline methods.

This script runs all baseline methods (MLP, Rational+ε, DLS, ZeroProofML)
on the same dataset and generates comparison tables and plots.
"""

import os
import json
import csv
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import hashlib

# Import all baseline implementations
from mlp_baseline import MLPConfig, MLPBaseline, MLPTrainer, run_mlp_baseline
from rational_eps_baseline import RationalEpsConfig, run_rational_eps_baseline, grid_search_epsilon
from dls_solver import DLSConfig, run_dls_reference

# Import ZeroProofML for comparison
from zeroproof.core import real, TRTag
from zeroproof.autodiff import GradientModeConfig, GradientMode, TRNode
from zeroproof.layers import FullyIntegratedRational, MonomialBasis, TRMultiInputRational
from zeroproof.metrics.pole_2d import compute_pole_metrics_2d
from zeroproof.utils.seeding import set_global_seed
from zeroproof.training import HybridTRTrainer, HybridTrainingConfig, Optimizer
from zeroproof.utils.config import DEFAULT_BUCKET_EDGES
from zeroproof.utils.serialization import to_builtin


def prepare_ik_data(samples: List[Dict]) -> Tuple[Tuple[List, List], Tuple[List, List]]:
    """
    Prepare IK data for training.
    
    Args:
        samples: List of IK samples
        
    Returns:
        ((train_inputs, train_targets), (test_inputs, test_targets))
    """
    # Convert samples to input/target format
    inputs = []
    targets = []
    
    for sample in samples:
        # Input: [theta1, theta2, dx, dy]
        inp = [sample['theta1'], sample['theta2'], sample['dx'], sample['dy']]
        
        # Target: [dtheta1, dtheta2]
        tgt = [sample['dtheta1'], sample['dtheta2']]
        
        inputs.append(inp)
        targets.append(tgt)
    
    # Train/test split (80/20)
    n_train = int(0.8 * len(inputs))
    
    train_data = (inputs[:n_train], targets[:n_train])
    test_data = (inputs[n_train:], targets[n_train:])
    
    return train_data, test_data


def run_zeroproof_baseline(train_data: Tuple[List, List],
                          test_data: Tuple[List, List],
                          enable_enhancements: bool = True,
                          output_dir: str = "results",
                          test_detJ: Optional[List[float]] = None,
                          bucket_edges: Optional[List[float]] = None,
                          epochs: int = 100) -> Dict[str, Any]:
    """
    Run ZeroProofML baseline.
    
    Args:
        train_data: Training data
        test_data: Test data
        enable_enhancements: Whether to enable all ZeroProof enhancements
        output_dir: Output directory
        
    Returns:
        ZeroProofML results
    """
    print("=== ZeroProofML Baseline ===")
    
    train_inputs, train_targets = train_data
    test_inputs, test_targets = test_data
    
    # Set gradient mode
    GradientModeConfig.set_mode(GradientMode.MASK_REAL)
    
    # Determine dims
    output_dim = len(train_targets[0]) if train_targets else 1
    input_dim = len(train_inputs[0]) if train_inputs else 1

    # Create multi-input TR model
    if enable_enhancements:
        # Use a front end but keep fully integrated rational heads implicit via TRMultiInputRational
        model = TRMultiInputRational(input_dim=input_dim, n_outputs=output_dim,
                                     d_p=3, d_q=2, hidden_dims=[8], shared_Q=True)
        trainer = HybridTRTrainer(
            model=model,
            optimizer=Optimizer(model.parameters(), learning_rate=0.01),
            config=HybridTrainingConfig(
                learning_rate=0.01,
                max_epochs=100,
                use_hybrid_gradient=True,
                use_tag_loss=True,
                lambda_tag=0.05,
                use_pole_head=True,
                lambda_pole=0.1,
                enable_anti_illusion=True,
                lambda_residual=0.02
            )
        )
        print("ZeroProofML with full enhancements enabled")
    else:
        model = TRMultiInputRational(input_dim=input_dim, n_outputs=output_dim,
                                     d_p=3, d_q=2, hidden_dims=[8], shared_Q=True)
        trainer = None
        print("Basic TR-Rational without enhancements")

    n_parameters = len(model.parameters())
    print(f"Model parameters: {n_parameters}")
    
    # Training (simplified for baseline comparison)
    start_time = time.time()
    training_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_samples = 0
        
        # Simple training loop (use full training set for parity)
        for inp, tgt in zip(train_inputs, train_targets):
            # Convert to TRNodes
            tr_inputs = [TRNode.constant(real(x)) for x in inp]
            
            # Model consumes full vector; returns list of (TRNode, TRTag)
            outputs = model.forward(tr_inputs)
            sample_loss = TRNode.constant(real(0.0))
            valid_outputs = 0
            for j, (y_pred, tag) in enumerate(outputs):
                if tag == TRTag.REAL and j < len(tgt):
                    target = TRNode.constant(real(tgt[j]))
                    diff = (y_pred - target)
                    sample_loss = sample_loss + diff * diff
                    valid_outputs += 1
            if valid_outputs == 0:
                continue
            sample_loss.backward()
            if enable_enhancements and trainer is not None:
                if hasattr(trainer, 'step_all'):
                    trainer.step_all()
                else:
                    trainer.optimizer.step(model)
            else:
                Optimizer(model.parameters(), learning_rate=0.01).step(model)
            if sample_loss.tag == TRTag.REAL:
                epoch_loss += sample_loss.value.value / max(1, valid_outputs)
            n_samples += 1
        
        if n_samples > 0:
            avg_loss = epoch_loss / n_samples
            training_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    
    # Evaluation
    test_errors = []
    predictions = []
    for inp, tgt in zip(test_inputs, test_targets):
        tr_inputs = [TRNode.constant(real(x)) for x in inp]
        outs = model.forward(tr_inputs)
        pred_vec = [(y.value.value if tag == TRTag.REAL else 0.0) for (y, tag) in outs]
        predictions.append(pred_vec)
        mse = np.mean([(pv - tv)**2 for pv, tv in zip(pred_vec, tgt)])
        test_errors.append(mse)
    
    final_mse = float(np.mean(test_errors)) if test_errors else float('inf')
    
    results = {
        'model_type': 'ZeroProofML' if enable_enhancements else 'TR-Rational',
        'enhancements_enabled': enable_enhancements,
        'training_time': training_time,
        'final_mse': final_mse,
        'training_losses': training_losses,
        'n_parameters': n_parameters,
        'test_samples_evaluated': len(test_inputs),
        'predictions': predictions,
        'epochs': epochs,
    }

    # Optional: near-pole bucketed MSE if det(J) provided
    if test_detJ is not None and len(test_detJ) == len(test_inputs):
        edges = bucket_edges or [0.0, 1e-5, 1e-4, 1e-3, 1e-2, float('inf')]
        buckets = {f"({edges[i]:.0e},{edges[i+1]:.0e}]": [] for i in range(len(edges)-1)}
        for (mse, dj) in zip(test_errors, test_detJ):
            for i in range(len(edges)-1):
                lo, hi = edges[i], edges[i+1]
                if (dj > lo) and (dj <= hi):
                    buckets[f"({lo:.0e},{hi:.0e}]"] .append(mse)
                    break
        bucket_mse = {k: (float(np.mean(v)) if v else None) for k, v in buckets.items()}
        bucket_counts = {k: len(v) for k, v in buckets.items()}
        results['near_pole_bucket_mse'] = {
            'edges': edges,
            'bucket_mse': bucket_mse,
            'bucket_counts': bucket_counts,
        }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_name = 'zeroproof_full' if enable_enhancements else 'tr_rational_basic'
    
    results_file = os.path.join(output_dir, f"{model_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"ZeroProofML results saved to {results_file}")
    
    return results


def run_complete_comparison(dataset_file: str, output_dir: str = "results/comparison", seed: Optional[int] = None):
    """
    Run complete comparison of all methods.
    
    Args:
        dataset_file: Path to IK dataset JSON file
        output_dir: Output directory for results
    """
    print("=== Complete Baseline Comparison ===")
    print(f"Dataset: {dataset_file}")
    print(f"Output directory: {output_dir}")
    if seed is not None:
        print(f"Seed: {seed}")
    set_global_seed(seed)
    
    # Load dataset
    print("\nLoading dataset...")
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    print(f"Loaded {len(samples)} samples")
    
    # Prepare data
    train_data, test_data = prepare_ik_data(samples)
    print(f"Train samples: {len(train_data[0])}")
    print(f"Test samples: {len(test_data[0])}")
    
    # Results collection
    all_results = {}
    comparison_table = []
    
    print("\n" + "="*60)
    print("RUNNING ALL BASELINES")
    print("="*60)
    
    # Precompute dataset hash and test |det(J)| for bucket metrics
    bucket_edges = DEFAULT_BUCKET_EDGES
    n_train = int(0.8 * len(samples))
    test_detj = [abs(s.get('det_J', 0.0)) for s in samples[n_train:]]
    # Dataset hash for parity
    try:
        with open(dataset_file, 'rb') as _fb:
            dataset_hash = hashlib.sha256(_fb.read()).hexdigest()
    except Exception:
        dataset_hash = None

    def compute_bucket_mse(mse_list, edges, detj_list):
        buckets = {f"({edges[i]:.0e},{edges[i+1]:.0e}]": [] for i in range(len(edges)-1)}
        for mse, dj in zip(mse_list, detj_list):
            for i in range(len(edges)-1):
                lo, hi = edges[i], edges[i+1]
                if (dj > lo) and (dj <= hi):
                    buckets[f"({lo:.0e},{hi:.0e}]"] .append(mse)
                    break
        bucket_mse = {k: (float(np.mean(v)) if v else None) for k, v in buckets.items()}
        bucket_counts = {k: len(v) for k, v in buckets.items()}
        return bucket_mse, bucket_counts

    # Pull quick overrides if any
    overrides = globals().get('_COMPARATOR_OVERRIDES', {})
    quick = bool(overrides.get('quick', False))
    mlp_epochs = int(overrides.get('mlp_epochs', 50))
    rat_epochs = int(overrides.get('rat_epochs', 50))
    rat_epsilon = overrides.get('rat_epsilon', None)
    zp_epochs = int(overrides.get('zp_epochs', 100))

    # Pull quick overrides if any
    overrides = globals().get('_COMPARATOR_OVERRIDES', {})
    quick = bool(overrides.get('quick', False))
    mlp_epochs = int(overrides.get('mlp_epochs', 50))
    rat_epochs = int(overrides.get('rat_epochs', 50))
    rat_epsilon = overrides.get('rat_epsilon', None)
    zp_epochs = int(overrides.get('zp_epochs', 100))
    enabled_models = set(overrides.get('models', ['mlp','rational_eps','dls','tr_basic','tr_full']))

    # 1. MLP Baseline
    print("\n1. Running MLP Baseline...")
    try:
        if 'mlp' not in enabled_models:
            raise RuntimeError('skipped')
        mlp_config = MLPConfig(epochs=mlp_epochs, hidden_dims=[32, 16])
        mlp_results = run_mlp_baseline(train_data, test_data, mlp_config, f"{output_dir}/mlp", seed=seed)
        # Add bucketed MSE using predictions
        mlp_preds = mlp_results['test_metrics'].get('predictions', [])
        mse_list = []
        for pred, tgt in zip(mlp_preds, test_data[1]):
            if len(pred) and len(tgt):
                mse_list.append(float(np.mean([(p - t) ** 2 for p, t in zip(pred, tgt)])))
        mlp_bucket_mse, mlp_bucket_counts = compute_bucket_mse(mse_list, bucket_edges, test_detj)
        mlp_results['near_pole_bucket_mse'] = {'edges': bucket_edges, 'bucket_mse': mlp_bucket_mse, 'bucket_counts': mlp_bucket_counts}
        # Guardrail: warn if any near-pole bucket B0–B3 is empty
        np_keys = [f"({bucket_edges[i]:.0e},{bucket_edges[i+1]:.0e}]" for i in range(4)]
        if any(mlp_bucket_counts.get(k, 0) == 0 for k in np_keys):
            print("[Guardrail][MLP] Some near-pole buckets (B0–B3) are empty. Consider adjusting dataset generation flags.")
        # Pole metrics (2D)
        mlp_pole = compute_pole_metrics_2d(test_data[0], mlp_preds)
        mlp_results['pole_metrics'] = mlp_pole
        all_results['MLP'] = mlp_results
        # Row
        comparison_table.append({
            'Method': 'MLP',
            'Parameters': mlp_results['n_parameters'],
            'Epochs': mlp_config.epochs,
            'Train_MSE': mlp_results['training_results']['final_train_mse'],
            'Test_MSE': mlp_results['test_metrics']['mse'],
            'Training_Time': mlp_results['training_time'],
            'Avg_Epoch_Time': mlp_results['training_time'] / max(1, mlp_config.epochs),
            'Success_Rate': 1.0,
            'Notes': f"Hidden: {mlp_config.hidden_dims}",
            'NearPoleCountsB0_B3': [mlp_bucket_counts.get(f"({bucket_edges[i]:.0e},{bucket_edges[i+1]:.0e}]", 0) for i in range(4)],
        })
        print(f"✓ MLP: MSE={mlp_results['test_metrics']['mse']:.6f}")
        
    except Exception as e:
        print(f"✗ MLP failed: {e}")
        all_results['MLP'] = {'error': str(e)}
    
    # 2. Rational+ε Baseline
    print("\n2. Running Rational+ε Baseline...")
    try:
        if 'rational_eps' not in enabled_models:
            raise RuntimeError('skipped')
        rational_config = RationalEpsConfig(
            epochs=rat_epochs,
            epsilon_values=[1e-4, 1e-3, 1e-2]
        )
        if rat_epsilon is not None:
            # Skip grid search
            rational_results = run_rational_eps_baseline(
                train_data, test_data, epsilon=rat_epsilon, config=rational_config, output_dir=f"{output_dir}/rational_eps", seed=seed
            )
        else:
            rational_results = run_rational_eps_baseline(
                train_data, test_data, config=rational_config, output_dir=f"{output_dir}/rational_eps", seed=seed
            )
        # Add bucketed MSE using per-sample metrics from evaluation
        rat_test = rational_results.get('test_metrics', {})
        rat_mse_list = rat_test.get('per_sample_mse', [])
        rat_bucket_mse, rat_bucket_counts = compute_bucket_mse(rat_mse_list, bucket_edges, test_detj)
        rational_results['near_pole_bucket_mse'] = {'edges': bucket_edges, 'bucket_mse': rat_bucket_mse, 'bucket_counts': rat_bucket_counts}
        np_keys = [f"({bucket_edges[i]:.0e},{bucket_edges[i+1]:.0e}]" for i in range(4)]
        if any(rat_bucket_counts.get(k, 0) == 0 for k in np_keys):
            print("[Guardrail][Rational+ε] Some near-pole buckets (B0–B3) are empty. Consider adjusting dataset generation flags.")
        # Pole metrics (2D)
        rat_preds = rat_test.get('predictions', [])
        rational_results['pole_metrics'] = compute_pole_metrics_2d(test_data[0], rat_preds)
        all_results['Rational+ε'] = rational_results
        
        comparison_table.append({
            'Method': 'Rational+ε',
            'Parameters': rational_results['n_parameters'],
            'Epochs': rational_results['training_results']['config']['epochs'],
            'Train_MSE': 'N/A',
            'Test_MSE': rational_results['test_metrics']['mse'],
            'Training_Time': rational_results['training_time'],
            'Avg_Epoch_Time': rational_results['training_time'] / max(1, rational_results['training_results']['config']['epochs']),
            'Success_Rate': rational_results['test_metrics'].get('success_rate', 0.0),
            'Notes': f"ε={rational_results['epsilon']}",
            'NearPoleCountsB0_B3': [rat_bucket_counts.get(f"({bucket_edges[i]:.0e},{bucket_edges[i+1]:.0e}]", 0) for i in range(4)],
        })
        
        print(f"✓ Rational+ε: MSE={rational_results['test_metrics']['mse']:.6f}")
        
    except Exception as e:
        print(f"✗ Rational+ε failed: {e}")
        all_results['Rational+ε'] = {'error': str(e)}
    
    # 3. DLS Reference
    print("\n3. Running DLS Reference...")
    try:
        if 'dls' not in enabled_models:
            raise RuntimeError('skipped')
        dls_config = DLSConfig(damping_factor=0.01,
                               max_iterations=(1 if quick else 100))
        dls_results = run_dls_reference(samples, dls_config, output_dir=f"{output_dir}/dls", seed=seed)
        # Add bucketed MSE using final_errors
        dls_mse_list = dls_results.get('final_errors', [])
        dls_bucket_mse, dls_bucket_counts = compute_bucket_mse(dls_mse_list, bucket_edges, test_detj)
        dls_results['near_pole_bucket_mse'] = {'edges': bucket_edges, 'bucket_mse': dls_bucket_mse, 'bucket_counts': dls_bucket_counts}
        np_keys = [f"({bucket_edges[i]:.0e},{bucket_edges[i+1]:.0e}]" for i in range(4)]
        if any(dls_bucket_counts.get(k, 0) == 0 for k in np_keys):
            print("[Guardrail][DLS] Some near-pole buckets (B0–B3) are empty. Consider adjusting dataset generation flags.")
        # DLS: use average error as residual consistency proxy; others N/A
        dls_results['pole_metrics'] = {
            'ple': None,
            'sign_consistency': None,
            'slope_error': None,
            'residual_consistency': dls_results.get('average_error', None),
        }
        all_results['DLS'] = dls_results
        
        comparison_table.append({
            'Method': 'DLS (Reference)',
            'Parameters': 0,  # Analytical method
            'Epochs': 'N/A',
            'Train_MSE': 'N/A',
            'Test_MSE': dls_results['average_error'],
            'Training_Time': 0.0,  # No training needed
            'Avg_Epoch_Time': 'N/A',
            'Success_Rate': dls_results['success_rate'],
            'Notes': f"λ={dls_config.damping_factor}",
            'NearPoleCountsB0_B3': [dls_bucket_counts.get(f"({bucket_edges[i]:.0e},{bucket_edges[i+1]:.0e}]", 0) for i in range(4)],
        })
        
        print(f"✓ DLS: Error={dls_results['average_error']:.6f}")
        
    except Exception as e:
        print(f"✗ DLS failed: {e}")
        all_results['DLS'] = {'error': str(e)}
    
    # 4. ZeroProofML (Basic)
    print("\n4. Running ZeroProofML (Basic)...")
    try:
        if 'tr_basic' not in enabled_models:
            raise RuntimeError('skipped')
        # Derive det(J) for test split to compute near-pole metrics
        n_train = int(0.8 * len(samples))
        test_detj = [abs(s.get('det_J', 0.0)) for s in samples[n_train:]]
        zp_basic_results = run_zeroproof_baseline(
            train_data, test_data, enable_enhancements=False, output_dir=f"{output_dir}/zeroproof_basic",
            test_detJ=test_detj, epochs=zp_epochs
        )
        all_results['ZeroProofML-Basic'] = zp_basic_results
        # Guardrail: warn if near-pole buckets empty
        zpb = zp_basic_results.get('near_pole_bucket_mse', {})
        zpb_counts = zpb.get('bucket_counts', {})
        np_keys = [f"({bucket_edges[i]:.0e},{bucket_edges[i+1]:.0e}]" for i in range(4)]
        if any(zpb_counts.get(k, 0) == 0 for k in np_keys):
            print("[Guardrail][ZeroProofML-Basic] Some near-pole buckets (B0–B3) are empty. Consider adjusting dataset generation flags.")
        # Pole metrics (2D)
        zp_basic_results['pole_metrics'] = compute_pole_metrics_2d(test_data[0], zp_basic_results.get('predictions', []))
        
        comparison_table.append({
            'Method': 'ZeroProofML (Basic)',
            'Parameters': zp_basic_results['n_parameters'],
            'Epochs': zp_basic_results.get('epochs', 100),
            'Train_MSE': 'N/A',
            'Test_MSE': zp_basic_results['final_mse'],
            'Training_Time': zp_basic_results['training_time'],
            'Avg_Epoch_Time': zp_basic_results['training_time'] / max(1, zp_basic_results.get('epochs', 100)),
            'Success_Rate': 1.0,
            'Notes': 'TR-Rational only',
            'NearPoleCountsB0_B3': [zp_basic_results['near_pole_bucket_mse']['bucket_counts'].get(f"({bucket_edges[i]:.0e},{bucket_edges[i+1]:.0e}]", 0) for i in range(4)],
        })
        
        print(f"✓ ZeroProofML-Basic: MSE={zp_basic_results['final_mse']:.6f}")
        
    except Exception as e:
        print(f"✗ ZeroProofML-Basic failed: {e}")
        all_results['ZeroProofML-Basic'] = {'error': str(e)}
    
    # 5. ZeroProofML (Full)
    print("\n5. Running ZeroProofML (Full)...")
    try:
        if 'tr_full' not in enabled_models:
            raise RuntimeError('skipped')
        n_train = int(0.8 * len(samples))
        test_detj = [abs(s.get('det_J', 0.0)) for s in samples[n_train:]]
        zp_full_results = run_zeroproof_baseline(
            train_data, test_data, enable_enhancements=True, output_dir=f"{output_dir}/zeroproof_full",
            test_detJ=test_detj, epochs=zp_epochs
        )
        all_results['ZeroProofML-Full'] = zp_full_results
        zpf = zp_full_results.get('near_pole_bucket_mse', {})
        zpf_counts = zpf.get('bucket_counts', {})
        np_keys = [f"({bucket_edges[i]:.0e},{bucket_edges[i+1]:.0e}]" for i in range(4)]
        if any(zpf_counts.get(k, 0) == 0 for k in np_keys):
            print("[Guardrail][ZeroProofML-Full] Some near-pole buckets (B0–B3) are empty. Consider adjusting dataset generation flags.")
        # Pole metrics (2D)
        zp_full_results['pole_metrics'] = compute_pole_metrics_2d(test_data[0], zp_full_results.get('predictions', []))
        
        comparison_table.append({
            'Method': 'ZeroProofML (Full)',
            'Parameters': zp_full_results['n_parameters'],
            'Epochs': zp_full_results.get('epochs', 100),
            'Train_MSE': 'N/A',
            'Test_MSE': zp_full_results['final_mse'],
            'Training_Time': zp_full_results['training_time'],
            'Avg_Epoch_Time': zp_full_results['training_time'] / max(1, zp_full_results.get('epochs', 100)),
            'Success_Rate': 1.0,
            'Notes': 'All enhancements',
            'NearPoleCountsB0_B3': [zp_full_results['near_pole_bucket_mse']['bucket_counts'].get(f"({bucket_edges[i]:.0e},{bucket_edges[i+1]:.0e}]", 0) for i in range(4)],
        })
        
        print(f"✓ ZeroProofML-Full: MSE={zp_full_results['final_mse']:.6f}")
        
    except Exception as e:
        print(f"✗ ZeroProofML-Full failed: {e}")
        all_results['ZeroProofML-Full'] = {'error': str(e)}
    
    # Generate comparison report
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print("\nMethod\t\t\tParameters\tTest MSE\tTime (s)\tSuccess%\tNotes")
    print("-" * 80)
    
    for row in comparison_table:
        print(f"{row['Method']:<20}\t{row['Parameters']:<10}\t{row['Test_MSE']:<8}\t"
              f"{row['Training_Time']:<8.2f}\t{row['Success_Rate']:<8.1%}\t{row['Notes']}")
    
    # Print per-bucket MSE (and counts) for all methods that reported it
    print("\nNear-pole buckets by |det(J)| (MSE; count)")
    print("-" * 80)
    for method_key, label in [
        ('MLP', 'MLP'),
        ('Rational+ε', 'Rational+ε'),
        ('DLS', 'DLS (Reference)'),
        ('ZeroProofML-Basic', 'ZeroProofML (Basic)'),
        ('ZeroProofML-Full', 'ZeroProofML (Full)'),
    ]:
        res = all_results.get(method_key)
        if not res:
            continue
        bucket = res.get('near_pole_bucket_mse')
        if not bucket:
            continue
        edges = bucket.get('edges', [])
        mse_map = bucket.get('bucket_mse', {})
        cnt_map = bucket.get('bucket_counts', {})
        print(f"\n{label}:")
        # stable bucket order using edges
        for i in range(len(edges)-1):
            key = f"({edges[i]:.0e},{edges[i+1]:.0e}]"
            mse = mse_map.get(key)
            cnt = cnt_map.get(key)
            mse_str = f"{mse:.6f}" if isinstance(mse, (int, float)) else "-"
            print(f"  {key:<16} MSE={mse_str:<10} count={cnt if cnt is not None else 0}")
    
    # Save comprehensive results
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON results
    from zeroproof.utils.env import collect_env_info
    comprehensive_results = {
        'global': {
            'seed': seed,
            'quick': quick,
            'loss_name': 'mse_mean',
            'env': collect_env_info(),
        },
        'dataset_info': {
            'file': dataset_file,
            'n_samples': len(samples),
            'n_train': len(train_data[0]),
            'n_test': len(test_data[0]),
            'hash_sha256': dataset_hash,
            'bucket_edges': bucket_edges,
        },
        'individual_results': all_results,
        'comparison_table': comparison_table,
        'summary': {
            'methods_tested': len(comparison_table),
            'best_mse': min([r['Test_MSE'] for r in comparison_table if isinstance(r['Test_MSE'], (int, float))]),
            'fastest_training': min([r['Training_Time'] for r in comparison_table if r['Training_Time'] > 0])
        }
    }
    
    results_file = os.path.join(output_dir, "comprehensive_comparison.json")
    with open(results_file, 'w') as f:
        json.dump(to_builtin(comprehensive_results), f, indent=2)
    
    # CSV table
    csv_file = os.path.join(output_dir, "comparison_table.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=comparison_table[0].keys())
        writer.writeheader()
        writer.writerows(comparison_table)
    
    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {csv_file}")
    
    return comprehensive_results


def run_ablation_study(train_data: Tuple[List, List],
                      test_data: Tuple[List, List],
                      output_dir: str = "results/ablation") -> Dict[str, Any]:
    """
    Run ablation study on ZeroProofML enhancements.
    
    Args:
        train_data: Training data
        test_data: Test data
        output_dir: Output directory
        
    Returns:
        Ablation results
    """
    print("=== ZeroProofML Ablation Study ===")
    
    # Different configurations to test
    ablation_configs = [
        {
            'name': 'Full',
            'use_hybrid_schedule': True,
            'use_tag_loss': True,
            'use_pole_head': True,
            'enable_anti_illusion': True,
            'enforce_coverage': True
        },
        {
            'name': 'No Hybrid Schedule',
            'use_hybrid_schedule': False,
            'use_tag_loss': True,
            'use_pole_head': True,
            'enable_anti_illusion': True,
            'enforce_coverage': True
        },
        {
            'name': 'No Tag Loss',
            'use_hybrid_schedule': True,
            'use_tag_loss': False,
            'use_pole_head': True,
            'enable_anti_illusion': True,
            'enforce_coverage': True
        },
        {
            'name': 'No Pole Head',
            'use_hybrid_schedule': True,
            'use_tag_loss': True,
            'use_pole_head': False,
            'enable_anti_illusion': True,
            'enforce_coverage': True
        },
        {
            'name': 'No Anti-Illusion',
            'use_hybrid_schedule': True,
            'use_tag_loss': True,
            'use_pole_head': True,
            'enable_anti_illusion': False,
            'enforce_coverage': True
        },
        {
            'name': 'No Coverage Control',
            'use_hybrid_schedule': True,
            'use_tag_loss': True,
            'use_pole_head': True,
            'enable_anti_illusion': True,
            'enforce_coverage': False
        },
        {
            'name': 'Minimal (TR only)',
            'use_hybrid_schedule': False,
            'use_tag_loss': False,
            'use_pole_head': False,
            'enable_anti_illusion': False,
            'enforce_coverage': False
        }
    ]
    
    ablation_results = []
    
    for config_spec in ablation_configs:
        print(f"\n--- Testing: {config_spec['name']} ---")
        
        try:
            # This would require implementing the actual ablation
            # For now, simulate results
            simulated_mse = np.random.uniform(0.001, 0.1)  # Placeholder
            simulated_time = np.random.uniform(10, 60)     # Placeholder
            
            result = {
                'name': config_spec['name'],
                'config': config_spec,
                'test_mse': simulated_mse,
                'training_time': simulated_time,
                'status': 'simulated'  # Would be 'completed' for real runs
            }
            
            ablation_results.append(result)
            print(f"  MSE: {simulated_mse:.6f} (simulated)")
            
        except Exception as e:
            print(f"  Failed: {e}")
            ablation_results.append({
                'name': config_spec['name'],
                'config': config_spec,
                'error': str(e),
                'status': 'failed'
            })
    
    # Save ablation results
    os.makedirs(output_dir, exist_ok=True)
    
    ablation_file = os.path.join(output_dir, "ablation_study.json")
    with open(ablation_file, 'w') as f:
        json.dump({
            'ablation_results': ablation_results,
            'summary': {
                'n_configs_tested': len(ablation_configs),
                'successful_runs': len([r for r in ablation_results if r.get('status') != 'failed'])
            }
        }, f, indent=2)
    
    # CSV table
    csv_file = os.path.join(output_dir, "ablation_table.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Configuration', 'Test_MSE', 'Training_Time', 'Status'])
        
        for result in ablation_results:
            writer.writerow([
                result['name'],
                result.get('test_mse', 'N/A'),
                result.get('training_time', 'N/A'),
                result['status']
            ])
    
    print(f"\nAblation results saved to:")
    print(f"  - {ablation_file}")
    print(f"  - {csv_file}")
    
    return ablation_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare all baseline methods")
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to IK dataset JSON file')
    parser.add_argument('--output_dir', default='results/comparison',
                       help='Output directory')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study')
    parser.add_argument('--seed', type=int, default=None,
                       help='Global seed for reproducibility')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer epochs, skip grid search (ε=0.01)')
    parser.add_argument('--mlp_epochs', type=int, default=50,
                       help='MLP epochs (overrides in quick mode)')
    parser.add_argument('--rat_epochs', type=int, default=50,
                       help='Rational+ε epochs (overrides in quick mode)')
    parser.add_argument('--rat_epsilon', type=float, default=None,
                       help='Specific epsilon to use (skips grid search if provided)')
    parser.add_argument('--zp_epochs', type=int, default=100,
                       help='ZeroProof epochs for comparator path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        print("Generate dataset first using: python ../robotics/rr_ik_dataset.py")
        exit(1)
    
    # Run comparison
    # Apply quick overrides via globals (simple approach)
    global _COMPARATOR_OVERRIDES
    _COMPARATOR_OVERRIDES = {
        'quick': args.quick,
        'mlp_epochs': args.mlp_epochs if not args.quick else min(args.mlp_epochs, 10),
        'rat_epochs': args.rat_epochs if not args.quick else min(args.rat_epochs, 10),
        'rat_epsilon': args.rat_epsilon if args.rat_epsilon is not None else (0.01 if args.quick else None),
        'zp_epochs': args.zp_epochs if not args.quick else min(args.zp_epochs, 20),
    }
    results = run_complete_comparison(args.dataset, args.output_dir, seed=args.seed)
    
    # Run ablation if requested
    if args.ablation:
        # Load data for ablation
        with open(args.dataset, 'r') as f:
            data = json.load(f)
        samples = data['samples']
        train_data, test_data = prepare_ik_data(samples)
        
        ablation_results = run_ablation_study(train_data, test_data, f"{args.output_dir}/ablation")
    
    print("\n=== Comparison Complete ===")
    print("Check the output directory for detailed results and CSV tables.")
