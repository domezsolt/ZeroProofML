"""
Integration test for end-to-end synthetic rational regression.

Verifies that coverage adapts to target, λ_rej stabilizes, and
actual singularities are encountered during training.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass

from zeroproof.core import TRTag, real, pinf, ninf, phi
from zeroproof.layers import TRRational
from zeroproof.training import (
    HybridTRTrainer,
    HybridTrainingConfig,
    create_advanced_controller,
    create_integrated_sampler,
    CoverageTracker,
)
from zeroproof.utils import SingularDatasetGenerator


@dataclass
class RegressionTestConfig:
    """Configuration for regression test."""
    # Target function
    n_poles: int = 3
    pole_locations: List[float] = None
    
    # Training
    n_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # Coverage
    target_coverage: float = 0.85
    coverage_tolerance: float = 0.05
    
    # Convergence criteria
    loss_threshold: float = 0.1
    lambda_stability_threshold: float = 0.1
    min_non_real_ratio: float = 0.1  # At least 10% non-REAL
    
    def __post_init__(self):
        if self.pole_locations is None:
            # Default pole locations
            self.pole_locations = [-1.0, 0.0, 1.0]


class SyntheticRationalDataset:
    """
    Synthetic dataset with known rational function and poles.
    """
    
    def __init__(self, config: RegressionTestConfig):
        """
        Initialize synthetic dataset.
        
        Args:
            config: Test configuration
        """
        self.config = config
        
        # Generate data with actual singularities
        self.generator = SingularDatasetGenerator(
            domain=(-1.0, 1.0),
            seed=42
        )
        
        # Add poles from config
        for pole_loc in config.pole_locations:
            self.generator.add_pole(pole_loc, strength=0.01)
        
        # Generate training data
        x_train, y_train, _ = self.generator.generate_rational_function_data(
            n_samples=1000, 
            singularity_ratio=0.3,
            force_exact_singularities=True
        )
        # Convert to tensors, preserving infinities
        x_values = []
        y_values = []
        for x, y in zip(x_train, y_train):
            x_values.append(x.value)
            if y.tag == TRTag.REAL:
                y_values.append(y.value)
            elif y.tag == TRTag.PINF:
                y_values.append(float('inf'))
            elif y.tag == TRTag.NINF:
                y_values.append(float('-inf'))
            else:  # PHI
                y_values.append(float('nan'))
        self.x_train = torch.tensor(x_values, dtype=torch.float32)
        self.y_train = torch.tensor(y_values, dtype=torch.float32)
        
        # Generate test data
        x_test, y_test, _ = self.generator.generate_rational_function_data(
            n_samples=200,
            singularity_ratio=0.3,
            force_exact_singularities=True
        )
        # Convert test data to tensors, preserving infinities
        x_test_values = []
        y_test_values = []
        for x, y in zip(x_test, y_test):
            x_test_values.append(x.value)
            if y.tag == TRTag.REAL:
                y_test_values.append(y.value)
            elif y.tag == TRTag.PINF:
                y_test_values.append(float('inf'))
            elif y.tag == TRTag.NINF:
                y_test_values.append(float('-inf'))
            else:  # PHI
                y_test_values.append(float('nan'))
        self.x_test = torch.tensor(x_test_values, dtype=torch.float32)
        self.y_test = torch.tensor(y_test_values, dtype=torch.float32)
    
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get random batch from training data."""
        indices = np.random.choice(len(self.x_train), batch_size, replace=False)
        return self.x_train[indices], self.y_train[indices]


class TestSyntheticRationalRegression:
    """Integration test for synthetic rational regression."""
    
    def test_end_to_end_training(self):
        """Test complete training pipeline with all components."""
        # Configuration
        config = RegressionTestConfig(
            n_epochs=50,
            target_coverage=0.85,
            n_poles=3,
        )
        
        # Create dataset
        dataset = SyntheticRationalDataset(config)
        
        # Create model
        model = TRRational(
            d_p=4,
            d_q=3,
            lambda_rej=1.0,
            projection_index=0  # tests pass [[x]] vectors in some places
        )
        
        # Create advanced controller
        controller = create_advanced_controller(
            control_type="hybrid",
            target_coverage=config.target_coverage,
            pole_locations=config.pole_locations,
            kp=1.0,
            ki=0.1,
            dead_band=0.02,
        )
        
        # Create integrated sampler
        sampler = create_integrated_sampler(
            strategy="hybrid",
            weight_power=2.0,
            export_path="test_diagnostics",
        )
        
        # Training configuration
        train_config = HybridTrainingConfig(
            max_epochs=config.n_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            target_coverage=config.target_coverage,
            use_hybrid_gradient=True,
            use_enhanced_coverage=True,
        )
        
        # Create trainer
        trainer = HybridTRTrainer(train_config)
        
        # Tracking metrics
        coverage_history = []
        lambda_history = []
        loss_history = []
        non_real_counts = []
        
        # Training loop
        for epoch in range(config.n_epochs):
            epoch_loss = 0.0
            epoch_tags = []
            batch_q_values = []
            
            # Train epoch
            for _ in range(10):  # 10 batches per epoch
                # Get batch with importance sampling
                x_batch, y_batch = dataset.get_batch(config.batch_size)
                
                # Forward pass
                y_pred_nodes = model.forward_batch(x_batch.tolist())
                y_pred = y_pred_nodes
                
                # Track Q values for sampling
                batch_q_values.extend(model.get_q_values(x_batch))
                
                # Compute loss
                loss = trainer.compute_loss(y_pred, y_batch)
                epoch_loss += loss.item()
                
                # Track tags
                for pred in y_pred:
                    epoch_tags.append(pred.tag)
                
                # Backward pass
                loss.backward()
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
            
            # Compute epoch metrics
            coverage = sum(1 for t in epoch_tags if t == TRTag.REAL) / len(epoch_tags)
            n_non_real = sum(1 for t in epoch_tags if t != TRTag.REAL)
            
            # Update controller
            control_result = controller.update(
                epoch=epoch,
                coverage=coverage,
                loss=epoch_loss / 10,
            )
            
            # Update model's lambda_rej
            if 'lambda_rej' in control_result:
                model.lambda_rej = control_result['lambda_rej']
            
            # Update sampler diagnostics
            sampler.update_diagnostics(
                epoch=epoch,
                metrics={'coverage_train': coverage, 'loss_train': epoch_loss / 10},
                q_values=torch.tensor(batch_q_values) if batch_q_values else None,
                tags=epoch_tags,
            )
            
            # Track history
            coverage_history.append(coverage)
            lambda_history.append(model.lambda_rej)
            loss_history.append(epoch_loss / 10)
            non_real_counts.append(n_non_real)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Coverage={coverage:.3f}, "
                      f"λ_rej={model.lambda_rej:.3f}, "
                      f"Loss={epoch_loss/10:.4f}, "
                      f"Non-REAL={n_non_real}")
        
        # Verification 1: Coverage adapts to target
        final_coverage = np.mean(coverage_history[-10:])
        assert abs(final_coverage - config.target_coverage) <= config.coverage_tolerance, \
            f"Coverage {final_coverage:.3f} not within tolerance of target {config.target_coverage}"
        
        # Verification 2: λ_rej stabilizes
        lambda_std = np.std(lambda_history[-10:])
        assert lambda_std <= config.lambda_stability_threshold, \
            f"λ_rej not stable: std={lambda_std:.3f}"
        
        # Verification 3: Coverage < 100% (singularities encountered)
        assert all(c < 1.0 for c in coverage_history[-10:]), \
            "Coverage reached 100% - no singularities encountered!"
        
        # Verification 4: Actual non-REAL outputs produced
        total_non_real = sum(non_real_counts)
        total_samples = len(non_real_counts) * config.batch_size * 10
        non_real_ratio = total_non_real / total_samples
        assert non_real_ratio >= config.min_non_real_ratio, \
            f"Not enough non-REAL outputs: {non_real_ratio:.2%} < {config.min_non_real_ratio:.2%}"
        
        print("\n✓ All integration tests passed!")
        print(f"  Final coverage: {final_coverage:.3f}")
        print(f"  λ_rej stability: {lambda_std:.3f}")
        print(f"  Non-REAL ratio: {non_real_ratio:.2%}")
    
    def test_coverage_control_effectiveness(self):
        """Test that coverage control achieves different targets."""
        target_coverages = [0.70, 0.85, 0.95]
        
        for target in target_coverages:
            print(f"\nTesting target coverage {target:.2f}")
            
            # Simple configuration
            config = RegressionTestConfig(
                n_epochs=30,
                target_coverage=target,
            )
            
            # Create dataset and model
            dataset = SyntheticRationalDataset(config)
            model = TRRational(d_p=3, d_q=2, projection_index=0)
            
            # Create controller
            controller = create_advanced_controller(
                control_type="pi",
                target_coverage=target,
                kp=2.0,  # Higher gain for faster convergence
                ki=0.2,
            )
            
            # Track coverage
            coverages = []
            
            # Simple training loop
            for epoch in range(config.n_epochs):
                # Get batch
                x_batch, y_batch = dataset.get_batch(32)
                
                # Forward pass (batched)
                y_pred = model.forward_batch(x_batch.tolist())
                
                # Compute coverage
                tags = [pred.tag for pred in y_pred]
                coverage = sum(1 for t in tags if t == TRTag.REAL) / len(tags)
                coverages.append(coverage)
                
                # Update controller
                result = controller.update(epoch, coverage, 0.0)
                if 'lambda_rej' in result:
                    model.lambda_rej = result['lambda_rej']
            
            # Verify convergence to target
            final_coverage = np.mean(coverages[-10:])
            assert abs(final_coverage - target) <= 0.1, \
                f"Failed to achieve target {target}: got {final_coverage:.3f}"
            
            print(f"  ✓ Achieved coverage {final_coverage:.3f}")
    
    def test_gradient_stability_near_poles(self):
        """Test that gradients remain stable near poles."""
        config = RegressionTestConfig(n_epochs=20)
        dataset = SyntheticRationalDataset(config)
        
        # Model with hybrid gradient schedule
        model = TRRational(d_p=3, d_q=2, projection_index=0)
        
        # Track gradient magnitudes
        gradient_history = []
        
        # Test specifically near poles
        for pole in config.pole_locations:
            # Create inputs near pole
            near_pole_inputs = torch.tensor([
                [pole - 0.001],
                [pole],
                [pole + 0.001],
            ], requires_grad=True)
            
            # Forward pass
            outputs = model.forward_batch(near_pole_inputs.tolist())
            
            # Create dummy loss
            loss = sum(o.value if o.tag == TRTag.REAL else 0.0 for o in outputs)
            
            # Backward pass (should not explode)
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                
                # Check gradient magnitude
                if near_pole_inputs.grad is not None:
                    grad_norm = torch.norm(near_pole_inputs.grad)
                    gradient_history.append(grad_norm.item())
                    
                    # Gradient should be bounded
                    assert grad_norm < 1000.0, \
                        f"Gradient explosion near pole {pole}: {grad_norm:.2e}"
        
        if gradient_history:
            print(f"\n✓ Gradients stable near poles")
            print(f"  Max gradient norm: {max(gradient_history):.2f}")
            print(f"  Mean gradient norm: {np.mean(gradient_history):.2f}")


class TestDatasetQuality:
    """Test that datasets contain actual singularities."""
    
    def test_dataset_has_singularities(self):
        """Verify dataset includes singular points."""
        config = RegressionTestConfig()
        dataset = SyntheticRationalDataset(config)
        
        # Check training data for singularities
        non_real_count = 0
        for x, y in zip(dataset.x_train, dataset.y_train):
            # Evaluate at x
            if isinstance(y, (float, int)):
                if np.isinf(y) or np.isnan(y):
                    non_real_count += 1
            elif hasattr(y, 'tag'):
                if y.tag != TRTag.REAL:
                    non_real_count += 1
        
        # Should have some singular points
        singularity_ratio = non_real_count / len(dataset.x_train)
        assert singularity_ratio > 0.05, \
            f"Dataset has too few singularities: {singularity_ratio:.2%}"
        
        print(f"\n✓ Dataset quality verified")
        print(f"  Singularity ratio: {singularity_ratio:.2%}")
        print(f"  Total singular points: {non_real_count}/{len(dataset.x_train)}")
    
    def test_near_pole_sampling(self):
        """Test that near-pole regions are adequately sampled."""
        config = RegressionTestConfig()
        dataset = SyntheticRationalDataset(config)
        
        # Check distribution near poles
        for pole in config.pole_locations:
            near_pole_count = 0
            threshold = 0.1
            
            for x in dataset.x_train:
                if isinstance(x, torch.Tensor):
                    x_val = x.item() if x.dim() == 0 else x[0].item()
                else:
                    x_val = x
                
                if abs(x_val - pole) < threshold:
                    near_pole_count += 1
            
            near_pole_ratio = near_pole_count / len(dataset.x_train)
            assert near_pole_ratio > 0.05, \
                f"Insufficient sampling near pole {pole}: {near_pole_ratio:.2%}"
            
            print(f"  Pole {pole:+.1f}: {near_pole_ratio:.2%} of samples within ±{threshold}")


class TestConvergenceMetrics:
    """Test convergence and stability metrics."""
    
    def test_loss_convergence(self):
        """Test that loss decreases and converges."""
        config = RegressionTestConfig(n_epochs=50)
        dataset = SyntheticRationalDataset(config)
        model = TRRational(d_p=3, d_q=2, projection_index=0)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        losses = []
        for epoch in range(config.n_epochs):
            epoch_loss = 0.0
            for _ in range(10):
                x_batch, y_batch = dataset.get_batch(32)
                y_pred = model(x_batch)
                
                # Simple MSE loss on REAL outputs
                loss = 0.0
                count = 0
                for pred, target in zip(y_pred, y_batch):
                    if pred.tag == TRTag.REAL:
                        loss += (pred.value - target) ** 2
                        count += 1
                
                if count > 0:
                    loss = loss / count
                    optimizer.zero_grad()
                    if isinstance(loss, torch.Tensor):
                        loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() if hasattr(loss, 'item') else loss
            
            losses.append(epoch_loss / 10)
        
        # Check convergence
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        
        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        
        # Check stability
        final_std = np.std(losses[-10:])
        assert final_std < 0.1, \
            f"Loss not stable: std={final_std:.4f}"
        
        print(f"\n✓ Loss convergence verified")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Improvement: {(initial_loss - final_loss) / initial_loss:.1%}")


if __name__ == "__main__":
    # Run all tests
    print("=" * 60)
    print("Synthetic Rational Regression Integration Tests")
    print("=" * 60)
    
    # Test dataset quality first
    dataset_test = TestDatasetQuality()
    dataset_test.test_dataset_has_singularities()
    dataset_test.test_near_pole_sampling()
    
    # Test convergence
    convergence_test = TestConvergenceMetrics()
    convergence_test.test_loss_convergence()
    
    # Main integration test
    main_test = TestSyntheticRationalRegression()
    main_test.test_coverage_control_effectiveness()
    main_test.test_gradient_stability_near_poles()
    main_test.test_end_to_end_training()
    
    print("\n" + "=" * 60)
    print("All Integration Tests Passed! ✓")
    print("=" * 60)
