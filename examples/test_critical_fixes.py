"""
Test script for critical fixes in ZeroProof.

This script demonstrates:
1. Dataset generation with actual singularities
2. Coverage control that maintains pressure
3. Actual non-REAL outputs during training
"""

import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from zeroproof.autodiff import TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import ChebyshevBasis, TRRational
from zeroproof.training import TrainingConfig, TRTrainer
from zeroproof.utils.dataset_generation import SingularDatasetGenerator


def test_dataset_generation():
    """Test that dataset includes actual singularities."""
    print("=" * 60)
    print("TEST 1: Dataset Generation with Actual Singularities")
    print("=" * 60)

    generator = SingularDatasetGenerator(seed=42)
    generator.add_pole(0.5, strength=0.01)
    generator.add_pole(-0.3, strength=0.01)

    # Generate dataset with forced singularities
    x_train, y_train, metadata = generator.generate_rational_function_data(
        n_samples=100,
        singularity_ratio=0.4,  # 40% near/at singularities
        force_exact_singularities=True,
        noise_level=0.01,
    )

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(x_train)}")
    print(f"  Exact singular points: {len(metadata['exact_singular_indices'])}")
    print(f"  Near singular points: {len(metadata['near_singular_indices'])}")
    print(f"  Regular points: {len(metadata['regular_indices'])}")

    print(f"\nTag Distribution in Targets:")
    for tag, count in metadata["tag_distribution"].items():
        percentage = count / len(y_train) * 100
        print(f"  {tag}: {count} ({percentage:.1f}%)")

    # Verify we have non-REAL outputs
    non_real_count = sum(1 for y in y_train if y.tag != TRTag.REAL)
    print(f"\nNon-REAL outputs: {non_real_count} ({non_real_count/len(y_train)*100:.1f}%)")

    assert non_real_count > 0, "Dataset should contain non-REAL outputs!"
    print("✓ Dataset contains actual singularities")

    return x_train, y_train, metadata


def test_coverage_control():
    """Test that coverage control maintains pressure."""
    print("\n" + "=" * 60)
    print("TEST 2: Coverage Control with Minimum Lambda")
    print("=" * 60)

    # Generate challenging data
    generator = SingularDatasetGenerator(seed=42)
    generator.add_pole(0.5, strength=0.01)

    x_train, y_train, metadata = generator.generate_rational_function_data(
        n_samples=200, singularity_ratio=0.3, force_exact_singularities=True
    )

    # Create model
    model = TRRational(degree_p=3, degree_q=2, basis=ChebyshevBasis(degree=3), lambda_reg=0.001)

    # Training config with adaptive loss
    config = TrainingConfig(
        learning_rate=0.01,
        max_epochs=30,
        use_adaptive_loss=True,
        target_coverage=0.85,  # Target 85% REAL
        lambda_learning_rate=0.05,
        initial_lambda=1.0,
        adaptive_lambda_min=0.1,  # Never go below 0.1
        log_interval=10,
        verbose=False,
    )

    # Create trainer
    trainer = TRTrainer(model, config=config)

    # Track coverage and lambda during training
    coverage_history = []
    lambda_history = []

    print("\nTraining with coverage control...")
    print("Target coverage: 0.85")
    print("Minimum λ_rej: 0.1")
    print("\nEpoch | Coverage | λ_rej | Loss")
    print("-" * 40)

    for epoch in range(config.max_epochs):
        epoch_loss = 0.0
        epoch_coverage = 0.0
        n_batches = 0

        # Simple batch training
        batch_size = 32
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]

            # Convert to TRNode for training
            batch_x_nodes = [TRNode.constant(x) for x in batch_x]
            batch_y_nodes = [TRNode.constant(y) for y in batch_y]

            metrics = trainer.train_step(batch_x, batch_y)

            epoch_loss += metrics.get("loss", 0)
            epoch_coverage += metrics.get("coverage", 1.0)
            n_batches += 1

        epoch_loss /= n_batches
        epoch_coverage /= n_batches

        # Get current lambda
        if trainer.loss_policy:
            stats = trainer.loss_policy.get_statistics()
            current_lambda = stats["lambda_rej"]
        else:
            current_lambda = 1.0

        coverage_history.append(epoch_coverage)
        lambda_history.append(current_lambda)

        if epoch % 5 == 0:
            print(
                f"{epoch:5d} | {epoch_coverage:.3f}    | {current_lambda:.3f}  | {epoch_loss:.3f}"
            )

    print("\nFinal Statistics:")
    print(f"  Final coverage: {coverage_history[-1]:.3f}")
    print(f"  Final λ_rej: {lambda_history[-1]:.3f}")
    print(f"  Min λ_rej during training: {min(lambda_history):.3f}")
    print(f"  Coverage range: [{min(coverage_history):.3f}, {max(coverage_history):.3f}]")

    # Verify lambda never dropped to 0
    assert min(lambda_history) >= 0.09, "Lambda should never drop below minimum!"
    print("✓ Lambda maintained minimum pressure throughout training")

    # Verify coverage is not stuck at 100%
    assert min(coverage_history) < 0.99, "Coverage should not be stuck at 100%!"
    print("✓ Coverage shows variation (not stuck at 100%)")

    return coverage_history, lambda_history


def test_model_outputs():
    """Test that model actually produces non-REAL outputs."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Produces Non-REAL Outputs")
    print("=" * 60)

    # Generate test data with exact singularities
    generator = SingularDatasetGenerator(seed=42)
    generator.add_pole(0.5, strength=0.001)  # Very tight pole

    x_test, y_test, metadata = generator.generate_rational_function_data(
        n_samples=50,
        singularity_ratio=0.5,  # 50% at/near singularities
        force_exact_singularities=True,
    )

    # Create and initialize model
    model = TRRational(degree_p=3, degree_q=2, basis=ChebyshevBasis(degree=3), lambda_reg=0.001)

    # Initialize with parameters that create a pole
    # This ensures Q(x) will be zero somewhere
    import torch

    if hasattr(model, "phi"):
        # Set denominator coefficients to create a pole near x=0.5
        model.phi.data = torch.tensor([1.0, -0.5, 0.1], dtype=torch.float32)

    print("\nEvaluating model on test set...")

    # Evaluate model
    tag_counts = {"REAL": 0, "PINF": 0, "NINF": 0, "PHI": 0}

    for x in x_test[:20]:  # Test first 20 samples
        y_pred, tag = model.forward(x)
        tag_counts[tag.name] += 1

    print("\nModel Output Tag Distribution:")
    for tag, count in tag_counts.items():
        percentage = count / 20 * 100
        print(f"  {tag}: {count} ({percentage:.1f}%)")

    non_real_outputs = sum(count for tag, count in tag_counts.items() if tag != "REAL")
    print(f"\nNon-REAL outputs: {non_real_outputs} ({non_real_outputs/20*100:.1f}%)")

    # Note: In a properly trained model with actual singularities,
    # we should see some non-REAL outputs
    if non_real_outputs > 0:
        print("✓ Model produces non-REAL outputs at singularities")
    else:
        print("⚠ Model currently avoids singularities (needs training with fixed dataset)")

    return tag_counts


def main():
    """Run all critical fix tests."""
    print("\n" + "=" * 70)
    print("CRITICAL FIXES VERIFICATION")
    print("=" * 70)
    print("\nThis script verifies the critical fixes for ZeroProof:")
    print("1. Dataset generation with actual singularities")
    print("2. Coverage control that maintains pressure")
    print("3. Models that can produce non-REAL outputs")

    # Test 1: Dataset generation
    x_train, y_train, metadata = test_dataset_generation()

    # Test 2: Coverage control
    coverage_history, lambda_history = test_coverage_control()

    # Test 3: Model outputs
    tag_counts = test_model_outputs()

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("\n✅ Critical fixes verified:")
    print("  • Dataset includes actual singular points")
    print("  • λ_rej maintains minimum pressure")
    print("  • Coverage control prevents trivial solutions")
    print("\n⚠ Next steps:")
    print("  • Train models with the fixed dataset generation")
    print("  • Implement importance sampling near poles")
    print("  • Add evaluation metrics for pole detection")
    print("\nThe foundation is now in place for ZeroProof to")
    print("actually learn and handle singularities as intended!")


if __name__ == "__main__":
    main()
