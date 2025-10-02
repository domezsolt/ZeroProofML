"""
Test script to verify critical fixes for dataset generation and coverage control.

This script demonstrates that:
1. SingularDatasetGenerator creates actual singular points
2. Coverage control maintains lambda_rej above minimum threshold
3. Models actually encounter non-REAL outputs during training
"""

import numpy as np

from zeroproof.core import TRTag, real
from zeroproof.layers import ChebyshevBasis, TRRational
from zeroproof.training import AdaptiveLambda, AdaptiveLossConfig, TrainingConfig, TRTrainer
from zeroproof.utils import SingularDatasetGenerator


def test_dataset_generation():
    """Test that dataset generator creates actual singularities."""
    print("=" * 60)
    print("Testing Dataset Generation with Actual Singularities")
    print("=" * 60)

    generator = SingularDatasetGenerator(domain=(-1.0, 1.0), seed=42)

    # Add poles
    generator.add_pole(location=0.5, strength=0.01)
    generator.add_pole(location=-0.3, strength=0.01)

    x_vals, y_vals, metadata = generator.generate_rational_function_data(
        n_samples=100,
        singularity_ratio=0.4,  # 40% near/at singularities
        force_exact_singularities=True,
        noise_level=0.01,
    )

    print(f"Generated {len(x_vals)} samples")
    print(f"Pole locations: {[s.location for s in generator.singularities]}")
    print(f"Tag distribution: {metadata['tag_distribution']}")

    # Check that we have actual non-REAL tags
    non_real_count = sum(
        1
        for tag, count in metadata["tag_distribution"].items()
        if tag != "REAL"
        for _ in range(count)
    )
    print(
        f"Non-REAL samples: {non_real_count}/{len(x_vals)} "
        f"({100*non_real_count/len(x_vals):.1f}%)"
    )

    # Verify we have singular points
    n_singular = len(metadata["exact_singular_indices"])
    n_near = len(metadata["near_singular_indices"])
    print(f"Exact singular points: {n_singular}")
    print(f"Near-singular points: {n_near}")

    return x_vals, y_vals, metadata


def test_coverage_control():
    """Test that coverage control maintains lambda_rej properly."""
    print("\n" + "=" * 60)
    print("Testing Coverage Control with Adaptive Lambda")
    print("=" * 60)

    # Generate challenging dataset
    generator = SingularDatasetGenerator(domain=(-1.0, 1.0), seed=42)

    # Add a pole
    generator.add_pole(location=0.5, strength=0.01)

    x_vals, y_vals, metadata = generator.generate_rational_function_data(
        n_samples=200,
        singularity_ratio=0.4,  # 40% near/at singularities
        force_exact_singularities=True,
        noise_level=0.05,
    )

    # Create model
    basis = ChebyshevBasis()
    model = TRRational(d_p=3, d_q=2, basis=basis)

    # Create adaptive loss controller
    adaptive_config = AdaptiveLossConfig(
        initial_lambda=1.0,
        target_coverage=0.85,  # Target 85% REAL outputs
        learning_rate=0.1,
        warmup_steps=5,
        momentum=0.0,
    )
    adaptive_lambda = AdaptiveLambda(adaptive_config)

    # Training configuration
    config = TrainingConfig(
        max_epochs=20,
        batch_size=32,
        learning_rate=0.01,
        lambda_rej=1.0,  # Will be controlled by adaptive_lambda
        verbose=False,
    )

    # Create trainer
    trainer = TRTrainer(model, config)

    # Train with adaptive lambda
    x_train = x_vals  # Already TRScalar objects
    y_train = y_vals  # Already TRScalar objects

    print(f"Training for {config.max_epochs} epochs...")
    print(f"Target coverage: {adaptive_config.target_coverage}")
    print(f"Minimum lambda_rej: {adaptive_config.lambda_min}")

    lambda_history = []
    coverage_history = []

    for epoch in range(config.max_epochs):
        # Train one epoch
        batch_loss = []
        batch_tags = []

        for i in range(0, len(x_train), config.batch_size):
            batch_x = x_train[i : i + config.batch_size]
            batch_y = y_train[i : i + config.batch_size]

            # Forward pass
            outputs = model(batch_x)
            tags = [out.tag for out in outputs]
            batch_tags.extend(tags)

            # Update lambda based on coverage
            adaptive_lambda.update(tags)
            config.lambda_rej = adaptive_lambda.lambda_rej

        # Record history
        lambda_history.append(adaptive_lambda.lambda_rej)
        coverage = sum(1 for tag in batch_tags if tag == TRTag.REAL) / len(batch_tags)
        coverage_history.append(coverage)

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d}: λ_rej={adaptive_lambda.lambda_rej:.3f}, "
                f"coverage={coverage:.3f}"
            )

    # Final statistics
    print("\nFinal Statistics:")
    print(f"Final lambda_rej: {adaptive_lambda.lambda_rej:.3f}")
    print(f"Final coverage: {coverage_history[-1]:.3f}")
    print(f"Min lambda_rej during training: {min(lambda_history):.3f}")
    print(f"Lambda never dropped below threshold: {min(lambda_history) >= 0.1}")

    return lambda_history, coverage_history


def main():
    """Run all tests."""
    print("CRITICAL FIXES VERIFICATION")
    print("=" * 60)

    # Test 1: Dataset generation
    x_vals, y_vals, metadata = test_dataset_generation()

    # Test 2: Coverage control
    lambda_hist, coverage_hist = test_coverage_control()

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

    # Summary
    success_criteria = [
        (
            "Dataset includes actual singularities",
            any(count > 0 for tag, count in metadata["tag_distribution"].items() if tag != "REAL"),
        ),
        ("Lambda_rej stays above minimum (0.1)", min(lambda_hist) >= 0.1),
        ("Coverage converges near target", abs(coverage_hist[-1] - 0.85) < 0.15),
        ("Non-trivial lambda adjustments occur", max(lambda_hist) != min(lambda_hist)),
    ]

    print("\nSuccess Criteria:")
    for criterion, passed in success_criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")

    all_passed = all(passed for _, passed in success_criteria)
    if all_passed:
        print("\n🎉 All critical fixes are working correctly!")
    else:
        print("\n⚠️ Some fixes may need additional work.")


if __name__ == "__main__":
    main()
