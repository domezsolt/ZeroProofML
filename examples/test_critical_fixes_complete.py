"""
Complete test of critical fixes with actual model training.

This test verifies that:
1. Dataset generation creates actual singularities
2. Models encounter non-REAL outputs during training
3. Coverage control maintains lambda_rej properly
"""

import numpy as np

from zeroproof.core import TRTag, real
from zeroproof.layers import ChebyshevBasis, TRRational
from zeroproof.training import Optimizer, TrainingConfig, TRTrainer
from zeroproof.utils import SingularDatasetGenerator


def main():
    """Run complete critical fixes test."""
    print("=" * 60)
    print("COMPLETE CRITICAL FIXES TEST")
    print("=" * 60)

    # 1. Generate dataset with actual singularities
    print("\n1. Generating dataset with actual singularities...")
    generator = SingularDatasetGenerator(domain=(-1.0, 1.0), seed=42)

    # Add poles
    generator.add_pole(0.5)
    generator.add_pole(-0.3)

    # Generate dataset with guaranteed singularities
    x_vals, y_vals, metadata = generator.generate_rational_function_data(
        n_samples=500,
        singularity_ratio=0.3,  # 30% near/at singularities
        force_exact_singularities=True,
        noise_level=0.01,
    )

    print(f"Generated {len(x_vals)} samples")
    print(f"Pole locations: {metadata['singularities']}")
    print(f"Tag distribution: {metadata['tag_distribution']}")

    # Count non-REAL samples from tag distribution
    tag_dist = metadata["tag_distribution"]
    non_real_count = sum(count for tag, count in tag_dist.items() if tag != "REAL")
    total_count = sum(tag_dist.values())
    print(
        f"Non-REAL samples: {non_real_count}/{total_count} ({100*non_real_count/total_count:.1f}%)"
    )

    # 2. Create and train model with adaptive loss
    print("\n2. Training model with adaptive loss control...")

    # Create model
    basis = ChebyshevBasis()
    model = TRRational(d_p=3, d_q=2, basis=basis)

    # Training configuration with adaptive loss
    config = TrainingConfig(
        max_epochs=30,
        batch_size=32,
        learning_rate=0.01,
        use_adaptive_loss=True,
        initial_lambda=1.0,
        target_coverage=0.85,
        lambda_learning_rate=0.1,
        adaptive_lambda_min=0.1,  # Never go below 0.1
        adaptive_warmup_steps=100,
        adaptive_momentum=0.0,
        verbose=False,
    )

    # Create optimizer
    optimizer = Optimizer(model.parameters(), learning_rate=config.learning_rate)

    # Create trainer with optimizer and config
    trainer = TRTrainer(model, optimizer, config)

    # Split data
    n_train = int(0.8 * len(x_vals))
    x_train, y_train = x_vals[:n_train], y_vals[:n_train]
    x_val, y_val = x_vals[n_train:], y_vals[n_train:]

    # Prepare data in batches for trainer
    batch_size = config.batch_size
    train_batches = []
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i : i + batch_size]
        batch_y = y_train[i : i + batch_size]
        train_batches.append((batch_x, batch_y))

    val_batches = []
    for i in range(0, len(x_val), batch_size):
        batch_x = x_val[i : i + batch_size]
        batch_y = y_val[i : i + batch_size]
        val_batches.append((batch_x, batch_y))

    # Train model
    print(f"Training for {config.max_epochs} epochs...")
    print(f"Target coverage: {config.target_coverage}")
    print(f"Minimum lambda_rej: {config.adaptive_lambda_min}")

    history = trainer.train(train_batches, val_batches)

    # 3. Analyze results
    print("\n3. Analyzing results...")

    # Check coverage history
    if "coverage" in history and len(history["coverage"]) > 0:
        final_coverage = history["coverage"][-1]
        avg_coverage = np.mean(history["coverage"])
        min_coverage = min(history["coverage"])
        print(f"Final coverage: {final_coverage:.3f}")
        print(f"Average coverage: {avg_coverage:.3f}")
        print(f"Minimum coverage: {min_coverage:.3f}")
    else:
        print("No coverage history available")

    # Check lambda history
    if "lambda_rej" in history and len(history["lambda_rej"]) > 0:
        final_lambda = history["lambda_rej"][-1]
        min_lambda = min(history["lambda_rej"])
        print(f"Final lambda_rej: {final_lambda:.3f}")
        print(f"Minimum lambda_rej: {min_lambda:.3f}")
        print(f"Lambda stayed above threshold: {min_lambda >= 0.1}")
    else:
        print("No lambda history available")

    # Test model on validation set
    print("\n4. Testing model on validation set...")
    val_outputs = []
    for x in x_val:
        output = model(x)
        val_outputs.append(output)
    val_tags = [out.tag for out in val_outputs]

    # Count tags
    tag_counts = {
        "REAL": sum(1 for tag in val_tags if tag == TRTag.REAL),
        "PINF": sum(1 for tag in val_tags if tag == TRTag.PINF),
        "NINF": sum(1 for tag in val_tags if tag == TRTag.NINF),
        "PHI": sum(1 for tag in val_tags if tag == TRTag.PHI),
    }

    print("Validation output tags:")
    for tag_name, count in tag_counts.items():
        pct = 100 * count / len(val_tags)
        print(f"  {tag_name}: {count}/{len(val_tags)} ({pct:.1f}%)")

    # Success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)

    success_criteria = []

    # Check 1: Dataset has actual singularities
    has_singularities = non_real_count > 0
    success_criteria.append(("Dataset includes actual singularities", has_singularities))

    # Check 2: Model produces non-REAL outputs
    model_produces_non_real = (tag_counts["PINF"] + tag_counts["NINF"] + tag_counts["PHI"]) > 0
    success_criteria.append(("Model produces non-REAL outputs", model_produces_non_real))

    # Check 3: Lambda stays above minimum
    if "lambda_rej" in history and len(history["lambda_rej"]) > 0:
        lambda_above_min = min(history["lambda_rej"]) >= 0.1
        success_criteria.append(("Lambda_rej stays above minimum (0.1)", lambda_above_min))

    # Check 4: Coverage is not always 100%
    if "coverage" in history and len(history["coverage"]) > 0:
        coverage_varies = min(history["coverage"]) < 1.0
        success_criteria.append(("Coverage is not always 100%", coverage_varies))

    # Print results
    all_passed = True
    for criterion, passed in success_criteria:
        status = "✓" if passed else "✗"
        print(f"{status} {criterion}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CRITICAL FIXES VERIFIED SUCCESSFULLY!")
    else:
        print("SOME CRITICAL FIXES NEED ATTENTION")
        print("\nNote: If coverage is still 100%, the model may need:")
        print("  - More aggressive pole placement in Q(x)")
        print("  - Lower learning rate to avoid jumping over poles")
        print("  - More training epochs to converge")
        print("  - Initialization closer to singular configurations")
    print("=" * 60)


if __name__ == "__main__":
    main()
