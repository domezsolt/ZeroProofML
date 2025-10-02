"""
Demonstration of adaptive loss policy in ZeroProof.

This example shows how the adaptive λ_rej automatically adjusts
to achieve a target coverage rate during training.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

import zeroproof as zp
from zeroproof.core import TRTag
from zeroproof.layers import ChebyshevBasis, TRRational
from zeroproof.training import (
    AdaptiveLambda,
    AdaptiveLossConfig,
    TrainingConfig,
    TRTrainer,
    create_adaptive_loss,
)
from zeroproof.utils import SingularDatasetGenerator

sys.stdout.reconfigure(encoding="utf-8")


def generate_synthetic_data(n_samples=1000, noise_level=0.1):
    """Generate data with actual singularities using SingularDatasetGenerator."""
    # Use the new dataset generator to ensure actual singularities
    generator = SingularDatasetGenerator(domain=(-1.0, 1.0), seed=42)

    # Add singularity at x=0.5
    generator.add_pole(location=0.5, strength=0.01)

    # Generate dataset with guaranteed singularities
    x_vals, y_vals, metadata = generator.generate_rational_function_data(
        n_samples=n_samples,
        singularity_ratio=0.3,  # 30% near/at singularities
        force_exact_singularities=True,
        noise_level=noise_level,
    )

    # Convert to numpy arrays for compatibility
    x = np.array(
        [
            float(xi.value)
            if xi.tag == TRTag.REAL
            else float("inf") * (1 if xi.tag == TRTag.PINF else -1 if xi.tag == TRTag.NINF else 0)
            for xi in x_vals
        ]
    )
    y = np.array(
        [
            float(yi.value)
            if yi.tag == TRTag.REAL
            else float("inf") * (1 if yi.tag == TRTag.PINF else -1 if yi.tag == TRTag.NINF else 0)
            for yi in y_vals
        ]
    )

    return x, y


def create_model():
    """Create a TR-Rational model."""
    # Use Chebyshev basis for better numerical stability
    basis = ChebyshevBasis()

    # Create adaptive loss policy
    adaptive_loss = create_adaptive_loss(
        target_coverage=0.90,  # Target 90% REAL outputs
        learning_rate=0.05,  # Lambda learning rate
        initial_lambda=1.0,  # Starting penalty
        base_loss="mse",
    )

    # Create TR-Rational layer
    model = TRRational(
        d_p=4,  # Numerator degree
        d_q=3,  # Denominator degree
        basis=basis,
        alpha_phi=0.01,  # Regularization
        adaptive_loss_policy=adaptive_loss,
    )

    return model, adaptive_loss


def prepare_data(x, y, batch_size=32):
    """Prepare data for training."""
    # Convert to TR scalars
    tr_data = []

    for i in range(0, len(x), batch_size):
        batch_x = []
        batch_y = []

        for j in range(i, min(i + batch_size, len(x))):
            # Input as TR scalar
            batch_x.append(zp.real(x[j]))

            # Target handling - convert infinities
            if np.isinf(y[j]):
                if y[j] > 0:
                    batch_y.append(zp.pinf())
                else:
                    batch_y.append(zp.ninf())
            elif np.isnan(y[j]):
                batch_y.append(zp.phi())
            else:
                batch_y.append(zp.real(y[j]))

        tr_data.append((batch_x, batch_y))

    return tr_data


def plot_training_history(history, adaptive_loss):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(history["loss"], label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True)

    # Coverage
    ax = axes[0, 1]
    if "coverage" in history:
        ax.plot(history["coverage"], label="Actual Coverage")
        ax.axhline(y=0.90, color="r", linestyle="--", label="Target Coverage")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Coverage")
        ax.set_title("REAL Output Coverage")
        ax.legend()
        ax.grid(True)

    # Lambda adaptation
    ax = axes[1, 0]
    if "lambda_rej" in history and len(history["lambda_rej"]) > 0:
        ax.plot(history["lambda_rej"], label="λ_rej")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("λ_rej")
        ax.set_title("Adaptive Rejection Penalty")
        ax.legend()
        ax.grid(True)

    # Lambda history (detailed)
    ax = axes[1, 1]
    if hasattr(adaptive_loss, "adaptive_lambda"):
        lambda_history = adaptive_loss.adaptive_lambda.lambda_history
        ax.plot(lambda_history, label="λ_rej (all steps)")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("λ_rej")
        ax.set_title("Detailed Lambda Evolution")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_model(model, x_test, y_test):
    """Evaluate trained model."""
    predictions = []
    tags = []

    for xi in x_test:
        y_pred = model(zp.real(xi))
        predictions.append(y_pred)
        tags.append(y_pred.tag)

    # Calculate coverage
    real_count = sum(1 for tag in tags if tag == zp.TRTag.REAL)
    coverage = real_count / len(tags)

    # Calculate error on REAL predictions
    errors = []
    for pred, true_val in zip(predictions, y_test):
        if pred.tag == zp.TRTag.REAL and np.isfinite(true_val):
            errors.append((pred.value.value - true_val) ** 2)

    mse = np.mean(errors) if errors else float("nan")

    print(f"\nEvaluation Results:")
    print(f"Coverage: {coverage:.3f} ({real_count}/{len(tags)} REAL)")
    print(f"MSE (REAL only): {mse:.6f}")

    # Tag distribution
    tag_counts = {tag.name: 0 for tag in zp.TRTag}
    for tag in tags:
        tag_counts[tag.name] += 1

    print("\nTag Distribution:")
    for tag_name, count in tag_counts.items():
        print(f"  {tag_name}: {count} ({count/len(tags)*100:.1f}%)")

    return predictions, coverage


def main():
    """Run adaptive loss demonstration."""
    print("Adaptive Loss Policy Demonstration")
    print("==================================")

    # Generate data
    print("\nGenerating synthetic data with singularity at x=0.5...")
    x_train, y_train = generate_synthetic_data(n_samples=800)
    x_test, y_test = generate_synthetic_data(n_samples=200)

    # Create model
    print("\nCreating TR-Rational model with adaptive loss...")
    model, adaptive_loss = create_model()

    # Prepare data
    train_data = prepare_data(x_train, y_train, batch_size=32)
    test_data = prepare_data(x_test, y_test, batch_size=32)

    # Training configuration
    config = TrainingConfig(
        learning_rate=0.01,
        max_epochs=50,
        use_adaptive_loss=True,
        target_coverage=0.90,
        log_interval=5,
        verbose=True,
    )

    # Create trainer
    trainer = TRTrainer(model, config=config)
    # Override with our adaptive loss
    trainer.loss_policy = adaptive_loss

    # Train model
    print(f"\nTraining with target coverage: {config.target_coverage}")
    history = trainer.train(train_data, test_data)

    # Evaluate
    predictions, coverage = evaluate_model(model, x_test, y_test)

    # Plot results
    plot_training_history(history, adaptive_loss)

    # Plot learned function
    plt.figure(figsize=(10, 6))

    # Sort for plotting
    idx = np.argsort(x_test)
    x_sorted = x_test[idx]
    y_sorted = y_test[idx]

    # True function
    plt.scatter(x_sorted[::5], y_sorted[::5], alpha=0.5, label="True", s=10)

    # Predictions
    y_pred = []
    for xi in x_sorted:
        pred = model(zp.real(xi))
        if pred.tag == zp.TRTag.REAL:
            y_pred.append(pred.value.value)
        else:
            y_pred.append(np.nan)

    plt.plot(x_sorted, y_pred, "r-", label="TR-Rational", linewidth=2)

    # Mark singularity
    plt.axvline(x=0.5, color="k", linestyle="--", alpha=0.5, label="Singularity")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Learned Function (Coverage: {coverage:.3f})")
    plt.legend()
    plt.ylim(-20, 20)
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print final statistics
    stats = adaptive_loss.get_statistics()
    print(f"\nFinal Adaptive Loss Statistics:")
    print(f"  Final λ_rej: {stats['lambda_rej']:.4f}")
    print(f"  Final coverage: {stats['current_coverage']:.3f}")
    print(f"  Coverage gap: {stats['coverage_gap']:.4f}")
    print(f"  Total samples: {stats['total_samples']}")


if __name__ == "__main__":
    main()
