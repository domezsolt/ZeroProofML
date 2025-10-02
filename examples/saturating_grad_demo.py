"""
Demonstration of saturating gradient mode in ZeroProof.

This example compares the behavior of Mask-REAL and saturating
gradient modes when training near singularities.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

import zeroproof as zp
from zeroproof.autodiff import GradientMode, TRNode, gradient_mode
from zeroproof.layers import ChebyshevBasis, SaturatingTRRational

sys.stdout.reconfigure(encoding="utf-8")


def generate_data_with_poles():
    """Generate synthetic data with known poles."""
    np.random.seed(42)

    # True function: 1/(x-0.3) + 0.5/(x+0.7)
    # Poles at x=0.3 and x=-0.7
    def true_function(x):
        if abs(x - 0.3) < 1e-10:
            return float("inf") if x > 0.3 else float("-inf")
        elif abs(x + 0.7) < 1e-10:
            return float("inf") if x > -0.7 else float("-inf")
        else:
            return 1.0 / (x - 0.3) + 0.5 / (x + 0.7)

    # Generate training points avoiding exact poles
    x_train = []
    y_train = []

    # Regular points
    for x in np.linspace(-1, 1, 100):
        if abs(x - 0.3) > 0.05 and abs(x + 0.7) > 0.05:
            x_train.append(x)
            y_train.append(true_function(x))

    # Add some points very close to poles
    for offset in [0.01, 0.02, -0.01, -0.02]:
        x_train.extend([0.3 + offset, -0.7 + offset])
        y_train.extend([true_function(0.3 + offset), true_function(-0.7 + offset)])

    # Add noise
    y_train = np.array(y_train)
    y_train += np.random.normal(0, 0.1, len(y_train))

    return np.array(x_train), y_train, true_function


def train_with_mode(x_train, y_train, mode, saturation_bound=1.0, epochs=100):
    """Train a rational model with specified gradient mode."""
    # Create model
    model = SaturatingTRRational(
        d_p=4,
        d_q=3,
        basis=ChebyshevBasis(),
        gradient_mode=mode,
        saturation_bound=saturation_bound,
        alpha_phi=0.01,  # Regularization
    )

    # Training history
    loss_history = []
    gradient_norms = []
    coverage_history = []

    # Simple gradient descent
    learning_rate = 0.001

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_gradients = []
        real_count = 0

        # Mini-batch training
        indices = np.random.permutation(len(x_train))

        for idx in indices:
            # Zero gradients
            for param in model.parameters():
                param.zero_grad()

            # Forward pass
            x = zp.real(x_train[idx])
            y_true = zp.real(y_train[idx])

            y_pred, tag = model.forward(x)

            if tag == zp.TRTag.REAL:
                real_count += 1

            # Loss computation
            if tag == zp.TRTag.REAL:
                loss = 0.5 * (y_pred - TRNode.constant(y_true)) ** 2
            else:
                # Rejection penalty
                loss = TRNode.constant(zp.real(1.0))

            # Backward pass
            loss.backward()

            # Collect gradient magnitudes
            grad_norm = 0.0
            param_count = 0
            for param in model.parameters():
                if param.gradient is not None and param.gradient.tag == zp.TRTag.REAL:
                    grad_norm += param.gradient.value**2
                    param_count += 1

            if param_count > 0:
                epoch_gradients.append(np.sqrt(grad_norm))

            # Update parameters
            for param in model.parameters():
                if param.gradient is not None and param.gradient.tag == zp.TRTag.REAL:
                    new_value = param.value.value - learning_rate * param.gradient.value
                    param._value = zp.real(new_value)

            if loss.value.tag == zp.TRTag.REAL:
                epoch_loss += loss.value.value

        # Record metrics
        loss_history.append(epoch_loss / len(x_train))
        gradient_norms.append(np.mean(epoch_gradients) if epoch_gradients else 0.0)
        coverage_history.append(real_count / len(x_train))

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch}: Loss={loss_history[-1]:.4f}, "
                f"Coverage={coverage_history[-1]:.3f}, "
                f"Avg Gradient Norm={gradient_norms[-1]:.4f}"
            )

    return model, loss_history, gradient_norms, coverage_history


def evaluate_model(model, x_range):
    """Evaluate model over a range."""
    y_pred = []
    tags = []

    for x in x_range:
        y, tag = model.forward(zp.real(x))
        if tag == zp.TRTag.REAL:
            y_pred.append(y.value.value)
        else:
            y_pred.append(np.nan)
        tags.append(tag)

    return np.array(y_pred), tags


def main():
    """Run saturating gradient demonstration."""
    print("Saturating Gradient Mode Demonstration")
    print("======================================\n")

    # Generate data
    x_train, y_train, true_func = generate_data_with_poles()
    print(f"Generated {len(x_train)} training points with poles at x=0.3 and x=-0.7\n")

    # Train with different modes
    print("Training with Mask-REAL mode...")
    model_mask, loss_mask, grad_mask, coverage_mask = train_with_mode(
        x_train, y_train, GradientMode.MASK_REAL, epochs=100
    )

    print("\nTraining with Saturating mode (bound=1.0)...")
    model_sat1, loss_sat1, grad_sat1, coverage_sat1 = train_with_mode(
        x_train, y_train, GradientMode.SATURATING, saturation_bound=1.0, epochs=100
    )

    print("\nTraining with Saturating mode (bound=5.0)...")
    model_sat5, loss_sat5, grad_sat5, coverage_sat5 = train_with_mode(
        x_train, y_train, GradientMode.SATURATING, saturation_bound=5.0, epochs=100
    )

    # Evaluation
    x_eval = np.linspace(-1, 1, 500)
    y_mask, tags_mask = evaluate_model(model_mask, x_eval)
    y_sat1, tags_sat1 = evaluate_model(model_sat1, x_eval)
    y_sat5, tags_sat5 = evaluate_model(model_sat5, x_eval)

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loss history
    ax = axes[0, 0]
    ax.plot(loss_mask, label="Mask-REAL", alpha=0.7)
    ax.plot(loss_sat1, label="Saturating (b=1)", alpha=0.7)
    ax.plot(loss_sat5, label="Saturating (b=5)", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient norms
    ax = axes[0, 1]
    ax.plot(grad_mask, label="Mask-REAL", alpha=0.7)
    ax.plot(grad_sat1, label="Saturating (b=1)", alpha=0.7)
    ax.plot(grad_sat5, label="Saturating (b=5)", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Gradient Norm")
    ax.set_title("Gradient Magnitudes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Coverage
    ax = axes[0, 2]
    ax.plot(coverage_mask, label="Mask-REAL", alpha=0.7)
    ax.plot(coverage_sat1, label="Saturating (b=1)", alpha=0.7)
    ax.plot(coverage_sat5, label="Saturating (b=5)", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Coverage (REAL outputs)")
    ax.set_title("Coverage Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learned functions
    for idx, (y_pred, tags, title) in enumerate(
        [
            (y_mask, tags_mask, "Mask-REAL"),
            (y_sat1, tags_sat1, "Saturating (b=1)"),
            (y_sat5, tags_sat5, "Saturating (b=5)"),
        ]
    ):
        ax = axes[1, idx]

        # Plot training data
        ax.scatter(x_train, y_train, alpha=0.3, s=10, label="Data")

        # Plot predictions
        mask_real = [tag == zp.TRTag.REAL for tag in tags]
        ax.plot(x_eval[mask_real], y_pred[mask_real], "r-", linewidth=2, label="Model")

        # Mark non-REAL regions
        mask_nonreal = [tag != zp.TRTag.REAL for tag in tags]
        if any(mask_nonreal):
            x_nonreal = x_eval[mask_nonreal]
            if len(x_nonreal) > 0:
                ax.axvspan(x_nonreal[0], x_nonreal[-1], alpha=0.1, color="gray", label="Non-REAL")

        # Mark true poles
        ax.axvline(x=0.3, color="k", linestyle="--", alpha=0.5)
        ax.axvline(x=-0.7, color="k", linestyle="--", alpha=0.5)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Learned Function: {title}")
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Print final statistics
    print("\nFinal Statistics:")
    print(f"{'Mode':<20} {'Final Loss':<15} {'Final Coverage':<15} {'Avg Gradient':<15}")
    print("-" * 65)
    print(
        f"{'Mask-REAL':<20} {loss_mask[-1]:<15.4f} {coverage_mask[-1]:<15.3f} {grad_mask[-1]:<15.4f}"
    )
    print(
        f"{'Saturating (b=1)':<20} {loss_sat1[-1]:<15.4f} {coverage_sat1[-1]:<15.3f} {grad_sat1[-1]:<15.4f}"
    )
    print(
        f"{'Saturating (b=5)':<20} {loss_sat5[-1]:<15.4f} {coverage_sat5[-1]:<15.3f} {grad_sat5[-1]:<15.4f}"
    )

    # Analyze gradient behavior near poles
    print("\nGradient Analysis Near Poles:")
    test_points = [0.31, 0.29, -0.69, -0.71]  # Points near poles

    for x_test in test_points:
        print(f"\nAt x = {x_test}:")

        # Test each model
        for model, mode_name in [
            (model_mask, "Mask-REAL"),
            (model_sat1, "Saturating (b=1)"),
            (model_sat5, "Saturating (b=5)"),
        ]:
            # Zero gradients
            for param in model.parameters():
                param.zero_grad()

            # Forward and backward
            x = zp.real(x_test)
            y, tag = model.forward(x)
            y.backward()

            # Collect gradient info
            grad_norm = 0.0
            for param in model.parameters():
                if param.gradient is not None and param.gradient.tag == zp.TRTag.REAL:
                    grad_norm += param.gradient.value**2
            grad_norm = np.sqrt(grad_norm)

            print(f"  {mode_name}: tag={tag.name}, grad_norm={grad_norm:.4f}")


if __name__ == "__main__":
    main()
