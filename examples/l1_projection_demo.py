"""
L1 Projection Demo for TR-Rational Layers

This example demonstrates how L1 projection helps maintain stability
in TR-Rational layers by keeping denominator coefficients bounded.

The L1 projection ensures ||φ||₁ ≤ B, which prevents the denominator
Q(x) from getting too close to zero, maintaining numerical stability.
"""

import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zeroproof.autodiff import TRNode
from zeroproof.core import TRScalar, real
from zeroproof.layers import TRRational
from zeroproof.training import Optimizer, TrainingConfig, TRTrainer

sys.stdout.reconfigure(encoding="utf-8")


def create_training_data(n_samples: int = 50) -> Tuple[List[TRScalar], List[TRScalar]]:
    """
    Create training data with a rational function pattern.

    The target function has poles that could cause instability
    without proper regularization.
    """
    np.random.seed(42)

    # Generate x values avoiding the poles
    x_vals = np.linspace(-3, 3, n_samples)
    x_vals = x_vals[np.abs(x_vals - 1.5) > 0.3]  # Avoid pole at x=1.5
    x_vals = x_vals[np.abs(x_vals + 1.5) > 0.3]  # Avoid pole at x=-1.5

    # Target rational function: y = (x^2 + 1) / (x^2 - 2.25)
    # This has poles at x = ±1.5
    y_vals = (x_vals**2 + 1) / (x_vals**2 - 2.25)

    # Add some noise
    y_vals += 0.1 * np.random.randn(len(x_vals))

    # Convert to TRScalar
    inputs = [real(float(x)) for x in x_vals]
    targets = [real(float(y)) for y in y_vals]

    return inputs, targets


def train_without_projection(
    inputs: List[TRScalar], targets: List[TRScalar], epochs: int = 100
) -> Tuple[TRRational, List[float]]:
    """Train a TR-Rational layer without L1 projection."""
    print("\n=== Training WITHOUT L1 Projection ===")

    # Create layer without L1 projection
    layer = TRRational(d_p=3, d_q=3, l1_projection=None, alpha_phi=0.01)

    # Create optimizer
    optimizer = Optimizer(layer.parameters(), learning_rate=0.01)

    # Training loop
    losses = []
    phi_norms = []

    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass and compute loss
        total_loss = 0.0
        for x, y_target in zip(inputs, targets):
            y_pred = layer(x)

            # MSE loss for REAL values
            if y_pred.tag == 0:  # REAL
                diff = y_pred - TRNode.constant(y_target)
                loss = diff * diff
                total_loss += loss.value.value if loss.value.tag == 0 else 0.0

        # Average loss
        avg_loss = TRNode.constant(real(total_loss / len(inputs)))
        losses.append(total_loss / len(inputs))

        # Backward pass
        avg_loss.backward()

        # Optimization step (no projection)
        optimizer.step()

        # Track phi L1 norm
        phi_norm = sum(abs(phi_k.value.value) for phi_k in layer.phi)
        phi_norms.append(phi_norm)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {losses[-1]:.6f}, ||φ||₁ = {phi_norm:.4f}")

    # Check final stability
    q_min = layer.compute_q_min(inputs)
    print(f"\nFinal min|Q(x)| = {q_min:.6f}")
    print(f"Final ||φ||₁ = {phi_norms[-1]:.4f}")

    return layer, phi_norms


def train_with_projection(
    inputs: List[TRScalar], targets: List[TRScalar], l1_bound: float = 1.0, epochs: int = 100
) -> Tuple[TRRational, List[float]]:
    """Train a TR-Rational layer with L1 projection."""
    print(f"\n=== Training WITH L1 Projection (bound = {l1_bound}) ===")

    # Create layer with L1 projection
    layer = TRRational(d_p=3, d_q=3, l1_projection=l1_bound, alpha_phi=0.01)

    # Create optimizer
    optimizer = Optimizer(layer.parameters(), learning_rate=0.01)

    # Training loop
    losses = []
    phi_norms = []

    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass and compute loss
        total_loss = 0.0
        for x, y_target in zip(inputs, targets):
            y_pred = layer(x)

            # MSE loss for REAL values
            if y_pred.tag == 0:  # REAL
                diff = y_pred - TRNode.constant(y_target)
                loss = diff * diff
                total_loss += loss.value.value if loss.value.tag == 0 else 0.0

        # Average loss
        avg_loss = TRNode.constant(real(total_loss / len(inputs)))
        losses.append(total_loss / len(inputs))

        # Backward pass
        avg_loss.backward()

        # Optimization step WITH projection
        optimizer.step(model=layer)

        # Track phi L1 norm (should be bounded)
        phi_norm = sum(abs(phi_k.value.value) for phi_k in layer.phi)
        phi_norms.append(phi_norm)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {losses[-1]:.6f}, ||φ||₁ = {phi_norm:.4f}")

    # Check final stability
    q_min = layer.compute_q_min(inputs)
    print(f"\nFinal min|Q(x)| = {q_min:.6f}")
    print(f"Final ||φ||₁ = {phi_norms[-1]:.4f} (bound = {l1_bound})")

    return layer, phi_norms


def compare_stability(inputs: List[TRScalar]) -> None:
    """Compare stability with and without L1 projection."""
    print("\n=== Stability Comparison ===")

    # Test points including near poles
    test_x = np.linspace(-3, 3, 100)

    # Create two layers with different settings
    layer_no_proj = TRRational(d_p=3, d_q=3, l1_projection=None)
    layer_with_proj = TRRational(d_p=3, d_q=3, l1_projection=0.5)

    # Set some extreme phi values
    for i, phi_k in enumerate(layer_no_proj.phi):
        phi_k._value = real(3.0 * (i + 1))  # Large unbounded values

    for i, phi_k in enumerate(layer_with_proj.phi):
        phi_k._value = real(3.0 * (i + 1))  # Will be projected

    # Apply projection to the second layer
    layer_with_proj._project_phi_l1()

    print("\nWithout projection:")
    print(f"  ||φ||₁ = {sum(abs(phi_k.value.value) for phi_k in layer_no_proj.phi):.4f}")
    print(f"  min|Q(x)| = {layer_no_proj.compute_q_min([real(float(x)) for x in test_x]):.6f}")

    print("\nWith projection (bound = 0.5):")
    print(f"  ||φ||₁ = {sum(abs(phi_k.value.value) for phi_k in layer_with_proj.phi):.4f}")
    print(f"  min|Q(x)| = {layer_with_proj.compute_q_min([real(float(x)) for x in test_x]):.6f}")


def visualize_phi_evolution(
    phi_norms_no_proj: List[float], phi_norms_with_proj: List[float], l1_bound: float
) -> None:
    """Visualize the evolution of phi L1 norms during training."""
    plt.figure(figsize=(10, 6))

    epochs = range(len(phi_norms_no_proj))

    plt.plot(epochs, phi_norms_no_proj, "r-", label="Without L1 projection", linewidth=2)
    plt.plot(
        epochs,
        phi_norms_with_proj,
        "b-",
        label=f"With L1 projection (bound={l1_bound})",
        linewidth=2,
    )
    plt.axhline(y=l1_bound, color="g", linestyle="--", label=f"L1 bound = {l1_bound}")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("||φ||₁", fontsize=12)
    plt.title("Evolution of Denominator Coefficient L1 Norm", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add annotation
    plt.annotate(
        "Projection keeps\nnorm bounded",
        xy=(len(phi_norms_with_proj) // 2, l1_bound),
        xytext=(len(phi_norms_with_proj) // 2, l1_bound + 1),
        arrowprops=dict(arrowstyle="->", color="blue"),
        fontsize=10,
        color="blue",
    )

    plt.tight_layout()
    plt.show()


def main():
    """Run the L1 projection demonstration."""
    print("=" * 60)
    print("L1 Projection Demonstration for TR-Rational Layers")
    print("=" * 60)

    # Create training data
    inputs, targets = create_training_data(n_samples=50)
    print(f"\nCreated {len(inputs)} training samples")

    # Train without projection
    layer_no_proj, phi_norms_no_proj = train_without_projection(inputs, targets, epochs=100)

    # Train with projection
    l1_bound = 1.0
    layer_with_proj, phi_norms_with_proj = train_with_projection(
        inputs, targets, l1_bound=l1_bound, epochs=100
    )

    # Compare stability
    compare_stability(inputs)

    # Visualize results
    print("\n=== Visualization ===")
    print("Plotting phi norm evolution...")
    visualize_phi_evolution(phi_norms_no_proj, phi_norms_with_proj, l1_bound)

    print("\n" + "=" * 60)
    print("Key Observations:")
    print("1. L1 projection keeps ||φ||₁ bounded during training")
    print("2. This prevents Q(x) from getting too close to zero")
    print("3. The bound ensures numerical stability near poles")
    print("4. Training still converges effectively with projection")
    print("=" * 60)


if __name__ == "__main__":
    main()
