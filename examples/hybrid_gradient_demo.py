"""
Demonstration of Hybrid Gradient Schedule for near-pole learning.

This example shows how the hybrid gradient schedule enables learning
near singularities by transitioning from stable Mask-REAL mode to
targeted Saturating mode during training.

Run:
    python examples/hybrid_gradient_demo.py
"""

import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zeroproof.autodiff import TRNode
from zeroproof.autodiff.grad_mode import GradientMode, GradientModeConfig
from zeroproof.autodiff.hybrid_gradient import (
    HybridGradientContext,
    HybridGradientSchedule,
    ScheduleType,
)
from zeroproof.core import TRTag, real
from zeroproof.layers import MonomialBasis
from zeroproof.layers.hybrid_rational import HybridTRRational
from zeroproof.training.hybrid_trainer import HybridTrainingConfig, HybridTRTrainer
from zeroproof.utils import SingularDatasetGenerator

sys.stdout.reconfigure(encoding="utf-8")


def generate_near_pole_data(n_samples: int = 100, pole_location: float = 0.5) -> Tuple[List, List]:
    """
    Generate synthetic data with actual singularities at the specified location.

    The true function is: y = 1/(x - pole_location) + 0.5*x
    This has a simple pole at x = pole_location.

    Args:
        n_samples: Number of training samples
        pole_location: Location of the pole

    Returns:
        Tuple of (inputs, targets) as TR scalars
    """
    # Use the new dataset generator to ensure actual singularities
    generator = SingularDatasetGenerator(domain=(-1.0, 1.0), seed=42)

    # Add pole at specified location
    generator.add_pole(location=pole_location, strength=0.01)

    # Generate dataset with guaranteed singularities
    x_vals, y_vals, metadata = generator.generate_rational_function_data(
        n_samples=n_samples,
        singularity_ratio=0.35,  # 35% near/at singularities
        force_exact_singularities=True,
        noise_level=0.05,
    )

    # x_vals and y_vals are already TRScalar objects
    return x_vals, y_vals


def visualize_training_progress(
    trainer: HybridTRTrainer, test_x: List, test_y: List, pole_location: float = 0.5
):
    """
    Visualize the training progress and gradient mode evolution.

    Args:
        trainer: Trained hybrid trainer
        test_x: Test inputs
        test_y: Test targets
        pole_location: True pole location
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Loss curve
    ax = axes[0, 0]
    history = trainer.history
    if "loss" in history:
        epochs = range(len(history["loss"]))
        ax.plot(epochs, history["loss"], label="Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Evolution")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 2. Coverage evolution
    ax = axes[0, 1]
    if "coverage" in history:
        ax.plot(epochs, history["coverage"], label="REAL Coverage", color="green")
        ax.axhline(y=0.95, color="red", linestyle="--", label="Target Coverage")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Coverage")
        ax.set_title("Output Coverage (% REAL)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 3. Gradient mode progression
    ax = axes[1, 0]
    mode_history = trainer.gradient_mode_history
    if mode_history:
        epochs = [m["epoch"] for m in mode_history]
        deltas = [m["delta"] if m["delta"] is not None else 0 for m in mode_history]

        # Color code by phase
        colors = []
        for m in mode_history:
            if "warmup" in m["mode"].lower():
                colors.append("blue")
            elif "transitioning" in m["mode"].lower():
                colors.append("orange")
            else:
                colors.append("green")

        ax.scatter(epochs, deltas, c=colors, alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Delta Threshold")
        ax.set_title("Hybrid Schedule Evolution")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="blue", label="Warmup (Mask-REAL)"),
            Patch(facecolor="orange", label="Transitioning"),
            Patch(facecolor="green", label="Converged"),
        ]
        ax.legend(handles=legend_elements)

    # 4. Model predictions near pole
    ax = axes[1, 1]

    # Evaluate model on fine grid
    x_grid = np.linspace(-1, 1, 200)
    y_pred = []
    y_true = []

    for x_val in x_grid:
        # Skip exact pole
        if abs(x_val - pole_location) < 0.001:
            y_pred.append(np.nan)
            y_true.append(np.nan)
            continue

        # Model prediction
        x_tr = TRNode.constant(real(x_val))
        y_model, tag = trainer.model.forward(x_tr)

        if tag == TRTag.REAL:
            y_pred.append(y_model.value.value)
        else:
            # Non-REAL output (infinity or PHI)
            y_pred.append(np.nan)

        # True value
        y_true_val = 1.0 / (x_val - pole_location) + 0.5 * x_val
        y_true_val = np.clip(y_true_val, -20, 20)
        y_true.append(y_true_val)

    # Plot
    ax.plot(x_grid, y_true, "k-", label="True Function", alpha=0.5)
    ax.plot(x_grid, y_pred, "r-", label="Model Prediction", linewidth=2)
    ax.axvline(x=pole_location, color="gray", linestyle="--", alpha=0.5, label="True Pole")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Function Approximation Near Pole")
    ax.set_ylim([-20, 20])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("hybrid_gradient_progress.png", dpi=150)
    plt.show()

    print("\nPlot saved as 'hybrid_gradient_progress.png'")


def run_comparison():
    """Compare training with and without hybrid gradient schedule."""

    print("=" * 70)
    print("HYBRID GRADIENT SCHEDULE DEMONSTRATION")
    print("=" * 70)

    # Generate data
    print("\n1. Generating synthetic data with pole at x=0.5...")
    train_x, train_y = generate_near_pole_data(n_samples=150, pole_location=0.5)
    test_x, test_y = generate_near_pole_data(n_samples=50, pole_location=0.5)

    print(f"   Training samples: {len(train_x)}")
    print(f"   Test samples: {len(test_x)}")

    # Create models
    print("\n2. Creating rational models...")

    # Model 1: Standard (Mask-REAL only)
    model_standard = HybridTRRational(
        d_p=3,
        d_q=3,
        basis=MonomialBasis(),
        alpha_phi=1e-3,
        lambda_rej=1.0,
        hybrid_schedule=None,  # No hybrid schedule
    )

    # Model 2: With hybrid gradient schedule
    hybrid_schedule = HybridGradientSchedule(
        warmup_epochs=10,
        transition_epochs=20,
        delta_init=5e-2,
        delta_final=1e-4,
        schedule_type=ScheduleType.EXPONENTIAL,
        enable=True,
    )

    model_hybrid = HybridTRRational(
        d_p=3,
        d_q=3,
        basis=MonomialBasis(),
        alpha_phi=1e-3,
        lambda_rej=1.0,
        hybrid_schedule=hybrid_schedule,
        track_Q_values=True,
    )

    # Train standard model
    print("\n3. Training STANDARD model (Mask-REAL only)...")
    print("-" * 50)

    config_standard = HybridTrainingConfig(
        learning_rate=0.01,
        max_epochs=40,
        use_adaptive_loss=True,
        target_coverage=0.90,
        use_hybrid_gradient=False,  # No hybrid
        verbose=False,
    )

    trainer_standard = HybridTRTrainer(model=model_standard, config=config_standard)

    history_standard = trainer_standard.train(
        train_data=[(train_x, train_y)], val_data=[(test_x, test_y)]
    )

    # Train hybrid model
    print("\n4. Training HYBRID model (with gradient schedule)...")
    print("-" * 50)

    config_hybrid = HybridTrainingConfig(
        learning_rate=0.01,
        max_epochs=40,
        use_adaptive_loss=True,
        target_coverage=0.90,
        use_hybrid_gradient=True,
        hybrid_warmup_epochs=10,
        hybrid_transition_epochs=20,
        hybrid_delta_init=5e-2,
        hybrid_delta_final=1e-4,
        verbose=False,
    )

    trainer_hybrid = HybridTRTrainer(model=model_hybrid, config=config_hybrid)

    history_hybrid = trainer_hybrid.train(
        train_data=[(train_x, train_y)], val_data=[(test_x, test_y)]
    )

    # Compare results
    print("\n5. COMPARISON OF RESULTS")
    print("=" * 70)

    # Final losses
    final_loss_standard = (
        history_standard["loss"][-1] if "loss" in history_standard else float("inf")
    )
    final_loss_hybrid = history_hybrid["loss"][-1] if "loss" in history_hybrid else float("inf")

    print(f"\nFinal Training Loss:")
    print(f"  Standard (Mask-REAL):  {final_loss_standard:.6f}")
    print(f"  Hybrid Schedule:       {final_loss_hybrid:.6f}")
    print(f"  Improvement:           {(1 - final_loss_hybrid/final_loss_standard)*100:.1f}%")

    # Coverage
    final_coverage_standard = (
        history_standard["coverage"][-1] if "coverage" in history_standard else 0
    )
    final_coverage_hybrid = history_hybrid["coverage"][-1] if "coverage" in history_hybrid else 0

    print(f"\nFinal Coverage (% REAL outputs):")
    print(f"  Standard:              {final_coverage_standard:.3f}")
    print(f"  Hybrid:                {final_coverage_hybrid:.3f}")

    # Gradient mode statistics (for hybrid)
    if hasattr(model_hybrid, "get_hybrid_statistics"):
        stats = model_hybrid.get_hybrid_statistics()
        print(f"\nHybrid Gradient Statistics:")
        print(f"  Mode: {stats.get('mode_description', 'N/A')}")
        print(f"  Near-pole gradient calls: {stats.get('near_pole_ratio', 0)*100:.1f}%")
        print(f"  Min |Q| observed: {stats.get('q_min', 'N/A')}")

    # Test set evaluation
    print("\n6. TEST SET EVALUATION")
    print("-" * 50)

    def evaluate_model(model, test_x, test_y):
        """Evaluate model on test set."""
        total_loss = 0
        real_count = 0

        for x, y_true in zip(test_x, test_y):
            y_pred, tag = model.forward(x)

            if tag == TRTag.REAL:
                loss = 0.5 * (y_pred.value.value - y_true.value) ** 2
                total_loss += loss
                real_count += 1

        avg_loss = total_loss / real_count if real_count > 0 else float("inf")
        coverage = real_count / len(test_x)

        return avg_loss, coverage

    test_loss_standard, test_cov_standard = evaluate_model(model_standard, test_x, test_y)
    test_loss_hybrid, test_cov_hybrid = evaluate_model(model_hybrid, test_x, test_y)

    print(f"\nTest Loss (MSE on REAL outputs):")
    print(f"  Standard:              {test_loss_standard:.6f}")
    print(f"  Hybrid:                {test_loss_hybrid:.6f}")

    print(f"\nTest Coverage:")
    print(f"  Standard:              {test_cov_standard:.3f}")
    print(f"  Hybrid:                {test_cov_hybrid:.3f}")

    # Visualize hybrid training
    print("\n7. Generating visualization for hybrid model...")
    visualize_training_progress(trainer_hybrid, test_x, test_y, pole_location=0.5)

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

    # Summary
    print("\nKEY INSIGHTS:")
    print("• The hybrid schedule enables learning near poles without instability")
    print("• Warmup phase ensures stable initial training")
    print("• Transition phase gradually enables near-pole gradient flow")
    print("• Final performance shows improved accuracy near singularities")

    return trainer_standard, trainer_hybrid


if __name__ == "__main__":
    # Run the comparison
    trainer_standard, trainer_hybrid = run_comparison()

    print("\nTo explore further:")
    print("• Adjust hybrid_warmup_epochs and hybrid_transition_epochs")
    print("• Try different delta_init and delta_final values")
    print("• Experiment with ScheduleType.LINEAR or ScheduleType.COSINE")
    print("• Enable tag_loss and pole_head for advanced features")
