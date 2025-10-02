"""
Demonstration of Coverage Control with adaptive lambda.

This example shows how coverage control prevents the model from trivially
rejecting too many near-pole points by dynamically adjusting the rejection
penalty lambda to maintain a target coverage level.

Run:
    python examples/coverage_control_demo.py
"""

import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zeroproof.autodiff import TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import MonomialBasis
from zeroproof.layers.hybrid_rational import HybridTRRational
from zeroproof.training.enhanced_coverage import CoverageStrategy
from zeroproof.training.hybrid_trainer import HybridTrainingConfig, HybridTRTrainer
from zeroproof.utils import SingularDatasetGenerator

sys.stdout.reconfigure(encoding="utf-8")


def generate_challenging_data(n_samples: int = 200) -> Tuple[List, List]:
    """
    Generate data that challenges coverage control using SingularDatasetGenerator.

    This dataset has actual singular points and many near-pole samples
    that the model might want to reject to minimize loss.

    Args:
        n_samples: Number of samples

    Returns:
        Tuple of (inputs, targets)
    """
    # Use the new dataset generator to ensure actual singularities
    generator = SingularDatasetGenerator(domain=(-1.0, 1.0), seed=42)

    # Add pole at x=0.5
    generator.add_pole(location=0.5, strength=0.01)

    # Generate dataset with guaranteed singularities
    x_vals, y_vals, metadata = generator.generate_rational_function_data(
        n_samples=n_samples,
        singularity_ratio=0.4,  # 40% near/at singularities
        force_exact_singularities=True,
        noise_level=0.05,
    )

    # x_vals and y_vals are already TRScalar objects
    return x_vals, y_vals


def visualize_coverage_control(trainer: HybridTRTrainer, test_x: List, test_y: List):
    """
    Visualize coverage control dynamics.

    Args:
        trainer: Trained model with coverage control
        test_x: Test inputs
        test_y: Test targets
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    history = trainer.history

    # 1. Loss evolution
    ax = axes[0, 0]
    if "loss" in history:
        epochs = range(len(history["loss"]))
        ax.plot(epochs, history["loss"], "b-", label="Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Evolution")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 2. Coverage evolution
    ax = axes[0, 1]
    if "coverage" in history:
        epochs = range(len(history["coverage"]))
        coverage_vals = history["coverage"]

        ax.plot(epochs, coverage_vals, "g-", label="Coverage", linewidth=2)

        # Mark target and min coverage
        if trainer.coverage_policy:
            target = trainer.coverage_policy.config.target_coverage
            min_cov = trainer.coverage_policy.config.min_coverage
            ax.axhline(
                y=target, color="blue", linestyle="--", label=f"Target ({target:.2f})", alpha=0.7
            )
            ax.axhline(
                y=min_cov, color="red", linestyle="--", label=f"Min ({min_cov:.2f})", alpha=0.7
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Coverage (% REAL)")
        ax.set_title("Coverage Control")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 3. Lambda evolution
    ax = axes[0, 2]
    if "lambda_rej" in history and len(history["lambda_rej"]) > 0:
        epochs = range(len(history["lambda_rej"]))
        lambda_vals = history["lambda_rej"]

        ax.plot(epochs, lambda_vals, "r-", label="λ_rej", linewidth=2)

        # Mark interventions if any
        if trainer.coverage_policy:
            interventions = trainer.coverage_policy.interventions
            for interv in interventions:
                ax.axvline(x=interv["epoch"], color="orange", linestyle=":", alpha=0.7)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rejection Penalty (λ)")
        ax.set_title("Adaptive Lambda Control")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 4. Coverage vs Lambda correlation
    ax = axes[1, 0]
    if (
        "coverage" in history
        and len(history["coverage"]) > 0
        and "lambda_rej" in history
        and len(history["lambda_rej"]) > 0
        and len(history["coverage"]) == len(history["lambda_rej"])
    ):
        # Show relationship
        ax.scatter(
            history["lambda_rej"],
            history["coverage"],
            c=range(len(history["coverage"])),
            cmap="viridis",
            alpha=0.6,
        )
        ax.set_xlabel("Lambda (λ_rej)")
        ax.set_ylabel("Coverage")
        ax.set_title("Coverage-Lambda Relationship")

        # Add colorbar for epoch
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Epoch")

        ax.grid(True, alpha=0.3)

    # 5. Model predictions
    ax = axes[1, 1]

    # Evaluate on test set
    x_vals = []
    y_pred_vals = []
    tag_colors = []

    for x in test_x:
        x_tr = TRNode.constant(x)
        y_pred, tag = trainer.model.forward(x_tr)

        x_vals.append(x.value)
        if tag == TRTag.REAL:
            y_pred_vals.append(y_pred.value.value)
            tag_colors.append("blue")
        else:
            y_pred_vals.append(0)  # Placeholder
            if tag == TRTag.PINF:
                tag_colors.append("red")
            elif tag == TRTag.NINF:
                tag_colors.append("darkred")
            else:
                tag_colors.append("orange")

    # Plot with colors indicating tags
    scatter = ax.scatter(x_vals, y_pred_vals, c=tag_colors, alpha=0.6, s=20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Model Predictions (colored by tag)")
    ax.set_ylim([-10, 10])
    ax.grid(True, alpha=0.3)

    # 6. Coverage statistics
    ax = axes[1, 2]

    if trainer.coverage_policy:
        stats = trainer.coverage_policy.get_statistics()
        lambda_stats = stats.get("lambda_stats", {})

        # Create bar chart of key metrics
        metrics = {
            "Adjustments": lambda_stats.get("adjustments", 0),
            "Violations": lambda_stats.get("violations", 0),
            "Interventions": stats.get("interventions", 0),
        }

        if lambda_stats.get("avg_coverage"):
            metrics["Avg Coverage"] = lambda_stats["avg_coverage"] * 100

        bars = ax.bar(range(len(metrics)), list(metrics.values()))
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics.keys(), rotation=45, ha="right")
        ax.set_title("Coverage Control Statistics")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}" if height > 1 else f"{height:.0f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.savefig("coverage_control_dynamics.png", dpi=150)
    plt.show()

    print("\nVisualization saved as 'coverage_control_dynamics.png'")


def run_coverage_control_comparison():
    """Compare training with and without coverage control."""

    print("=" * 70)
    print("COVERAGE CONTROL DEMONSTRATION")
    print("=" * 70)

    # Generate challenging data
    print("\n1. Generating challenging data with 40% near-pole samples...")
    train_x, train_y = generate_challenging_data(n_samples=300)
    test_x, test_y = generate_challenging_data(n_samples=100)

    print(f"   Training samples: {len(train_x)}")
    print(f"   Test samples: {len(test_x)}")

    # Model 1: Without coverage control
    print("\n2. Training WITHOUT coverage control...")
    print("-" * 50)

    model_no_control = HybridTRRational(
        d_p=3, d_q=3, basis=MonomialBasis(), alpha_phi=1e-3, lambda_rej=1.0  # Fixed lambda
    )

    config_no_control = HybridTrainingConfig(
        learning_rate=0.01,
        max_epochs=40,
        use_adaptive_loss=False,  # No adaptive lambda
        enforce_coverage=False,  # No coverage control
        verbose=False,
    )

    trainer_no_control = HybridTRTrainer(model=model_no_control, config=config_no_control)

    history_no_control = trainer_no_control.train(
        train_data=[(train_x, train_y)], val_data=[(test_x, test_y)]
    )

    # Model 2: With Lagrange multiplier control
    print("\n3. Training WITH coverage control (Lagrange)...")
    print("-" * 50)

    model_lagrange = HybridTRRational(d_p=3, d_q=3, basis=MonomialBasis(), alpha_phi=1e-3)

    config_lagrange = HybridTrainingConfig(
        learning_rate=0.01,
        max_epochs=40,
        use_adaptive_loss=True,
        enforce_coverage=True,
        target_coverage=0.80,  # Target 80% REAL outputs
        min_coverage=0.65,  # Minimum acceptable 65%
        coverage_strategy=CoverageStrategy.LAGRANGE,
        coverage_window_size=20,
        verbose=False,
    )

    trainer_lagrange = HybridTRTrainer(model=model_lagrange, config=config_lagrange)

    history_lagrange = trainer_lagrange.train(
        train_data=[(train_x, train_y)], val_data=[(test_x, test_y)]
    )

    # Model 3: With PID control
    print("\n4. Training WITH coverage control (PID)...")
    print("-" * 50)

    model_pid = HybridTRRational(d_p=3, d_q=3, basis=MonomialBasis(), alpha_phi=1e-3)

    config_pid = HybridTrainingConfig(
        learning_rate=0.01,
        max_epochs=40,
        use_adaptive_loss=True,
        enforce_coverage=True,
        target_coverage=0.80,
        min_coverage=0.65,
        coverage_strategy=CoverageStrategy.PID,
        verbose=False,
    )

    trainer_pid = HybridTRTrainer(model=model_pid, config=config_pid)

    history_pid = trainer_pid.train(train_data=[(train_x, train_y)], val_data=[(test_x, test_y)])

    # Compare results
    print("\n5. COMPARISON OF RESULTS")
    print("=" * 70)

    def get_final_metrics(history):
        """Extract final metrics from history."""
        metrics = {}
        metrics["loss"] = (
            history["loss"][-1] if "loss" in history and len(history["loss"]) > 0 else float("inf")
        )
        metrics["coverage"] = (
            history["coverage"][-1] if "coverage" in history and len(history["coverage"]) > 0 else 0
        )
        metrics["lambda"] = (
            history["lambda_rej"][-1]
            if "lambda_rej" in history and len(history["lambda_rej"]) > 0
            else 1.0
        )
        return metrics

    metrics_no_control = get_final_metrics(history_no_control)
    metrics_lagrange = get_final_metrics(history_lagrange)
    metrics_pid = get_final_metrics(history_pid)

    print("\nFinal Metrics:")
    print("-" * 50)
    print(f"{'Method':<20} {'Loss':<12} {'Coverage':<12} {'Lambda':<12}")
    print("-" * 50)
    print(
        f"{'No Control':<20} {metrics_no_control['loss']:<12.6f} "
        f"{metrics_no_control['coverage']:<12.3f} {metrics_no_control['lambda']:<12.3f}"
    )
    print(
        f"{'Lagrange Control':<20} {metrics_lagrange['loss']:<12.6f} "
        f"{metrics_lagrange['coverage']:<12.3f} {metrics_lagrange['lambda']:<12.3f}"
    )
    print(
        f"{'PID Control':<20} {metrics_pid['loss']:<12.6f} "
        f"{metrics_pid['coverage']:<12.3f} {metrics_pid['lambda']:<12.3f}"
    )

    # Coverage analysis over time
    print("\n6. COVERAGE STABILITY")
    print("-" * 50)

    def analyze_coverage_stability(history):
        """Analyze coverage stability."""
        if "coverage" not in history:
            return {}

        coverage = history["coverage"]
        return {
            "mean": np.mean(coverage),
            "std": np.std(coverage),
            "min": np.min(coverage),
            "max": np.max(coverage),
            "final": coverage[-1],
        }

    stab_no_control = analyze_coverage_stability(history_no_control)
    stab_lagrange = analyze_coverage_stability(history_lagrange)
    stab_pid = analyze_coverage_stability(history_pid)

    print(f"{'Method':<20} {'Mean±Std':<15} {'Min-Max':<15} {'Final':<10}")
    print("-" * 50)

    if stab_no_control:
        print(
            f"{'No Control':<20} "
            f"{stab_no_control['mean']:.3f}±{stab_no_control['std']:.3f}    "
            f"{stab_no_control['min']:.3f}-{stab_no_control['max']:.3f}    "
            f"{stab_no_control['final']:.3f}"
        )

    if stab_lagrange:
        print(
            f"{'Lagrange Control':<20} "
            f"{stab_lagrange['mean']:.3f}±{stab_lagrange['std']:.3f}    "
            f"{stab_lagrange['min']:.3f}-{stab_lagrange['max']:.3f}    "
            f"{stab_lagrange['final']:.3f}"
        )

    if stab_pid:
        print(
            f"{'PID Control':<20} "
            f"{stab_pid['mean']:.3f}±{stab_pid['std']:.3f}    "
            f"{stab_pid['min']:.3f}-{stab_pid['max']:.3f}    "
            f"{stab_pid['final']:.3f}"
        )

    # Enforcement statistics
    print("\n7. COVERAGE ENFORCEMENT STATISTICS")
    print("-" * 50)

    if trainer_lagrange.coverage_policy:
        stats = trainer_lagrange.coverage_policy.get_statistics()
        lambda_stats = stats.get("lambda_stats", {})
        print(f"Lagrange Controller:")
        print(f"  Lambda adjustments: {lambda_stats.get('adjustments', 0)}")
        print(f"  Coverage violations: {lambda_stats.get('violations', 0)}")
        print(f"  Interventions: {stats.get('interventions', 0)}")
        if stats.get("coverage_restored_epoch"):
            print(f"  Coverage restored at epoch: {stats['coverage_restored_epoch']}")

    if trainer_pid.coverage_policy:
        stats = trainer_pid.coverage_policy.get_statistics()
        lambda_stats = stats.get("lambda_stats", {})
        print(f"\nPID Controller:")
        print(f"  Lambda adjustments: {lambda_stats.get('adjustments', 0)}")
        print(f"  Coverage violations: {lambda_stats.get('violations', 0)}")
        print(f"  Interventions: {stats.get('interventions', 0)}")
        if stats.get("coverage_restored_epoch"):
            print(f"  Coverage restored at epoch: {stats['coverage_restored_epoch']}")

    # Visualize best performer
    print("\n8. Visualizing Lagrange control dynamics...")
    visualize_coverage_control(trainer_lagrange, test_x, test_y)

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

    # Key insights
    print("\nKEY INSIGHTS:")
    print("• Coverage control maintains desired REAL output percentage")
    print("• Adaptive lambda prevents trivial rejection of difficult samples")
    print("• Different strategies (Lagrange, PID) offer different convergence behaviors")
    print("• Interventions trigger when coverage drops critically low")
    print("• Model learns to handle near-pole samples rather than rejecting them")

    return trainer_no_control, trainer_lagrange, trainer_pid


if __name__ == "__main__":
    # Run the demonstration
    trainers = run_coverage_control_comparison()

    print("\nTo explore further:")
    print("• Try different target_coverage values (0.7 to 0.95)")
    print("• Experiment with coverage_strategy (LAGRANGE, PID, ADAPTIVE_RATE)")
    print("• Enable oversample_near_pole for weighted sampling")
    print("• Combine with hybrid gradients and tag-loss for maximum benefit")
