"""
Demonstration of Tag-Loss for non-REAL outputs.

This example shows how tag-loss enables non-REAL samples (infinities and PHI)
to contribute supervision through auxiliary classification, ensuring that
difficult samples near poles still provide learning signals.

Run:
    python examples/tag_loss_demo.py
"""

import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zeroproof.autodiff import TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import MonomialBasis
from zeroproof.layers.tag_aware_rational import TagAwareRational
from zeroproof.training.hybrid_trainer import HybridTrainingConfig, HybridTRTrainer
from zeroproof.training.tag_loss import TagClass, compute_tag_accuracy

sys.stdout.reconfigure(encoding="utf-8")


def generate_multi_pole_data(n_samples: int = 200) -> Tuple[List, List]:
    """
    Generate data with multiple types of singularities.

    True function has:
    - Simple pole at x = 0.3 (produces PINF/NINF)
    - Removable singularity at x = -0.5 (produces PHI)
    - Regular regions (produces REAL)

    Args:
        n_samples: Number of samples

    Returns:
        Tuple of (inputs, targets)
    """
    x_vals = []
    y_vals = []

    for _ in range(n_samples):
        # Sample with bias toward interesting regions
        region = np.random.choice(["pole1", "pole2", "regular"], p=[0.3, 0.3, 0.4])

        if region == "pole1":
            # Near simple pole at 0.3
            offset = np.random.uniform(0.01, 0.15) * np.random.choice([-1, 1])
            x = 0.3 + offset
        elif region == "pole2":
            # Near removable singularity at -0.5
            offset = np.random.uniform(0.01, 0.15) * np.random.choice([-1, 1])
            x = -0.5 + offset
        else:
            # Regular region
            x = np.random.uniform(-1, 1)
            while abs(x - 0.3) < 0.1 or abs(x + 0.5) < 0.1:
                x = np.random.uniform(-1, 1)

        # Compute target with different singularity types
        if abs(x - 0.3) < 0.01:
            # Very near pole -> infinity
            y = 1.0 / (x - 0.3)
            y = np.clip(y, -100, 100)
        elif abs(x + 0.5) < 0.01:
            # Near removable singularity -> indeterminate
            # Model sin(x+0.5)/(x+0.5) behavior
            y = np.sin(x + 0.5) / (x + 0.5) if abs(x + 0.5) > 1e-6 else 1.0
        else:
            # Regular function
            y = np.sin(2 * x) + 0.5 * x

        # Add noise
        y += np.random.normal(0, 0.01)

        x_vals.append(real(x))
        y_vals.append(real(y))

    return x_vals, y_vals


def visualize_tag_predictions(model: TagAwareRational, test_x: List, test_y: List):
    """
    Visualize model predictions and tag classification.

    Args:
        model: Trained tag-aware model
        test_x: Test inputs
        test_y: Test targets
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Evaluate on grid for visualization
    x_grid = np.linspace(-1, 1, 300)
    y_pred = []
    tags_true = []
    tags_pred = []
    tag_probs_real = []
    tag_probs_pinf = []
    tag_probs_ninf = []
    tag_probs_phi = []

    for x_val in x_grid:
        x_tr = TRNode.constant(real(x_val))
        y, tag, tag_logits = model.forward_with_tag_pred(x_tr)

        # Store prediction
        if tag == TRTag.REAL:
            y_pred.append(y.value.value)
        else:
            y_pred.append(np.nan)

        tags_true.append(tag)

        # Get tag prediction
        if tag_logits:
            from zeroproof.training.tag_loss import softmax

            probs = softmax(tag_logits)

            # Extract probabilities
            prob_vals = []
            for p in probs:
                if p.value.tag == TRTag.REAL:
                    prob_vals.append(p.value.value)
                else:
                    prob_vals.append(0.0)

            tag_probs_real.append(prob_vals[0])
            tag_probs_pinf.append(prob_vals[1])
            tag_probs_ninf.append(prob_vals[2])
            tag_probs_phi.append(prob_vals[3])

            # Predicted class
            pred_idx = np.argmax(prob_vals)
            tags_pred.append(TagClass(pred_idx))
        else:
            tag_probs_real.append(0)
            tag_probs_pinf.append(0)
            tag_probs_ninf.append(0)
            tag_probs_phi.append(0)
            tags_pred.append(TagClass.REAL)

    # 1. Function approximation
    ax = axes[0, 0]
    ax.plot(x_grid, y_pred, "b-", label="Model", linewidth=2)

    # Mark singularities
    ax.axvline(x=0.3, color="red", linestyle="--", alpha=0.5, label="Simple Pole")
    ax.axvline(x=-0.5, color="orange", linestyle="--", alpha=0.5, label="Removable Sing.")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Function Approximation")
    ax.set_ylim([-10, 10])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. True tags
    ax = axes[0, 1]
    tag_colors = []
    for tag in tags_true:
        if tag == TRTag.REAL:
            tag_colors.append("blue")
        elif tag == TRTag.PINF:
            tag_colors.append("red")
        elif tag == TRTag.NINF:
            tag_colors.append("darkred")
        else:  # PHI
            tag_colors.append("orange")

    ax.scatter(x_grid, [1] * len(x_grid), c=tag_colors, s=5, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("Tag")
    ax.set_title("True Output Tags")
    ax.set_ylim([0.5, 1.5])

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", label="REAL"),
        Patch(facecolor="red", label="PINF"),
        Patch(facecolor="darkred", label="NINF"),
        Patch(facecolor="orange", label="PHI"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # 3. Tag probabilities
    ax = axes[1, 0]
    ax.plot(x_grid, tag_probs_real, "b-", label="P(REAL)", alpha=0.7)
    ax.plot(x_grid, tag_probs_pinf, "r-", label="P(PINF)", alpha=0.7)
    ax.plot(x_grid, tag_probs_ninf, "darkred", label="P(NINF)", alpha=0.7, linestyle="--")
    ax.plot(x_grid, tag_probs_phi, "orange", label="P(PHI)", alpha=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("Probability")
    ax.set_title("Tag Classification Probabilities")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Tag accuracy
    ax = axes[1, 1]

    # Compute accuracy in windows
    window_size = 30
    x_windows = []
    accuracies = []

    for i in range(0, len(x_grid) - window_size, window_size // 2):
        window_tags_true = tags_true[i : i + window_size]
        window_tags_pred = tags_pred[i : i + window_size]

        correct = sum(
            1 for t, p in zip(window_tags_true, window_tags_pred) if TagClass.from_tag(t) == p
        )
        acc = correct / len(window_tags_true)

        x_windows.append(x_grid[i + window_size // 2])
        accuracies.append(acc)

    ax.plot(x_windows, accuracies, "g-", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("Accuracy")
    ax.set_title("Tag Classification Accuracy (windowed)")
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tag_loss_visualization.png", dpi=150)
    plt.show()

    print("\nVisualization saved as 'tag_loss_visualization.png'")


def run_tag_loss_demo():
    """Run tag-loss demonstration."""

    print("=" * 70)
    print("TAG-LOSS DEMONSTRATION")
    print("=" * 70)

    # Generate data
    print("\n1. Generating data with multiple singularity types...")
    train_x, train_y = generate_multi_pole_data(n_samples=300)
    test_x, test_y = generate_multi_pole_data(n_samples=100)

    print(f"   Training samples: {len(train_x)}")
    print(f"   Test samples: {len(test_x)}")

    # Create models for comparison
    print("\n2. Creating models...")

    # Model 1: Without tag loss
    model_notag = TagAwareRational(
        d_p=3, d_q=3, basis=MonomialBasis(), alpha_phi=1e-3, enable_tag_head=False  # No tag head
    )

    # Model 2: With tag loss
    model_withtag = TagAwareRational(
        d_p=3,
        d_q=3,
        basis=MonomialBasis(),
        alpha_phi=1e-3,
        enable_tag_head=True,
        tag_head_hidden_dim=8,
    )

    # Train without tag loss
    print("\n3. Training WITHOUT tag-loss...")
    print("-" * 50)

    config_notag = HybridTrainingConfig(
        learning_rate=0.01,
        max_epochs=30,
        use_adaptive_loss=True,
        target_coverage=0.85,
        use_tag_loss=False,  # No tag loss
        verbose=False,
    )

    trainer_notag = HybridTRTrainer(model=model_notag, config=config_notag)

    history_notag = trainer_notag.train(
        train_data=[(train_x, train_y)], val_data=[(test_x, test_y)]
    )

    # Train with tag loss
    print("\n4. Training WITH tag-loss...")
    print("-" * 50)

    config_withtag = HybridTrainingConfig(
        learning_rate=0.01,
        max_epochs=30,
        use_adaptive_loss=True,
        target_coverage=0.85,
        use_tag_loss=True,
        lambda_tag=0.1,  # Tag loss weight
        verbose=False,
    )

    trainer_withtag = HybridTRTrainer(model=model_withtag, config=config_withtag)

    history_withtag = trainer_withtag.train(
        train_data=[(train_x, train_y)], val_data=[(test_x, test_y)]
    )

    # Compare results
    print("\n5. COMPARISON OF RESULTS")
    print("=" * 70)

    # Training metrics
    final_loss_notag = history_notag["loss"][-1] if "loss" in history_notag else float("inf")
    final_loss_withtag = history_withtag["loss"][-1] if "loss" in history_withtag else float("inf")

    print(f"\nFinal Training Loss:")
    print(f"  Without tag-loss:  {final_loss_notag:.6f}")
    print(f"  With tag-loss:     {final_loss_withtag:.6f}")

    # Tag loss contribution
    if "tag_loss" in history_withtag:
        avg_tag_loss = np.mean([tl for tl in history_withtag["tag_loss"] if tl > 0])
        print(f"\nAverage tag-loss contribution: {avg_tag_loss:.6f}")

    # Evaluate tag prediction accuracy
    print("\n6. TAG PREDICTION ACCURACY")
    print("-" * 50)

    if model_withtag.tag_head is not None:
        # Reset statistics
        model_withtag.reset_tag_statistics()

        # Evaluate on test set
        correct = 0
        total = 0
        tag_counts = {cls: 0 for cls in TagClass}
        tag_correct = {cls: 0 for cls in TagClass}

        for x, y in zip(test_x, test_y):
            x_node = TRNode.constant(x)
            y_pred, true_tag, tag_logits = model_withtag.forward_with_tag_pred(x_node)

            if tag_logits:
                pred_class, probs = model_withtag.tag_head.predict_tag(x_node)
                true_class = TagClass.from_tag(true_tag)

                tag_counts[true_class] += 1
                if pred_class == true_class:
                    correct += 1
                    tag_correct[true_class] += 1
                total += 1

        overall_accuracy = correct / total if total > 0 else 0
        print(f"\nOverall tag accuracy: {overall_accuracy:.3f}")

        print("\nPer-class accuracy:")
        for cls in TagClass:
            if tag_counts[cls] > 0:
                class_acc = tag_correct[cls] / tag_counts[cls]
                print(f"  {cls.name:6s}: {class_acc:.3f} ({tag_counts[cls]} samples)")

    # Coverage analysis
    print("\n7. COVERAGE ANALYSIS")
    print("-" * 50)

    def analyze_coverage(model, data_x, data_y):
        """Analyze output tag distribution."""
        tag_dist = {TRTag.REAL: 0, TRTag.PINF: 0, TRTag.NINF: 0, TRTag.PHI: 0}

        for x in data_x:
            if hasattr(model, "forward_with_tag_pred"):
                y, tag, _ = model.forward_with_tag_pred(x)
            else:
                y, tag = model.forward(x)
            tag_dist[tag] += 1

        total = sum(tag_dist.values())
        return {k: v / total for k, v in tag_dist.items()}

    cov_notag = analyze_coverage(model_notag, test_x, test_y)
    cov_withtag = analyze_coverage(model_withtag, test_x, test_y)

    print("\nOutput tag distribution (test set):")
    print("  Without tag-loss:")
    for tag, ratio in cov_notag.items():
        print(f"    {str(tag):10s}: {ratio:.3f}")

    print("  With tag-loss:")
    for tag, ratio in cov_withtag.items():
        print(f"    {str(tag):10s}: {ratio:.3f}")

    # Visualize tag-aware model
    print("\n8. Generating visualization...")
    visualize_tag_predictions(model_withtag, test_x, test_y)

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

    # Key insights
    print("\nKEY INSIGHTS:")
    print("• Tag-loss enables non-REAL samples to contribute supervision")
    print("• Model learns to classify singularity types (PINF/NINF/PHI)")
    print("• Auxiliary loss improves overall training stability")
    print("• Tag prediction accuracy indicates understanding of pole geometry")

    return trainer_notag, trainer_withtag, model_withtag


if __name__ == "__main__":
    # Run the demonstration
    trainer_notag, trainer_withtag, model_withtag = run_tag_loss_demo()

    print("\nTo explore further:")
    print("• Adjust lambda_tag to control tag-loss weight")
    print("• Increase tag_head_hidden_dim for more complex classification")
    print("• Combine with hybrid gradient schedule for maximum benefit")
    print("• Use tag predictions for uncertainty estimation")
