"""
Demonstration of pole detection head for Q≈0 localization.

This example shows how the pole detection head learns to identify
where the denominator Q(x) approaches zero, enabling explicit
pole localization with teacher signals or self-supervision.
"""

import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zeroproof.autodiff import GradientMode, GradientModeConfig, TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import FullyIntegratedRational, MonomialBasis, PoleAwareRational
from zeroproof.training import AdaptiveLossPolicy, HybridTrainingConfig, HybridTRTrainer, Optimizer
from zeroproof.training.pole_detection import DomainSpecificPoleDetector, PoleDetectionConfig

sys.stdout.reconfigure(encoding="utf-8")


def generate_rational_data(n: int = 100) -> Tuple[List[float], List[float], List[bool]]:
    """
    Generate synthetic rational function data with known poles.

    Target: y = (x + 1) / (x - 0.5)(x + 0.7)
    Poles at x = 0.5 and x = -0.7

    Returns:
        Tuple of (inputs, targets, pole_indicators)
    """
    # Generate points avoiding exact poles
    x_vals = []
    y_vals = []
    is_pole = []

    # Sample uniformly but avoid exact poles
    x_candidates = np.linspace(-2, 2, n * 2)

    for x in x_candidates:
        # Skip if too close to poles
        if abs(x - 0.5) < 0.01 or abs(x + 0.7) < 0.01:
            continue

        # Compute true value
        numerator = x + 1
        denominator = (x - 0.5) * (x + 0.7)
        y = numerator / denominator

        # Mark as near-pole if |Q| < threshold
        near_pole = abs(denominator) < 0.2

        x_vals.append(x)
        y_vals.append(y)
        is_pole.append(near_pole)

        if len(x_vals) >= n:
            break

    return x_vals[:n], y_vals[:n], is_pole[:n]


def create_teacher_detector() -> DomainSpecificPoleDetector:
    """
    Create a pole detector with analytical teacher signals.

    Returns:
        Configured pole detector
    """
    detector = DomainSpecificPoleDetector()

    # Define teacher function for known poles
    def is_near_pole(x: float) -> bool:
        # True poles at x = 0.5 and x = -0.7
        return abs(x - 0.5) < 0.15 or abs(x + 0.7) < 0.15

    detector.set_teacher_function(is_near_pole)
    return detector


def train_with_pole_detection():
    """Train a model with pole detection head."""

    print("=== Pole Detection Demo ===\n")

    # Generate data
    x_train, y_train, pole_labels = generate_rational_data(200)
    print(f"Generated {len(x_train)} training samples")
    print(f"Near-pole samples: {sum(pole_labels)}/{len(pole_labels)}")

    # Create model with pole detection
    basis = MonomialBasis()
    model = FullyIntegratedRational(
        d_p=3,
        d_q=2,
        basis=basis,
        enable_tag_head=True,
        enable_pole_head=True,
        pole_config=PoleDetectionConfig(
            hidden_dim=8,
            use_basis=True,
            proximity_threshold=0.2,
            teacher_weight=0.8,  # Prefer teacher signals
        ),
    )

    print(f"\nModel initialized with {len(model.parameters())} parameters")

    # Create trainer with pole detection enabled
    config = HybridTrainingConfig(
        learning_rate=0.01,
        max_epochs=50,
        use_tag_loss=True,
        lambda_tag=0.05,
        use_pole_head=True,
        lambda_pole=0.1,
        use_teacher_signals=True,
        track_pole_metrics=True,
    )

    # Initialize trainer
    trainer = HybridTRTrainer(
        model=model, optimizer=None, config=config  # Let trainer create optimizer automatically
    )

    # Set up teacher detector
    trainer.pole_detector = create_teacher_detector()

    # Training loop
    print("\nTraining with pole detection...")
    pole_accuracies = []
    losses = []

    for epoch in range(config.max_epochs):
        epoch_loss = 0.0
        pole_predictions = []
        true_poles = []

        # Mini-batch training
        batch_size = 20
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]
            batch_poles = pole_labels[i : i + batch_size]

            # Forward pass with full integration
            batch_loss = 0.0
            batch_pole_preds = []

            for x_val, y_val, is_pole in zip(batch_x, batch_y, batch_poles):
                x = TRNode.constant(real(x_val))
                result = model.forward_fully_integrated(x)

                # Compute main loss
                y_pred = result["output"]
                if result["tag"] == TRTag.REAL:
                    diff = y_pred - TRNode.constant(real(y_val))
                    loss = diff * diff
                else:
                    loss = TRNode.constant(real(1.0))  # Penalty for non-REAL

                batch_loss += loss.value.value if loss.value.tag == TRTag.REAL else 1.0

                # Track pole predictions
                pole_prob = result.get("pole_probability", 0.5)
                batch_pole_preds.append(pole_prob)
                pole_predictions.append(pole_prob > 0.5)
                true_poles.append(is_pole)

            epoch_loss += batch_loss

        # Compute pole detection accuracy
        correct = sum(1 for pred, true in zip(pole_predictions, true_poles) if pred == true)
        accuracy = correct / len(pole_predictions)
        pole_accuracies.append(accuracy)
        losses.append(epoch_loss / len(x_train))

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss={losses[-1]:.4f}, Pole Accuracy={accuracy:.2%}")

    print("\n=== Training Complete ===")

    # Evaluate pole detection
    print("\nEvaluating pole detection on test set...")
    x_test = np.linspace(-2, 2, 100)
    pole_scores = []

    for x_val in x_test:
        x = TRNode.constant(real(x_val))
        result = model.forward_fully_integrated(x)
        pole_scores.append(result.get("pole_probability", 0.5))

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Training curves
    axes[0, 0].plot(losses)
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(pole_accuracies)
    axes[0, 1].set_title("Pole Detection Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True)

    # Pole detection visualization
    axes[1, 0].plot(x_test, pole_scores, label="Predicted Pole Probability")
    axes[1, 0].axvline(x=0.5, color="r", linestyle="--", label="True Pole 1")
    axes[1, 0].axvline(x=-0.7, color="r", linestyle="--", label="True Pole 2")
    axes[1, 0].axhline(y=0.5, color="k", linestyle=":", alpha=0.5)
    axes[1, 0].fill_between(x_test, 0, pole_scores, alpha=0.3)
    axes[1, 0].set_title("Pole Detection Score")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("P(pole)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Function approximation
    y_pred = []
    tags = []
    for x_val in x_test:
        x = TRNode.constant(real(x_val))
        result = model.forward_fully_integrated(x)
        if result["tag"] == TRTag.REAL:
            y_pred.append(result["output"].value.value)
        else:
            y_pred.append(np.nan)
        tags.append(result["tag"])

    # Plot predictions
    axes[1, 1].plot(x_test, y_pred, "b-", label="Predicted", alpha=0.7)

    # Overlay true function (avoiding poles)
    y_true = []
    for x in x_test:
        if abs(x - 0.5) < 0.05 or abs(x + 0.7) < 0.05:
            y_true.append(np.nan)
        else:
            y_true.append((x + 1) / ((x - 0.5) * (x + 0.7)))

    axes[1, 1].plot(x_test, y_true, "r--", label="True", alpha=0.5)
    axes[1, 1].axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    axes[1, 1].axvline(x=-0.7, color="gray", linestyle=":", alpha=0.5)
    axes[1, 1].set_title("Function Approximation")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].set_ylim(-10, 10)
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("pole_detection_results.png", dpi=150)
    # plt.show()  # Commented out to avoid hanging in batch mode

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Final pole detection accuracy: {pole_accuracies[-1]:.2%}")

    # Check detected pole locations
    detected_poles = [x_test[i] for i, p in enumerate(pole_scores) if p > 0.8]
    if detected_poles:
        print(f"Detected pole regions (P > 0.8): {detected_poles[:5]}...")

        # Check distance to true poles
        pole1_dist = min(abs(x - 0.5) for x in detected_poles) if detected_poles else np.inf
        pole2_dist = min(abs(x + 0.7) for x in detected_poles) if detected_poles else np.inf
        print(f"Distance to pole 1 (x=0.5): {pole1_dist:.3f}")
        print(f"Distance to pole 2 (x=-0.7): {pole2_dist:.3f}")

    # Get integration summary
    summary = model.get_integration_summary()
    print("\n=== Model Integration Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return model, pole_accuracies


def compare_with_without_pole_head():
    """Compare training with and without pole detection head."""

    print("\n=== Comparison: With vs Without Pole Head ===\n")

    # Generate data
    x_train, y_train, pole_labels = generate_rational_data(100)

    # Train without pole head
    print("Training WITHOUT pole head...")
    model_without = PoleAwareRational(d_p=3, d_q=2, basis=MonomialBasis(), enable_pole_head=False)

    optimizer = Optimizer(model_without.parameters(), learning_rate=0.01)
    losses_without = []

    for epoch in range(30):
        epoch_loss = 0.0
        for x_val, y_val in zip(x_train, y_train):
            x = TRNode.constant(real(x_val))
            y_pred, tag = model_without.forward(x)

            if tag == TRTag.REAL:
                diff = y_pred - TRNode.constant(real(y_val))
                loss = diff * diff
                loss.backward()
                optimizer.step(model_without)
                epoch_loss += loss.value.value
            else:
                epoch_loss += 1.0

        losses_without.append(epoch_loss / len(x_train))

    # Train with pole head
    print("Training WITH pole head...")
    model_with = PoleAwareRational(
        d_p=3,
        d_q=2,
        basis=MonomialBasis(),
        enable_pole_head=True,
        pole_config=PoleDetectionConfig(hidden_dim=8),
    )

    optimizer = Optimizer(model_with.parameters(), learning_rate=0.01)
    losses_with = []

    for epoch in range(30):
        epoch_loss = 0.0
        for x_val, y_val, is_pole in zip(x_train, y_train, pole_labels):
            x = TRNode.constant(real(x_val))
            y_pred, tag, pole_score = model_with.forward_with_pole_score(x)

            # Main loss
            if tag == TRTag.REAL:
                diff = y_pred - TRNode.constant(real(y_val))
                loss = diff * diff
            else:
                loss = TRNode.constant(real(1.0))

            # Add pole loss (simplified MSE)
            # For binary classification: loss = (prediction - target)^2
            target = 1.0 if is_pole else 0.0
            pole_diff = pole_score - TRNode.constant(real(target))
            pole_loss = pole_diff * pole_diff * TRNode.constant(real(0.1))

            total_loss = loss + pole_loss
            total_loss.backward()
            optimizer.step(model_with)

            epoch_loss += total_loss.value.value if total_loss.value.tag == TRTag.REAL else 1.0

        losses_with.append(epoch_loss / len(x_train))

    # Plot comparison
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses_without, label="Without Pole Head", linewidth=2)
    plt.plot(losses_with, label="With Pole Head", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)

    # Evaluate on test points near poles
    x_near_poles = [0.4, 0.45, 0.55, 0.6, -0.8, -0.75, -0.65, -0.6]

    plt.subplot(1, 2, 2)
    errors_without = []
    errors_with = []

    for x_val in x_near_poles:
        # True value
        y_true = (x_val + 1) / ((x_val - 0.5) * (x_val + 0.7))

        # Without pole head
        x = TRNode.constant(real(x_val))
        y_pred, tag = model_without.forward(x)
        if tag == TRTag.REAL:
            errors_without.append(abs(y_pred.value.value - y_true))
        else:
            errors_without.append(10.0)  # Large error for non-REAL

        # With pole head
        y_pred, tag, _ = model_with.forward_with_pole_score(x)
        if tag == TRTag.REAL:
            errors_with.append(abs(y_pred.value.value - y_true))
        else:
            errors_with.append(10.0)

    x_pos = np.arange(len(x_near_poles))
    width = 0.35

    plt.bar(x_pos - width / 2, errors_without, width, label="Without Pole Head", alpha=0.7)
    plt.bar(x_pos + width / 2, errors_with, width, label="With Pole Head", alpha=0.7)
    plt.xlabel("Test Points Near Poles")
    plt.ylabel("Absolute Error")
    plt.title("Prediction Error Near Poles")
    plt.xticks(x_pos, [f"{x:.2f}" for x in x_near_poles], rotation=45)
    plt.legend()
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("pole_head_comparison.png", dpi=150)
    # plt.show()  # Commented out to avoid hanging in batch mode

    print(f"\nFinal loss without pole head: {losses_without[-1]:.4f}")
    print(f"Final loss with pole head: {losses_with[-1]:.4f}")
    print(f"Average error near poles without head: {np.mean(errors_without):.3f}")
    print(f"Average error near poles with head: {np.mean(errors_with):.3f}")


if __name__ == "__main__":
    # Set gradient mode
    GradientModeConfig.set_mode(GradientMode.MASK_REAL)

    # Run main demo
    model, accuracies = train_with_pole_detection()

    # Run comparison
    compare_with_without_pole_head()

    print("\n=== Demo Complete ===")
    print("Results saved to:")
    print("  - pole_detection_results.png")
    print("  - pole_head_comparison.png")
