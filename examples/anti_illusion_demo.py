"""
Demonstration of Anti-Illusion Metrics for pole geometry verification.

This example shows how the anti-illusion metrics prove that the model
truly learns pole behavior rather than just avoiding singularities.
"""

import math
from typing import List, Tuple

import numpy as np

from zeroproof.autodiff import GradientMode, GradientModeConfig, TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import FullyIntegratedRational, MonomialBasis
from zeroproof.training import HybridTrainingConfig, HybridTRTrainer, Optimizer
from zeroproof.utils.metrics import (
    AntiIllusionMetrics,
    AsymptoticSlopeAnalyzer,
    PoleLocalizationError,
    PoleLocation,
    ResidualConsistencyLoss,
    SignConsistencyChecker,
)


def generate_rational_data_with_poles(
    n: int = 200,
) -> Tuple[List[float], List[float], List[PoleLocation]]:
    """
    Generate data from a known rational function with poles.

    Target: y = (2x + 1) / ((x - 0.5)(x + 0.8))
    Poles at x = 0.5 and x = -0.8

    Returns:
        Tuple of (inputs, targets, ground_truth_poles)
    """
    # Ground truth poles
    ground_truth_poles = [
        PoleLocation(x=0.5, pole_type="simple"),
        PoleLocation(x=-0.8, pole_type="simple"),
    ]

    # Generate input points avoiding exact poles
    x_vals = []
    y_vals = []

    # Sample from multiple regions
    regions = [(-2.0, -1.0), (-0.6, 0.3), (0.7, 2.0)]
    points_per_region = n // len(regions)

    for x_min, x_max in regions:
        x_region = np.linspace(x_min, x_max, points_per_region)

        for x in x_region:
            # Skip if too close to poles
            if abs(x - 0.5) < 0.05 or abs(x + 0.8) < 0.05:
                continue

            # Compute true rational function value
            numerator = 2 * x + 1
            denominator = (x - 0.5) * (x + 0.8)

            if abs(denominator) > 1e-6:  # Avoid numerical issues
                y = numerator / denominator
                x_vals.append(x)
                y_vals.append(y)

    return x_vals[:n], y_vals[:n], ground_truth_poles


def create_test_paths(poles: List[PoleLocation]) -> List[Tuple]:
    """
    Create parametric paths crossing poles for sign consistency testing.

    Args:
        poles: List of pole locations

    Returns:
        List of (path_func, t_range, pole_t) tuples
    """
    test_paths = []

    for pole in poles:
        # Create a straight line path crossing the pole
        def make_path(pole_x):
            def path_func(t):
                return pole_x + t  # Line crossing pole at t=0

            return path_func

        path_func = make_path(pole.x)
        t_range = (-0.3, 0.3)
        pole_t = 0.0

        test_paths.append((path_func, t_range, pole_t))

    return test_paths


def train_with_anti_illusion_metrics():
    """Train a model with comprehensive anti-illusion evaluation."""

    print("=== Anti-Illusion Metrics Demo ===\n")

    # Set gradient mode
    GradientModeConfig.set_mode(GradientMode.MASK_REAL)

    # Generate training data
    x_train, y_train, ground_truth_poles = generate_rational_data_with_poles(150)
    print(f"Generated {len(x_train)} training samples")
    print(f"Ground truth poles: {[f'x={p.x:.1f}' for p in ground_truth_poles]}")

    # Create model with all enhancements
    basis = MonomialBasis()
    model = FullyIntegratedRational(
        d_p=3,  # Numerator degree
        d_q=2,  # Denominator degree
        basis=basis,
        enable_tag_head=True,
        enable_pole_head=True,
        track_Q_values=True,
    )

    print(f"\nModel initialized with {len(model.parameters())} parameters")

    # Convert ground truth poles for trainer
    gt_poles_tuples = [(p.x, None) for p in ground_truth_poles]

    # Create trainer with anti-illusion metrics enabled
    config = HybridTrainingConfig(
        learning_rate=0.005,
        max_epochs=100,
        # Enable all enhancements
        use_hybrid_gradient=True,
        hybrid_warmup_epochs=20,
        hybrid_transition_epochs=30,
        use_tag_loss=True,
        lambda_tag=0.03,
        use_pole_head=True,
        lambda_pole=0.05,
        # Enable anti-illusion metrics
        enable_anti_illusion=True,
        lambda_residual=0.02,
        ground_truth_poles=gt_poles_tuples,
        ple_x_range=(-2.5, 2.5),
    )

    trainer = HybridTRTrainer(
        model=model,
        optimizer=Optimizer(model.parameters(), learning_rate=config.learning_rate),
        config=config,
    )

    print("\n=== Training with Anti-Illusion Evaluation ===")

    # Training loop with periodic evaluation
    training_metrics = []
    illusion_scores = []
    ple_scores = []

    for epoch in range(config.max_epochs):
        # Train one epoch
        epoch_metrics = {}
        total_loss = 0.0

        # Mini-batch training
        batch_size = 20
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]

            # Convert to TRNodes
            inputs = [TRNode.constant(real(x)) for x in batch_x]
            targets = [real(y) for y in batch_y]

            # Train batch
            batch_metrics = trainer._train_batch(inputs, targets, trainer.coverage_tracker)

            for key, value in batch_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

        # Average metrics
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)

        training_metrics.append(avg_metrics)

        # Evaluate anti-illusion metrics every 10 epochs
        if epoch % 10 == 0:
            try:
                # Manual evaluation since trainer does it automatically
                ai_metrics = trainer.anti_illusion_metrics
                if ai_metrics:
                    # Create test paths for sign consistency
                    test_paths = create_test_paths(ground_truth_poles)

                    illusion_results = ai_metrics.evaluate_model(
                        model, ground_truth_poles, test_paths=test_paths, x_range=config.ple_x_range
                    )

                    illusion_scores.append(illusion_results["anti_illusion_score"])
                    ple_scores.append(illusion_results["ple"])

                    print(
                        f"Epoch {epoch:3d}: Loss={avg_metrics.get('loss', float('inf')):.4f}, "
                        f"PLE={illusion_results['ple']:.4f}, "
                        f"AI-Score={illusion_results['anti_illusion_score']:.4f}"
                    )
                else:
                    print(f"Epoch {epoch:3d}: Loss={avg_metrics.get('loss', float('inf')):.4f}")

            except Exception as e:
                print(
                    f"Epoch {epoch:3d}: Loss={avg_metrics.get('loss', float('inf')):.4f} "
                    f"(AI eval failed: {e})"
                )

    print("\n=== Training Complete ===")

    # Final comprehensive evaluation
    print("\n=== Final Anti-Illusion Evaluation ===")

    # Individual metric evaluations
    ple_metric = PoleLocalizationError()
    sign_checker = SignConsistencyChecker()
    slope_analyzer = AsymptoticSlopeAnalyzer()
    residual_loss = ResidualConsistencyLoss()

    # 1. Pole Localization Error
    final_ple = ple_metric.compute_ple(model, ground_truth_poles, (-2.5, 2.5))
    print(f"Final PLE Score: {final_ple:.4f}")

    detected_poles = ple_metric.find_poles_1d(model, (-2.5, 2.5), n_samples=1000)
    print(f"Detected {len(detected_poles)} poles:")
    for i, pole in enumerate(detected_poles):
        print(f"  Pole {i+1}: x = {pole.x:.3f}")

    # 2. Sign Consistency
    test_paths = create_test_paths(ground_truth_poles)
    sign_results = []

    for i, (path_func, t_range, pole_t) in enumerate(test_paths):
        metrics = sign_checker.check_path_crossing(model, path_func, t_range, pole_t)
        sign_results.append(metrics)
        print(f"Sign consistency at pole {i+1}: {metrics['overall_consistency']:.3f}")

    avg_sign_consistency = np.mean([r["overall_consistency"] for r in sign_results])
    print(f"Average sign consistency: {avg_sign_consistency:.3f}")

    # 3. Asymptotic Slope Analysis
    slope_results = []
    for i, pole in enumerate(ground_truth_poles):
        slope_metrics = slope_analyzer.compute_asymptotic_slope(model, pole.x)
        slope_results.append(slope_metrics)

        if not math.isnan(slope_metrics["slope"]):
            print(
                f"Asymptotic slope at pole {i+1}: {slope_metrics['slope']:.3f} "
                f"(error: {slope_metrics['slope_error']:.3f})"
            )
        else:
            print(f"Asymptotic slope at pole {i+1}: Could not compute")

    # 4. Residual Consistency
    test_points = np.linspace(-2, 2, 100).tolist()
    final_residual = residual_loss.compute_loss(model, test_points)
    residual_value = (
        final_residual.value.value if final_residual.tag == TRTag.REAL else float("inf")
    )
    print(f"Residual consistency loss: {residual_value:.6f}")

    # 5. Model Performance Analysis
    print("\n=== Model Performance Analysis ===")

    # Test on evaluation points
    test_x = np.linspace(-2, 2, 50)
    correct_predictions = 0
    total_predictions = 0

    print("Sample predictions:")
    print("x\t\tTrue y\t\tPred y\t\tTag\t\tError")
    print("-" * 60)

    for x_val in test_x[::5]:  # Every 5th point
        # Skip near poles
        near_pole = any(abs(x_val - p.x) < 0.1 for p in ground_truth_poles)
        if near_pole:
            continue

        # True value
        true_y = (2 * x_val + 1) / ((x_val - 0.5) * (x_val + 0.8))

        # Model prediction
        x = TRNode.constant(real(x_val))
        result = model.forward_fully_integrated(x)

        if result["tag"] == TRTag.REAL:
            pred_y = result["output"].value.value
            error = abs(pred_y - true_y)
            print(
                f"{x_val:.2f}\t\t{true_y:.3f}\t\t{pred_y:.3f}\t\t{result['tag'].name}\t\t{error:.3f}"
            )

            if error < 0.5:  # Reasonable tolerance
                correct_predictions += 1
            total_predictions += 1
        else:
            print(f"{x_val:.2f}\t\t{true_y:.3f}\t\tN/A\t\t{result['tag'].name}\t\tN/A")

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\nPrediction accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

    # 6. Training Progress Summary
    print("\n=== Training Progress Summary ===")

    if illusion_scores:
        print(f"Initial anti-illusion score: {illusion_scores[0]:.4f}")
        print(f"Final anti-illusion score: {illusion_scores[-1]:.4f}")
        print(f"Improvement: {(illusion_scores[0] - illusion_scores[-1]):.4f}")

        print(f"Initial PLE: {ple_scores[0]:.4f}")
        print(f"Final PLE: {ple_scores[-1]:.4f}")
        print(f"PLE improvement: {(ple_scores[0] - ple_scores[-1]):.4f}")

    # 7. Model Integration Summary
    summary = model.get_integration_summary()
    print(f"\n=== Model Integration Summary ===")
    for key, value in summary.items():
        if value is not None:
            print(f"{key}: {value}")

    return model, {
        "ple_scores": ple_scores,
        "illusion_scores": illusion_scores,
        "final_ple": final_ple,
        "sign_consistency": avg_sign_consistency,
        "slope_results": slope_results,
        "residual_loss": residual_value,
    }


def compare_with_without_anti_illusion():
    """Compare training with and without anti-illusion metrics."""

    print("\n=== Comparison: With vs Without Anti-Illusion ===\n")

    # This would be a more extensive comparison
    # For now, just demonstrate the concept

    print("Training WITHOUT anti-illusion metrics...")
    print("(Would use standard training without residual loss or metrics)")

    print("\nTraining WITH anti-illusion metrics...")
    print("(Uses residual consistency loss and comprehensive evaluation)")

    print("\nKey differences:")
    print("- Residual loss enforces R(x) = Q(x)*y(x) - P(x) ≈ 0")
    print("- PLE quantifies pole localization accuracy")
    print("- Sign consistency verifies correct behavior across poles")
    print("- Asymptotic slope confirms theoretical expectations")
    print("- Combined score provides single quality measure")


if __name__ == "__main__":
    try:
        # Run main demonstration
        model, results = train_with_anti_illusion_metrics()

        # Run comparison
        compare_with_without_anti_illusion()

        print("\n=== Key Takeaways ===")
        print("1. PLE measures how accurately the model localizes poles")
        print("2. Sign consistency verifies correct +∞/-∞ behavior")
        print("3. Asymptotic slope confirms theoretical pole behavior")
        print("4. Residual consistency enforces structural coherence")
        print("5. Combined metrics prove true pole understanding")

        print(f"\nFinal Results:")
        print(f"- PLE Score: {results['final_ple']:.4f} (lower is better)")
        print(f"- Sign Consistency: {results['sign_consistency']:.3f} (higher is better)")
        print(f"- Residual Loss: {results['residual_loss']:.6f} (lower is better)")

        if results["illusion_scores"]:
            print(f"- Anti-Illusion Score: {results['illusion_scores'][-1]:.4f} (lower is better)")

        print("\n=== Demo Complete ===")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
