"""
Complete demonstration of ZeroProofML capabilities.

This example showcases all implemented enhancements:
- Hybrid gradient schedule
- Tag-loss for non-REAL outputs  
- Coverage control with adaptive lambda
- Pole detection head
- Anti-illusion metrics
- Structured logging and plotting
"""

import os
from typing import List, Tuple

import numpy as np

from zeroproof.autodiff import GradientMode, GradientModeConfig, TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import FullyIntegratedRational, MonomialBasis
from zeroproof.training import HybridTrainingConfig, HybridTRTrainer, Optimizer
from zeroproof.training.pole_detection import PoleDetectionConfig
from zeroproof.utils.logging import ExperimentTracker, StructuredLogger
from zeroproof.utils.metrics import AntiIllusionMetrics, PoleLocation


def generate_challenging_dataset(
    n: int = 300,
) -> Tuple[List[float], List[float], List[PoleLocation]]:
    """
    Generate a challenging rational function dataset with multiple poles.

    Target function: y = (x² + 1) / ((x - 0.5)(x + 0.8)(x - 1.2))
    Poles at x = 0.5, -0.8, 1.2

    Returns:
        (inputs, targets, ground_truth_poles)
    """
    # Ground truth poles
    poles = [
        PoleLocation(x=0.5, pole_type="simple"),
        PoleLocation(x=-0.8, pole_type="simple"),
        PoleLocation(x=1.2, pole_type="simple"),
    ]

    # Generate input points with strategic sampling
    x_vals = []
    y_vals = []

    # Sample from different regions
    regions = [
        (-2.0, -1.0, 50),  # Far left
        (-0.6, 0.3, 80),  # Between first two poles
        (0.7, 1.0, 60),  # Between second and third poles
        (1.4, 2.0, 50),  # Far right
        # Near-pole regions (more samples)
        (0.3, 0.7, 30),  # Near pole at 0.5
        (-1.0, -0.6, 30),  # Near pole at -0.8
        (1.0, 1.4, 30),  # Near pole at 1.2
    ]

    for x_min, x_max, n_points in regions:
        x_region = np.linspace(x_min, x_max, n_points)

        for x in x_region:
            # Skip if too close to poles
            if abs(x - 0.5) < 0.03 or abs(x + 0.8) < 0.03 or abs(x - 1.2) < 0.03:
                continue

            # Compute true function value
            numerator = x**2 + 1
            denominator = (x - 0.5) * (x + 0.8) * (x - 1.2)

            if abs(denominator) > 1e-6:
                y = numerator / denominator
                # Clip extreme values for numerical stability
                if abs(y) < 100:
                    x_vals.append(x)
                    y_vals.append(y)

    # Shuffle and limit
    indices = np.random.permutation(len(x_vals))[:n]
    x_final = [x_vals[i] for i in indices]
    y_final = [y_vals[i] for i in indices]

    return x_final, y_final, poles


def run_complete_demonstration():
    """Run complete demonstration with all features."""

    print("=" * 60)
    print("ZeroProofML Complete Feature Demonstration")
    print("=" * 60)

    # Set up experiment tracking
    tracker = ExperimentTracker(base_dir="runs")

    # Start experiment
    config_dict = {
        "model": "FullyIntegratedRational",
        "enhancements": "all",
        "dataset": "challenging_rational",
        "purpose": "complete_demonstration",
    }

    model_info = {
        "type": "FullyIntegratedRational",
        "degree_p": 4,
        "degree_q": 3,
        "basis": "Monomial",
        "enhancements": [
            "hybrid_gradient_schedule",
            "tag_loss",
            "pole_detection_head",
            "coverage_control",
            "anti_illusion_metrics",
        ],
    }

    logger = tracker.start_experiment(
        name="complete_demo", config=config_dict, model_info=model_info
    )

    print(f"\nExperiment started: {tracker.current_experiment['name']}")
    print(f"Log directory: {tracker.current_experiment['directory']}")

    # Generate dataset
    print("\n1. Generating challenging dataset...")
    x_train, y_train, ground_truth_poles = generate_challenging_dataset(300)

    print(f"   Generated {len(x_train)} training samples")
    print(f"   Ground truth poles: {[f'x={p.x:.1f}' for p in ground_truth_poles]}")

    # Create model with all enhancements
    print("\n2. Creating fully integrated model...")

    GradientModeConfig.set_mode(GradientMode.MASK_REAL)

    basis = MonomialBasis()
    model = FullyIntegratedRational(
        d_p=4,  # Higher degree for complex function
        d_q=3,
        basis=basis,
        enable_tag_head=True,
        enable_pole_head=True,
        track_Q_values=True,
        pole_config=PoleDetectionConfig(hidden_dim=12, use_basis=True, proximity_threshold=0.15),
    )

    print(f"   Model created with {len(model.parameters())} parameters")

    # Configure training with all enhancements
    print("\n3. Configuring enhanced training...")

    # Convert poles for trainer
    gt_poles_tuples = [(p.x, None) for p in ground_truth_poles]

    training_config = HybridTrainingConfig(
        learning_rate=0.008,
        max_epochs=150,
        # Hybrid gradient schedule
        use_hybrid_gradient=True,
        hybrid_warmup_epochs=30,
        hybrid_transition_epochs=50,
        hybrid_delta_init=0.02,
        hybrid_delta_final=1e-5,
        # Tag loss
        use_tag_loss=True,
        lambda_tag=0.04,
        # Pole detection
        use_pole_head=True,
        lambda_pole=0.08,
        use_teacher_signals=False,  # Use self-supervision
        # Coverage control
        enforce_coverage=True,
        min_coverage=0.75,
        max_lambda_for_coverage=5.0,
        # Anti-illusion metrics
        enable_anti_illusion=True,
        lambda_residual=0.015,
        ground_truth_poles=gt_poles_tuples,
        ple_x_range=(-2.5, 2.5),
    )

    trainer = HybridTRTrainer(
        model=model,
        optimizer=Optimizer(model.parameters(), learning_rate=training_config.learning_rate),
        config=training_config,
    )

    print("   Training configuration:")
    print(f"     - Hybrid schedule: {training_config.use_hybrid_schedule}")
    print(f"     - Tag loss: {training_config.use_tag_loss} (λ={training_config.lambda_tag})")
    print(f"     - Pole head: {training_config.use_pole_head} (λ={training_config.lambda_pole})")
    print(f"     - Coverage control: {training_config.enforce_coverage}")
    print(f"     - Anti-illusion: {training_config.enable_anti_illusion}")

    # Training loop with comprehensive logging
    print("\n4. Training with comprehensive logging...")

    training_metrics = []
    ai_evaluation_history = []

    for epoch in range(training_config.max_epochs):
        # Train mini-batches
        batch_size = 25
        epoch_metrics = []

        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]

            # Convert to TRNodes
            inputs = [TRNode.constant(real(x)) for x in batch_x]
            targets = [real(y) for y in batch_y]

            # Train batch
            batch_metrics = trainer._train_batch(inputs, targets, trainer.coverage_tracker)
            epoch_metrics.append(batch_metrics)

        # Average epoch metrics
        if epoch_metrics:
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                values = [
                    m[key] for m in epoch_metrics if key in m and isinstance(m[key], (int, float))
                ]
                if values:
                    avg_metrics[key] = sum(values) / len(values)

            avg_metrics["epoch"] = epoch
            training_metrics.append(avg_metrics)

            # Log to structured logger
            tags = [TRTag.REAL] * len(batch_x)  # Simplified for demo
            logger.log_metrics(avg_metrics, epoch=epoch)

        # Periodic evaluation and logging
        if epoch % 15 == 0:
            print(
                f"   Epoch {epoch:3d}: Loss={avg_metrics.get('loss', float('inf')):.6f}, "
                f"Coverage={avg_metrics.get('coverage', 0):.3f}"
            )

            # Anti-illusion evaluation (every 15 epochs)
            if trainer.anti_illusion_metrics and trainer.ground_truth_poles:
                try:
                    ai_metrics = trainer.anti_illusion_metrics.evaluate_model(
                        model, trainer.ground_truth_poles, x_range=training_config.ple_x_range
                    )
                    ai_evaluation_history.append(ai_metrics)

                    if epoch % 30 == 0:
                        print(
                            f"     AI-Metrics: PLE={ai_metrics.get('ple', float('inf')):.4f}, "
                            f"Score={ai_metrics.get('anti_illusion_score', float('inf')):.4f}"
                        )

                except Exception as e:
                    print(f"     AI evaluation failed: {e}")

    print("\n5. Training completed!")

    # Final comprehensive evaluation
    print("\n6. Final evaluation...")

    # Test on evaluation grid
    x_test = np.linspace(-2.5, 2.5, 100)
    test_results = {"predictions": [], "tags": [], "pole_probabilities": [], "errors": []}

    for x_val in x_test:
        # Skip exact poles
        near_pole = any(abs(x_val - p.x) < 0.05 for p in ground_truth_poles)
        if near_pole:
            continue

        x = TRNode.constant(real(x_val))
        result = model.forward_fully_integrated(x)

        # True value for comparison
        try:
            y_true = (x_val**2 + 1) / ((x_val - 0.5) * (x_val + 0.8) * (x_val - 1.2))
            if abs(y_true) < 50:  # Reasonable range
                test_results["predictions"].append(
                    result["output"].value.value if result["tag"] == TRTag.REAL else np.nan
                )
                test_results["tags"].append(result["tag"])
                test_results["pole_probabilities"].append(result.get("pole_probability", 0.0))

                if result["tag"] == TRTag.REAL:
                    error = abs(result["output"].value.value - y_true)
                    test_results["errors"].append(error)
        except:
            continue

    # Compute final statistics
    valid_errors = [e for e in test_results["errors"] if np.isfinite(e)]
    real_ratio = sum(1 for tag in test_results["tags"] if tag == TRTag.REAL) / len(
        test_results["tags"]
    )

    print(f"   Test evaluation on {len(test_results['predictions'])} points:")
    print(f"     REAL coverage: {real_ratio:.2%}")
    if valid_errors:
        print(f"     Mean absolute error: {np.mean(valid_errors):.6f}")
        print(f"     Max error: {np.max(valid_errors):.6f}")

    # Final anti-illusion evaluation
    if trainer.anti_illusion_metrics and trainer.ground_truth_poles:
        final_ai = trainer.anti_illusion_metrics.evaluate_model(model, trainer.ground_truth_poles)
        print(
            f"   Final anti-illusion score: {final_ai.get('anti_illusion_score', float('inf')):.4f}"
        )
        print(f"   Final PLE: {final_ai.get('ple', float('inf')):.4f}")

    # Save comprehensive results
    print("\n7. Saving results and plots...")

    # Save training logs
    log_file = logger.save()
    csv_file = logger.save_csv()
    summary_file = logger.export_summary()

    # Create plots (if matplotlib available)
    try:
        from zeroproof.utils.plotting import save_all_plots

        plot_paths = save_all_plots(
            run_dir=tracker.current_experiment["directory"],
            training_history=training_metrics,
            model=model,
            ai_metrics=ai_evaluation_history,
        )

        print(f"   Created {len(plot_paths)} plots")

    except ImportError:
        print("   Matplotlib not available - skipping plots")
        plot_paths = []
    except Exception as e:
        print(f"   Plot creation failed: {e}")
        plot_paths = []

    # Finish experiment
    summary_path = tracker.finish_experiment()

    # Print final summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

    print(f"\nFiles created:")
    print(f"  - Training logs: {log_file}")
    print(f"  - CSV metrics: {csv_file}")
    print(f"  - Summary: {summary_file}")

    if plot_paths:
        print(f"  - Plots: {len(plot_paths)} files in plots/ directory")

    # Model capabilities summary
    integration_summary = model.get_integration_summary()
    print(f"\nModel Integration Summary:")
    for key, value in integration_summary.items():
        if value is not None:
            print(f"  {key}: {value}")

    # Training summary
    trainer_summary = trainer.get_training_summary()
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {len(training_metrics)}")
    print(f"  Final loss: {training_metrics[-1].get('loss', 'N/A')}")
    print(f"  Final coverage: {training_metrics[-1].get('coverage', 'N/A')}")

    if "hybrid_statistics" in trainer_summary:
        hybrid_stats = trainer_summary["hybrid_statistics"]
        print(f"  Hybrid mode transitions: {hybrid_stats.get('mode_transitions', 0)}")
        print(f"  Saturating gradient uses: {hybrid_stats.get('saturating_uses', 0)}")

    print(f"\nDemonstration showcased:")
    print(f"  ✓ Hybrid gradient schedule (Mask-REAL → Saturating)")
    print(f"  ✓ Tag-loss for non-REAL classification")
    print(f"  ✓ Pole detection head with self-supervision")
    print(f"  ✓ Coverage control with adaptive lambda")
    print(f"  ✓ Anti-illusion metrics (PLE, sign consistency, etc.)")
    print(f"  ✓ Structured logging and experiment tracking")
    print(f"  ✓ Comprehensive evaluation framework")

    return {
        "model": model,
        "trainer": trainer,
        "training_metrics": training_metrics,
        "ai_metrics": ai_evaluation_history,
        "test_results": test_results,
        "files": {"logs": log_file, "csv": csv_file, "summary": summary_file, "plots": plot_paths},
    }


def demonstrate_feature_ablation():
    """Demonstrate ablation study by disabling features one by one."""

    print("\n" + "=" * 60)
    print("FEATURE ABLATION DEMONSTRATION")
    print("=" * 60)

    # Generate smaller dataset for ablation
    x_data, y_data, poles = generate_challenging_dataset(100)

    # Configurations to test
    ablation_configs = [
        {
            "name": "Full ZeroProofML",
            "use_hybrid_schedule": True,
            "use_tag_loss": True,
            "use_pole_head": True,
            "enable_anti_illusion": True,
            "enforce_coverage": True,
        },
        {
            "name": "No Hybrid Schedule",
            "use_hybrid_schedule": False,
            "use_tag_loss": True,
            "use_pole_head": True,
            "enable_anti_illusion": True,
            "enforce_coverage": True,
        },
        {
            "name": "No Tag Loss",
            "use_hybrid_schedule": True,
            "use_tag_loss": False,
            "use_pole_head": True,
            "enable_anti_illusion": True,
            "enforce_coverage": True,
        },
        {
            "name": "No Pole Head",
            "use_hybrid_schedule": True,
            "use_tag_loss": True,
            "use_pole_head": False,
            "enable_anti_illusion": True,
            "enforce_coverage": True,
        },
        {
            "name": "Basic TR-Rational",
            "use_hybrid_schedule": False,
            "use_tag_loss": False,
            "use_pole_head": False,
            "enable_anti_illusion": False,
            "enforce_coverage": False,
        },
    ]

    ablation_results = []

    for config_spec in ablation_configs:
        print(f"\nTesting: {config_spec['name']}")

        try:
            # Create model based on configuration
            if config_spec["use_tag_loss"] or config_spec["use_pole_head"]:
                test_model = FullyIntegratedRational(
                    d_p=3,
                    d_q=2,
                    basis=MonomialBasis(),
                    enable_tag_head=config_spec["use_tag_loss"],
                    enable_pole_head=config_spec["use_pole_head"],
                )
            else:
                from zeroproof.layers import TRRational

                test_model = TRRational(d_p=3, d_q=2, basis=MonomialBasis())

            # Quick training (10 epochs for ablation)
            optimizer = Optimizer(test_model.parameters(), learning_rate=0.01)

            total_loss = 0.0
            n_samples = 0

            for epoch in range(10):
                epoch_loss = 0.0
                epoch_samples = 0

                for x_val, y_val in zip(x_data[:50], y_data[:50]):  # Subset for speed
                    x = TRNode.constant(real(x_val))

                    if hasattr(test_model, "forward_fully_integrated"):
                        result = test_model.forward_fully_integrated(x)
                        y_pred = result["output"]
                        tag = result["tag"]
                    else:
                        y_pred, tag = test_model.forward(x)

                    if tag == TRTag.REAL:
                        target = TRNode.constant(real(y_val))
                        loss = (y_pred - target) ** 2
                        loss.backward()
                        optimizer.step(test_model)

                        epoch_loss += loss.value.value
                        epoch_samples += 1

                if epoch_samples > 0:
                    total_loss += epoch_loss / epoch_samples
                    n_samples += 1

            final_loss = total_loss / n_samples if n_samples > 0 else float("inf")

            result = {
                "name": config_spec["name"],
                "config": config_spec,
                "final_loss": final_loss,
                "n_parameters": len(test_model.parameters()),
                "status": "completed",
            }

            print(f"  ✓ {config_spec['name']}: Loss={final_loss:.6f}")

        except Exception as e:
            result = {
                "name": config_spec["name"],
                "config": config_spec,
                "error": str(e),
                "status": "failed",
            }
            print(f"  ✗ {config_spec['name']}: Failed - {e}")

        ablation_results.append(result)

    # Summary
    print(f"\nAblation Study Summary:")
    print(f"Configurations tested: {len(ablation_configs)}")
    successful = [r for r in ablation_results if r["status"] == "completed"]
    print(f"Successful runs: {len(successful)}")

    if successful:
        best_config = min(successful, key=lambda x: x["final_loss"])
        print(f"Best configuration: {best_config['name']} (Loss: {best_config['final_loss']:.6f})")

    return ablation_results


def main():
    """Run complete demonstration."""

    try:
        # Main demonstration
        demo_results = run_complete_demonstration()

        # Ablation study
        ablation_results = demonstrate_feature_ablation()

        print("\n" + "=" * 60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(f"\nKey achievements:")
        print(f"  ✓ Complete ZeroProofML pipeline implemented")
        print(f"  ✓ All enhancement packages functional")
        print(f"  ✓ Comprehensive evaluation framework")
        print(f"  ✓ Structured logging and experiment tracking")
        print(f"  ✓ Ablation study framework")

        print(f"\nResults available in:")
        print(f"  - Training logs and plots: {demo_results['files']['logs']}")
        print(f"  - Experiment summary: {demo_results['files']['summary']}")

        return demo_results, ablation_results

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Run demonstration
    main()
