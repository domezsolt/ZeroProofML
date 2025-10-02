"""
Demonstration of RR robot inverse kinematics with ZeroProofML.

This script shows the complete workflow:
1. Generate IK dataset with singularities
2. Train models (MLP, rational+eps, TR-rational)
3. Compare performance near singularities
4. Evaluate anti-illusion metrics
"""

import os
import time

import numpy as np

try:
    from rr_ik_dataset import RobotConfig, RRDatasetGenerator, RRKinematics
except ImportError:  # Fallback when executed via -m
    from .rr_ik_dataset import RobotConfig, RRDatasetGenerator, RRKinematics
# Support running both as a module and as a script
try:
    from rr_ik_train import IKTrainer, TrainingConfig
except ImportError:  # Fallback when executed via -m
    from .rr_ik_train import IKTrainer, TrainingConfig

from zeroproof.autodiff import GradientMode, GradientModeConfig
from zeroproof.core import TRTag, real
from zeroproof.utils.metrics import PoleLocation


def demonstrate_rr_kinematics():
    """Demonstrate basic RR robot kinematics."""
    print("=== RR Robot Kinematics Demo ===\n")

    # Create robot
    config = RobotConfig(L1=1.0, L2=1.0)
    robot = RRKinematics(config)

    print("Robot configuration:")
    print(f"  L1 = {config.L1}")
    print(f"  L2 = {config.L2}")

    # Test configurations
    test_configs = [
        (0.0, np.pi / 2),  # Regular configuration
        (np.pi / 4, np.pi / 4),  # Regular configuration
        (0.0, 0.0),  # Singular (fully retracted)
        (0.0, np.pi),  # Singular (fully extended)
        (np.pi / 2, 0.01),  # Near singular
    ]

    print("\nKinematics analysis:")
    print("θ1\t\tθ2\t\tx_ee\t\ty_ee\t\tdet(J)\t\tSingular?")
    print("-" * 70)

    for theta1, theta2 in test_configs:
        # Forward kinematics
        x_ee, y_ee = robot.forward_kinematics(theta1, theta2)

        # Jacobian determinant
        det_J = robot.jacobian_determinant(theta1, theta2)

        # Singularity check
        is_singular = robot.is_singular(theta1, theta2)

        print(
            f"{theta1:.3f}\t\t{theta2:.3f}\t\t{x_ee:.3f}\t\t{y_ee:.3f}\t\t{det_J:.6f}\t\t{is_singular}"
        )

    print("\nSingularity analysis:")
    print("- det(J) = L1 * L2 * sin(θ2)")
    print("- Singularities occur when θ2 = 0 (retracted) or θ2 = π (extended)")
    print("- Near singularities have small |det(J)|")


def demonstrate_dataset_generation():
    """Demonstrate dataset generation."""
    print("\n=== Dataset Generation Demo ===\n")

    config = RobotConfig()
    generator = RRDatasetGenerator(config)

    # Generate small dataset for demo
    samples = generator.generate_dataset(
        n_samples=100,
        singular_ratio=0.4,  # Higher ratio for demo
        displacement_scale=0.1,
        singularity_threshold=0.01,
    )

    print(f"\nDataset statistics:")
    print(f"Total samples: {len(samples)}")

    # Analyze singularity distribution
    singular_samples = [s for s in samples if s.is_singular]
    regular_samples = [s for s in samples if not s.is_singular]

    print(f"Singular samples: {len(singular_samples)} ({len(singular_samples)/len(samples):.1%})")
    print(f"Regular samples: {len(regular_samples)} ({len(regular_samples)/len(samples):.1%})")

    # Statistics
    det_J_values = [abs(s.det_J) for s in samples]
    cond_values = [s.cond_J for s in samples if not np.isinf(s.cond_J)]

    print(f"\n|det(J)| statistics:")
    print(f"  Mean: {np.mean(det_J_values):.6f}")
    print(f"  Min:  {np.min(det_J_values):.6f}")
    print(f"  Max:  {np.max(det_J_values):.6f}")

    if cond_values:
        print(f"\nCondition number statistics:")
        print(f"  Mean: {np.mean(cond_values):.2f}")
        print(f"  Max:  {np.max(cond_values):.2f}")

    # Show sample data
    print(f"\nSample data (first 5 entries):")
    print("θ1\t\tθ2\t\tdx\t\tdy\t\tdθ1\t\tdθ2\t\t|det(J)|\t\tSing?")
    print("-" * 80)

    for i, sample in enumerate(samples[:5]):
        print(
            f"{sample.theta1:.3f}\t\t{sample.theta2:.3f}\t\t{sample.dx:.3f}\t\t{sample.dy:.3f}\t\t"
            f"{sample.dtheta1:.3f}\t\t{sample.dtheta2:.3f}\t\t{abs(sample.det_J):.6f}\t\t{sample.is_singular}"
        )

    return samples


def demonstrate_training_comparison():
    """Demonstrate training different models."""
    print("\n=== Training Comparison Demo ===\n")

    # Generate dataset
    config = RobotConfig()
    generator = RRDatasetGenerator(config)
    samples = generator.generate_dataset(
        n_samples=200, singular_ratio=0.3, singularity_threshold=0.005
    )

    # Split data
    n_train = int(0.8 * len(samples))
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Set gradient mode
    GradientModeConfig.set_mode(GradientMode.MASK_REAL)

    # Model configurations
    configs = [
        (
            "MLP",
            TrainingConfig(
                model_type="mlp",
                epochs=30,
                learning_rate=0.01,
                hidden_dim=16,
                use_hybrid_schedule=False,
                use_tag_loss=False,
                use_pole_head=False,
                enable_anti_illusion=False,
            ),
        ),
        (
            "TR-Rational",
            TrainingConfig(
                model_type="tr_rat",
                epochs=30,
                learning_rate=0.01,
                degree_p=2,
                degree_q=1,
                use_hybrid_schedule=True,
                use_tag_loss=True,
                use_pole_head=True,
                enable_anti_illusion=True,
            ),
        ),
    ]

    results = {}

    for name, config in configs:
        print(f"\n--- Training {name} ---")

        start_time = time.time()

        # Create and train model
        trainer = IKTrainer(config)
        trainer.create_model(input_dim=4, output_dim=2)
        trainer.setup_trainer()
        trainer.train(train_samples)

        # Evaluate
        test_metrics = trainer.evaluate(test_samples)

        training_time = time.time() - start_time

        results[name] = {
            "mse": test_metrics["mse"],
            "training_time": training_time,
            "n_parameters": len(trainer.model.parameters())
            if hasattr(trainer.model, "parameters")
            else 0,
        }

        print(f"Test MSE: {test_metrics['mse']:.6f}")
        print(f"Training time: {training_time:.2f}s")

    # Compare results
    print(f"\n=== Results Comparison ===")
    print("Model\t\t\tMSE\t\tTime (s)\tParameters")
    print("-" * 60)

    for name, result in results.items():
        print(
            f"{name:<15}\t{result['mse']:.6f}\t{result['training_time']:.2f}\t\t{result['n_parameters']}"
        )

    return results


def demonstrate_singularity_analysis():
    """Demonstrate analysis near singularities."""
    print("\n=== Singularity Analysis Demo ===\n")

    config = RobotConfig()
    robot = RRKinematics(config)

    # Create path crossing singularity
    print("Analyzing path crossing singularity...")
    print("Path: θ1 = π/4, θ2 varies from -0.2 to +0.2 (crossing θ2 = 0)")

    theta1 = np.pi / 4
    theta2_values = np.linspace(-0.2, 0.2, 21)

    print("\nθ2\t\tx_ee\t\ty_ee\t\tdet(J)\t\tManipulability\tSingular?")
    print("-" * 70)

    for theta2 in theta2_values:
        x_ee, y_ee = robot.forward_kinematics(theta1, theta2)
        det_J = robot.jacobian_determinant(theta1, theta2)
        manip = robot.manipulability_index(theta1, theta2)
        is_singular = robot.is_singular(theta1, theta2, threshold=0.01)

        marker = " <-- SINGULAR" if is_singular else ""
        print(
            f"{theta2:.3f}\t\t{x_ee:.3f}\t\t{y_ee:.3f}\t\t{det_J:.6f}\t\t{manip:.6f}\t\t{is_singular}{marker}"
        )

    print("\nObservations:")
    print("- det(J) changes sign across θ2 = 0")
    print("- Manipulability approaches zero at singularity")
    print("- End-effector position is continuous")
    print("- Jacobian becomes rank-deficient")


def demonstrate_ik_solutions():
    """Demonstrate IK solutions near singularities."""
    print("\n=== IK Solutions Demo ===\n")

    config = RobotConfig()
    robot = RRKinematics(config)

    # Test configurations
    test_configs = [
        (0.0, 0.1, "Near retracted singularity"),
        (0.0, np.pi - 0.1, "Near extended singularity"),
        (np.pi / 4, np.pi / 2, "Regular configuration"),
    ]

    for theta1, theta2, description in test_configs:
        print(f"\n{description}:")
        print(f"Configuration: θ1={theta1:.3f}, θ2={theta2:.3f}")

        # Current end-effector position
        x_ee, y_ee = robot.forward_kinematics(theta1, theta2)
        print(f"Current end-effector: ({x_ee:.3f}, {y_ee:.3f})")

        # Test displacement
        dx, dy = 0.05, 0.03
        print(f"Desired displacement: ({dx:.3f}, {dy:.3f})")

        # DLS solution
        dtheta1_dls, dtheta2_dls = robot.damped_least_squares_ik(
            theta1, theta2, dx, dy, damping=0.01
        )

        print(f"DLS solution: dθ1={dtheta1_dls:.6f}, dθ2={dtheta2_dls:.6f}")

        # Verify solution
        new_theta1 = theta1 + dtheta1_dls
        new_theta2 = theta2 + dtheta2_dls
        new_x, new_y = robot.forward_kinematics(new_theta1, new_theta2)

        actual_dx = new_x - x_ee
        actual_dy = new_y - y_ee

        print(f"Actual displacement: ({actual_dx:.6f}, {actual_dy:.6f})")
        print(f"Error: ({abs(actual_dx - dx):.6f}, {abs(actual_dy - dy):.6f})")

        # Jacobian properties
        det_J = robot.jacobian_determinant(theta1, theta2)
        cond_J = robot.jacobian_condition_number(theta1, theta2)

        print(f"det(J): {det_J:.6f}, cond(J): {cond_J:.2f}")


def main():
    """Run complete demonstration."""
    print("RR Robot Inverse Kinematics with ZeroProofML")
    print("=" * 50)

    try:
        # Basic kinematics
        demonstrate_rr_kinematics()

        # Dataset generation
        samples = demonstrate_dataset_generation()

        # Singularity analysis
        demonstrate_singularity_analysis()

        # IK solutions
        demonstrate_ik_solutions()

        # Training comparison (simplified for demo)
        print("\n=== Training Demo (Simplified) ===")
        print("Note: Full training comparison takes longer.")
        print("Run 'python rr_ik_train.py --dataset data.json' for complete training.")

        # Save sample dataset for training
        if samples:
            config = RobotConfig()
            generator = RRDatasetGenerator(config)
            generator.samples = samples

            os.makedirs("data", exist_ok=True)
            generator.save_dataset("data/demo_rr_ik_dataset.json")
            print(f"\nSample dataset saved to: data/demo_rr_ik_dataset.json")
            print("Use this with: python rr_ik_train.py --dataset data/demo_rr_ik_dataset.json")

        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey takeaways:")
        print("- RR robot has singularities when θ2 = 0 or θ2 = π")
        print("- Dataset includes both regular and singular configurations")
        print("- ZeroProofML handles singularities gracefully")
        print("- Anti-illusion metrics verify true pole understanding")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
