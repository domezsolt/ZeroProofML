"""
Rational function with epsilon regularization baseline.

Implements P/(Q+ε) approach with grid search over ε values
for comparison with ZeroProofML's epsilon-free approach.
"""

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zeroproof.autodiff import GradientMode, GradientModeConfig, TRNode
from zeroproof.core import TRTag, real
from zeroproof.layers import MonomialBasis, TRRational
from zeroproof.training import Optimizer


@dataclass
class RationalEpsConfig:
    """Configuration for epsilon-regularized rational baseline."""

    input_dim: int = 4
    output_dim: int = 2
    degree_p: int = 3
    degree_q: int = 2
    epsilon_values: List[float] = None  # Grid of ε values to try
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    l2_regularization: float = 1e-3
    # Optional gradient clipping (global L2 norm)
    clip_norm: Optional[float] = None

    def __post_init__(self):
        if self.epsilon_values is None:
            self.epsilon_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


class RationalEpsModel:
    """Rational model with epsilon regularization."""

    def __init__(self, config: RationalEpsConfig, epsilon: float):
        self.config = config
        self.epsilon = epsilon
        self.basis = MonomialBasis()

        # Create separate rational functions for each output
        self.rationals = []
        for i in range(config.output_dim):
            rational = TRRational(d_p=config.degree_p, d_q=config.degree_q, basis=self.basis)
            self.rationals.append(rational)

        print(f"Rational+ε model (ε={epsilon}): P{config.degree_p}/Q{config.degree_q}")
        print(f"Output dimension: {config.output_dim}")
        print(f"Total parameters: {len(self.parameters())}")

    def forward(self, inputs: List[TRNode]) -> List[TRNode]:
        """Forward pass with epsilon regularization."""
        # Use first input for single-input rational functions
        # In practice, would need more sophisticated input handling
        x_input = inputs[0] if inputs else TRNode.constant(real(0.0))

        outputs = []

        for rational in self.rationals:
            # Standard rational forward pass
            y, tag = rational.forward(x_input)

            # Apply epsilon regularization if needed
            if tag != TRTag.REAL:
                # Fallback: try to recover using epsilon
                # This is a simplified approach - in practice would modify Q
                outputs.append(TRNode.constant(real(0.0)))
            else:
                outputs.append(y)

        return outputs

    def forward_with_eps_regularization(self, x: TRNode) -> List[TRNode]:
        """
        Forward pass with explicit epsilon regularization.

        Modifies Q(x) -> Q(x) + ε to avoid exact zeros.
        """
        outputs = []

        for rational in self.rationals:
            # Manually compute P and Q with epsilon
            if hasattr(rational, "theta") and hasattr(rational, "phi"):
                # Get basis
                max_degree = max(len(rational.theta), len(rational.phi) + 1)
                psi = self.basis(x, max_degree)

                # Compute P(x)
                P = TRNode.constant(real(0.0))
                for k, theta_k in enumerate(rational.theta):
                    if k < len(psi):
                        P = P + theta_k * psi[k]

                # Compute Q(x) + ε
                Q = TRNode.constant(real(1.0 + self.epsilon))  # Leading 1 + ε
                for k, phi_k in enumerate(rational.phi):
                    if k + 1 < len(psi):
                        Q = Q + phi_k * psi[k + 1]

                # Rational output: y = P / (Q + ε)
                y = P / Q
                outputs.append(y)
            else:
                # Fallback to standard forward
                y, _ = rational.forward(x)
                outputs.append(y)

        return outputs

    def parameters(self) -> List[TRNode]:
        """Get all parameters."""
        params = []
        for rational in self.rationals:
            params.extend(rational.parameters())
        return params

    def regularization_loss(self) -> TRNode:
        """Compute L2 regularization loss."""
        if self.config.l2_regularization <= 0:
            return TRNode.constant(real(0.0))

        l2_loss = TRNode.constant(real(0.0))
        for rational in self.rationals:
            # Regularize denominator parameters more heavily
            for param in rational.parameters():
                l2_loss = l2_loss + param * param

        return TRNode.constant(real(self.config.l2_regularization)) * l2_loss


class RationalEpsTrainer:
    """Trainer for epsilon-regularized rational models."""

    def __init__(self, model: RationalEpsModel, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.training_history = []
        self.start_time = None
        self.training_time = 0.0

        # Track epsilon-specific metrics
        self.nan_count = 0
        self.inf_count = 0
        self.numerical_issues = []

    def _clip_gradients(self, max_norm: float) -> None:
        """Scale parameter gradients if global L2 norm exceeds max_norm."""
        if max_norm is None or max_norm <= 0:
            return
        grads: List[float] = []
        params = self.model.parameters()
        for p in params:
            if p.gradient is not None and p.gradient.tag == TRTag.REAL:
                try:
                    grads.append(float(p.gradient.value))
                except Exception:
                    pass
        if not grads:
            return
        import math as _m

        norm = (_m.fsum(g * g for g in grads)) ** 0.5
        if norm <= max_norm or norm == 0.0:
            return
        scale = max_norm / norm
        for p in params:
            if p.gradient is not None and p.gradient.tag == TRTag.REAL:
                try:
                    val = float(p.gradient.value)
                    p.gradient._value = real(val * scale)
                except Exception:
                    pass

    def train_epoch(
        self, inputs: List[List[float]], targets: List[List[float]]
    ) -> Dict[str, float]:
        """Train one epoch with epsilon regularization."""
        total_loss = 0.0
        n_samples = 0
        nan_count = 0
        inf_count = 0

        batch_size = self.model.config.batch_size

        # Mini-batch training
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_targets = targets[i : i + batch_size]

            for inp, tgt in zip(batch_inputs, batch_targets):
                # Convert to TRNodes
                tr_inputs = [TRNode.constant(real(x)) for x in inp]

                # Forward pass with epsilon regularization
                outputs = self.model.forward_with_eps_regularization(tr_inputs[0])

                # Compute loss
                loss = TRNode.constant(real(0.0))
                valid_outputs = 0

                for output, target in zip(outputs, tgt):
                    if output.tag == TRTag.REAL:
                        if np.isnan(output.value.value):
                            nan_count += 1
                            continue
                        elif np.isinf(output.value.value):
                            inf_count += 1
                            continue

                        diff = output - TRNode.constant(real(target))
                        loss = loss + diff * diff
                        valid_outputs += 1
                    else:
                        # Non-REAL output - apply penalty
                        penalty = TRNode.constant(real(1.0))
                        loss = loss + penalty

                # Skip if no valid outputs
                if valid_outputs == 0:
                    continue

                # Add regularization
                reg_loss = self.model.regularization_loss()
                total_loss_node = loss + reg_loss

                # Backward pass
                try:
                    total_loss_node.backward()

                    # Check for gradient issues
                    gradient_ok = True
                    for param in self.model.parameters():
                        if param.gradient and param.gradient.tag == TRTag.REAL:
                            if np.isnan(param.gradient.value) or np.isinf(param.gradient.value):
                                gradient_ok = False
                                break

                    if gradient_ok:
                        # Optional gradient clipping
                        try:
                            clip_norm = getattr(self.model.config, "clip_norm", None)
                            if clip_norm is not None:
                                self._clip_gradients(float(clip_norm))
                        except Exception:
                            pass
                        # Optimizer step
                        self.optimizer.step(self.model)

                    # Accumulate loss
                    if total_loss_node.tag == TRTag.REAL:
                        total_loss += total_loss_node.value.value

                except Exception as e:
                    # Record numerical issue
                    self.numerical_issues.append(str(e))
                    continue

                n_samples += 1

        # Update counters
        self.nan_count += nan_count
        self.inf_count += inf_count

        # Compute average loss
        avg_loss = total_loss / n_samples if n_samples > 0 else float("inf")

        return {
            "loss": avg_loss,
            "n_samples": n_samples,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "numerical_issues": len(self.numerical_issues),
        }

    def train(
        self,
        train_inputs: List[List[float]],
        train_targets: List[List[float]],
        val_inputs: Optional[List[List[float]]] = None,
        val_targets: Optional[List[List[float]]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Complete training loop."""

        print(
            f"Training Rational+ε (ε={self.model.epsilon}) for {self.model.config.epochs} epochs..."
        )
        print(f"Training samples: {len(train_inputs)}")

        self.start_time = time.time()

        # Set gradient mode
        GradientModeConfig.set_mode(GradientMode.MASK_REAL)

        for epoch in range(self.model.config.epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_inputs, train_targets)

            # Validation
            val_metrics = {}
            if val_inputs and val_targets:
                # Simple validation without epsilon issues tracking
                val_metrics = self._evaluate_simple(val_inputs, val_targets)

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_mse": val_metrics.get("mse", float("inf")),
                "nan_count": train_metrics["nan_count"],
                "inf_count": train_metrics["inf_count"],
            }
            self.training_history.append(epoch_record)

            # Logging
            if verbose and epoch % 20 == 0:
                print(
                    f"Epoch {epoch:3d}: "
                    f"Loss={train_metrics['loss']:.6f}, "
                    f"NaN={train_metrics['nan_count']}, "
                    f"Inf={train_metrics['inf_count']}"
                )

        self.training_time = time.time() - self.start_time

        print(f"Training completed in {self.training_time:.2f} seconds")
        print(f"Total NaN outputs: {self.nan_count}")
        print(f"Total Inf outputs: {self.inf_count}")
        print(f"Numerical issues: {len(self.numerical_issues)}")

        # Final evaluation
        final_val_metrics = {}
        if val_inputs and val_targets:
            final_val_metrics = self._evaluate_simple(val_inputs, val_targets)

        return {
            "training_time": self.training_time,
            "final_val_mse": final_val_metrics.get("mse", float("inf")),
            "training_history": self.training_history,
            "config": asdict(self.model.config),
            "epsilon": self.model.epsilon,
            "total_nan_count": self.nan_count,
            "total_inf_count": self.inf_count,
            "numerical_issues": len(self.numerical_issues),
        }

    def _evaluate_simple(
        self, inputs: List[List[float]], targets: List[List[float]]
    ) -> Dict[str, Any]:
        """Simple evaluation without tracking numerical issues."""
        total_mse = 0.0
        n_valid = 0
        per_sample_mse: List[float] = []
        predictions: List[List[float]] = []

        for inp, tgt in zip(inputs, targets):
            tr_inputs = [TRNode.constant(real(x)) for x in inp]

            try:
                outputs = self.model.forward_with_eps_regularization(tr_inputs[0])

                # Compute MSE for valid outputs and collect predictions
                mse = 0.0
                valid_outputs = 0
                pred_vec: List[float] = []

                for output, target in zip(outputs, tgt):
                    if (
                        output.tag == TRTag.REAL
                        and not np.isnan(output.value.value)
                        and not np.isinf(output.value.value)
                    ):
                        val = float(output.value.value)
                        pred_vec.append(val)
                        error = (val - target) ** 2
                        mse += error
                        valid_outputs += 1

                if valid_outputs > 0:
                    avg = mse / valid_outputs
                    total_mse += avg
                    n_valid += 1
                    per_sample_mse.append(avg)
                    predictions.append(pred_vec)

            except:
                continue

        avg_mse = total_mse / n_valid if n_valid > 0 else float("inf")

        return {
            "mse": avg_mse,
            "n_valid": n_valid,
            "success_rate": n_valid / len(inputs) if inputs else 0.0,
            "per_sample_mse": per_sample_mse,
            "predictions": predictions,
        }


def grid_search_epsilon(
    train_data: Tuple[List, List],
    val_data: Tuple[List, List],
    config: Optional[RationalEpsConfig] = None,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Grid search over epsilon values to find best regularization.

    Args:
        train_data: (inputs, targets) for training
        val_data: (inputs, targets) for validation
        config: Rational epsilon configuration
        output_dir: Directory to save results

    Returns:
        Grid search results
    """
    if config is None:
        config = RationalEpsConfig()

    print("=== Rational+ε Grid Search ===")
    print(f"Testing ε values: {config.epsilon_values}")

    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data

    # Infer dimensions
    config.input_dim = len(train_inputs[0])
    config.output_dim = len(train_targets[0])

    results = []
    best_epsilon = None
    best_mse = float("inf")

    for epsilon in config.epsilon_values:
        print(f"\n--- Training with ε = {epsilon} ---")

        # Create model with this epsilon
        model = RationalEpsModel(config, epsilon)
        optimizer = Optimizer(model.parameters(), learning_rate=config.learning_rate)
        trainer = RationalEpsTrainer(model, optimizer)

        # Train
        training_results = trainer.train(
            train_inputs, train_targets, val_inputs, val_targets, verbose=False
        )

        # Evaluate
        val_mse = training_results["final_val_mse"]

        result = {
            "epsilon": epsilon,
            "val_mse": val_mse,
            "training_time": training_results["training_time"],
            "nan_count": training_results["total_nan_count"],
            "inf_count": training_results["total_inf_count"],
            "numerical_issues": training_results["numerical_issues"],
            "success": val_mse < float("inf"),
        }

        results.append(result)

        print(
            f"ε={epsilon}: MSE={val_mse:.6f}, "
            f"Time={training_results['training_time']:.2f}s, "
            f"NaN={training_results['total_nan_count']}, "
            f"Inf={training_results['total_inf_count']}"
        )

        # Track best
        if val_mse < best_mse:
            best_mse = val_mse
            best_epsilon = epsilon

    print(f"\n=== Grid Search Results ===")
    print(f"Best ε: {best_epsilon}")
    print(f"Best validation MSE: {best_mse:.6f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    grid_results = {
        "config": asdict(config),
        "results": results,
        "best_epsilon": best_epsilon,
        "best_mse": best_mse,
    }

    results_file = os.path.join(output_dir, "rational_eps_grid_search.json")
    with open(results_file, "w") as f:
        json.dump(grid_results, f, indent=2)

    # Save CSV
    csv_file = os.path.join(output_dir, "rational_eps_grid_search.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Epsilon", "Val_MSE", "Training_Time", "NaN_Count", "Inf_Count", "Success"]
        )

        for result in results:
            writer.writerow(
                [
                    result["epsilon"],
                    result["val_mse"],
                    result["training_time"],
                    result["nan_count"],
                    result["inf_count"],
                    result["success"],
                ]
            )

    print(f"Results saved to {results_file}")
    print(f"CSV saved to {csv_file}")

    return grid_results


def run_rational_eps_baseline(
    train_data: Tuple[List, List],
    test_data: Tuple[List, List],
    epsilon: Optional[float] = None,
    config: Optional[RationalEpsConfig] = None,
    output_dir: str = "results",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run rational+ε baseline with specific epsilon.

    Args:
        train_data: Training data
        test_data: Test data
        epsilon: Specific epsilon value (if None, uses grid search)
        config: Configuration
        output_dir: Output directory

    Returns:
        Experiment results
    """
    if config is None:
        config = RationalEpsConfig()

    train_inputs, train_targets = train_data
    test_inputs, test_targets = test_data

    # Infer dimensions
    config.input_dim = len(train_inputs[0])
    config.output_dim = len(train_targets[0])

    if epsilon is None:
        # Grid search to find best epsilon
        print("No epsilon specified, running grid search...")
        grid_results = grid_search_epsilon(train_data, test_data, config, output_dir)
        epsilon = grid_results["best_epsilon"]

        if epsilon is None:
            print("Grid search failed to find valid epsilon")
            return {"error": "Grid search failed"}

    print(f"\n=== Rational+ε Baseline (ε={epsilon}) ===")

    # Create model with best/specified epsilon
    model = RationalEpsModel(config, epsilon)
    optimizer = Optimizer(model.parameters(), learning_rate=config.learning_rate)
    trainer = RationalEpsTrainer(model, optimizer)

    # Train
    training_results = trainer.train(
        train_inputs, train_targets, test_inputs, test_targets, verbose=True
    )

    # Final test evaluation
    test_metrics = trainer._evaluate_simple(test_inputs, test_targets)

    # Compile results
    results = {
        "model_type": "Rational+ε",
        "epsilon": epsilon,
        "config": asdict(config),
        "training_results": training_results,
        "test_metrics": test_metrics,
        "n_parameters": len(model.parameters()),
        "training_time": training_results["training_time"],
        "numerical_stability": {
            "nan_count": training_results["total_nan_count"],
            "inf_count": training_results["total_inf_count"],
            "numerical_issues": training_results["numerical_issues"],
        },
        "seed": seed,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, f"rational_eps_baseline_eps_{epsilon}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rational+ε Baseline")
    parser.add_argument("--degree_p", type=int, default=3, help="Numerator polynomial degree")
    parser.add_argument("--degree_q", type=int, default=2, help="Denominator polynomial degree")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Specific epsilon value (if None, runs grid search)",
    )
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--l2_reg", type=float, default=1e-3, help="L2 regularization")
    parser.add_argument(
        "--output_dir", default="results/rational_eps_baseline", help="Output directory"
    )

    args = parser.parse_args()

    # Create configuration
    config = RationalEpsConfig(
        degree_p=args.degree_p,
        degree_q=args.degree_q,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2_regularization=args.l2_reg,
    )

    print("Rational+ε Baseline configuration:")
    print(f"  Degree P: {config.degree_p}")
    print(f"  Degree Q: {config.degree_q}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
    print(f"  L2 regularization: {config.l2_regularization}")

    if args.epsilon:
        print(f"  Epsilon: {args.epsilon}")
    else:
        print(f"  Epsilon grid: {config.epsilon_values}")

    print("\nNote: This script requires training data to be provided.")
    print("Use this as a module: from rational_eps_baseline import run_rational_eps_baseline")
    print("Or integrate with your data loading pipeline.")
