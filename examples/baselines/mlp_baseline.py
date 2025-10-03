"""
MLP baseline for comparison with ZeroProofML.

Standard Multi-Layer Perceptron using ReLU/Tanh activations
for comparison with TR-enhanced rational functions.
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
from zeroproof.training import Optimizer


@dataclass
class MLPConfig:
    """Configuration for MLP baseline."""

    input_dim: int = 4
    output_dim: int = 2
    hidden_dims: List[int] = None
    activation: str = "relu"  # "relu", "tanh", "sigmoid"
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    l2_regularization: float = 0.0
    dropout_rate: float = 0.0  # Not implemented in this simple version

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


class MLPLayer:
    """Single MLP layer with bias and activation."""

    def __init__(self, input_dim: int, output_dim: int, activation: str = "relu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        # Initialize weights using Xavier/He initialization
        if activation == "relu":
            # He initialization for ReLU
            std = np.sqrt(2.0 / input_dim)
        else:
            # Xavier initialization for tanh/sigmoid
            std = np.sqrt(1.0 / input_dim)

        # Create weight matrix
        self.weights = []
        for i in range(output_dim):
            row = []
            for j in range(input_dim):
                w = np.random.randn() * std
                row.append(TRNode.parameter(real(w), name=f"w_{i}_{j}"))
            self.weights.append(row)

        # Create bias vector
        self.biases = []
        for i in range(output_dim):
            b = TRNode.parameter(real(0.0), name=f"b_{i}")
            self.biases.append(b)

    def forward(self, inputs: List[TRNode]) -> List[TRNode]:
        """Forward pass through layer."""
        outputs = []

        for i in range(self.output_dim):
            # Linear combination: w^T x + b
            activation = self.biases[i]
            for j, x in enumerate(inputs):
                if j < len(self.weights[i]):
                    activation = activation + self.weights[i][j] * x

            # Apply activation function
            if self.activation == "relu":
                if activation.tag == TRTag.REAL and activation.value.value > 0:
                    outputs.append(activation)
                else:
                    outputs.append(TRNode.constant(real(0.0)))

            elif self.activation == "tanh":
                # Simple tanh approximation
                if activation.tag == TRTag.REAL:
                    val = activation.value.value
                    if val > 3:
                        outputs.append(TRNode.constant(real(1.0)))
                    elif val < -3:
                        outputs.append(TRNode.constant(real(-1.0)))
                    else:
                        # Taylor approximation: tanh(x) ≈ x - x³/3
                        x2 = activation * activation
                        x3 = x2 * activation
                        tanh_approx = activation - x3 * TRNode.constant(real(1.0 / 3.0))
                        outputs.append(tanh_approx)
                else:
                    outputs.append(TRNode.constant(real(0.0)))

            elif self.activation == "sigmoid":
                # Simple sigmoid approximation
                if activation.tag == TRTag.REAL:
                    val = activation.value.value
                    if val > 4:
                        outputs.append(TRNode.constant(real(1.0)))
                    elif val < -4:
                        outputs.append(TRNode.constant(real(0.0)))
                    else:
                        # Approximation: sigmoid(x) ≈ 0.5 + 0.25*x - 0.03125*x³
                        x2 = activation * activation
                        x3 = x2 * activation
                        sig_approx = (
                            TRNode.constant(real(0.5))
                            + TRNode.constant(real(0.25)) * activation
                            - TRNode.constant(real(0.03125)) * x3
                        )
                        outputs.append(sig_approx)
                else:
                    outputs.append(TRNode.constant(real(0.5)))

            elif self.activation == "linear":
                outputs.append(activation)

            else:
                # Default to linear
                outputs.append(activation)

        return outputs

    def parameters(self) -> List[TRNode]:
        """Get all parameters in this layer."""
        params = []
        for row in self.weights:
            params.extend(row)
        params.extend(self.biases)
        return params


class MLPBaseline:
    """Multi-Layer Perceptron baseline model."""

    def __init__(self, config: MLPConfig):
        self.config = config
        self.layers = []

        # Build network architecture
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]

        for i in range(len(dims) - 1):
            input_dim = dims[i]
            output_dim = dims[i + 1]

            # Use linear activation for output layer
            activation = config.activation if i < len(dims) - 2 else "linear"

            layer = MLPLayer(input_dim, output_dim, activation)
            self.layers.append(layer)

        print(f"MLP Architecture: {' -> '.join(map(str, dims))}")
        print(f"Activation: {config.activation}")
        print(f"Total parameters: {len(self.parameters())}")

    def forward(self, inputs: List[TRNode]) -> List[TRNode]:
        """Forward pass through entire network."""
        current = inputs

        for layer in self.layers:
            current = layer.forward(current)

        return current

    def parameters(self) -> List[TRNode]:
        """Get all network parameters."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def regularization_loss(self) -> TRNode:
        """Compute L2 regularization loss."""
        if self.config.l2_regularization <= 0:
            return TRNode.constant(real(0.0))

        l2_loss = TRNode.constant(real(0.0))
        for param in self.parameters():
            l2_loss = l2_loss + param * param

        return TRNode.constant(real(self.config.l2_regularization)) * l2_loss


class MLPTrainer:
    """Trainer for MLP baseline."""

    def __init__(self, model: MLPBaseline, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.training_history = []
        self.start_time = None
        self.training_time = 0.0

    def train_epoch(
        self, inputs: List[List[float]], targets: List[List[float]]
    ) -> Dict[str, float]:
        """Train one epoch."""
        total_loss = 0.0
        n_batches = 0
        n_samples = 0

        batch_size = self.model.config.batch_size

        # Mini-batch training
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_targets = targets[i : i + batch_size]

            batch_loss = 0.0

            for inp, tgt in zip(batch_inputs, batch_targets):
                # Convert to TRNodes
                tr_inputs = [TRNode.constant(real(x)) for x in inp]

                # Forward pass
                outputs = self.model.forward(tr_inputs)

                # Compute MSE loss
                loss = TRNode.constant(real(0.0))
                for output, target in zip(outputs, tgt):
                    diff = output - TRNode.constant(real(target))
                    loss = loss + diff * diff

                # Add regularization
                reg_loss = self.model.regularization_loss()
                total_loss_node = loss + reg_loss

                # Backward pass
                total_loss_node.backward()

                # Accumulate loss
                if total_loss_node.tag == TRTag.REAL:
                    batch_loss += total_loss_node.value.value

                n_samples += 1

            # Optimizer step
            self.optimizer.step(self.model)

            total_loss += batch_loss
            n_batches += 1

        # Compute average loss
        avg_loss = total_loss / n_samples if n_samples > 0 else float("inf")

        return {"loss": avg_loss, "n_samples": n_samples, "n_batches": n_batches}

    def evaluate(self, inputs: List[List[float]], targets: List[List[float]]) -> Dict[str, float]:
        """Evaluate model on test data."""
        total_mse = 0.0
        total_mae = 0.0
        n_samples = 0
        predictions = []

        for inp, tgt in zip(inputs, targets):
            # Convert to TRNodes
            tr_inputs = [TRNode.constant(real(x)) for x in inp]

            # Forward pass
            outputs = self.model.forward(tr_inputs)

            # Extract predictions
            pred = []
            for output in outputs:
                if output.tag == TRTag.REAL:
                    pred.append(output.value.value)
                else:
                    pred.append(0.0)  # Default for non-REAL

            predictions.append(pred)

            # Compute metrics
            mse = sum((p - t) ** 2 for p, t in zip(pred, tgt))
            mae = sum(abs(p - t) for p, t in zip(pred, tgt))

            total_mse += mse
            total_mae += mae
            n_samples += 1

        return {
            "mse": total_mse / n_samples if n_samples > 0 else float("inf"),
            "mae": total_mae / n_samples if n_samples > 0 else float("inf"),
            "predictions": predictions,
            "n_samples": n_samples,
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

        print(f"Training MLP for {self.model.config.epochs} epochs...")
        print(f"Training samples: {len(train_inputs)}")
        if val_inputs:
            print(f"Validation samples: {len(val_inputs)}")

        self.start_time = time.time()

        # Set gradient mode
        GradientModeConfig.set_mode(GradientMode.MASK_REAL)

        for epoch in range(self.model.config.epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_inputs, train_targets)

            # Validation
            val_metrics = {}
            if val_inputs and val_targets:
                val_metrics = self.evaluate(val_inputs, val_targets)

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_mse": val_metrics.get("mse", float("inf")),
                "val_mae": val_metrics.get("mae", float("inf")),
            }
            self.training_history.append(epoch_record)

            # Logging
            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d}: "
                    f"Train Loss={train_metrics['loss']:.6f}, "
                    f"Val MSE={val_metrics.get('mse', float('inf')):.6f}"
                )

        self.training_time = time.time() - self.start_time

        print(f"Training completed in {self.training_time:.2f} seconds")

        # Final evaluation
        final_train_metrics = self.evaluate(train_inputs, train_targets)
        final_val_metrics = {}
        if val_inputs and val_targets:
            final_val_metrics = self.evaluate(val_inputs, val_targets)

        return {
            "training_time": self.training_time,
            "final_train_mse": final_train_metrics["mse"],
            "final_val_mse": final_val_metrics.get("mse", float("inf")),
            "training_history": self.training_history,
            "config": asdict(self.model.config),
        }


def run_mlp_baseline(
    train_data: Tuple[List, List],
    test_data: Tuple[List, List],
    config: Optional[MLPConfig] = None,
    output_dir: str = "results",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run complete MLP baseline experiment.

    Args:
        train_data: (inputs, targets) for training
        test_data: (inputs, targets) for testing
        config: MLP configuration
        output_dir: Directory to save results

    Returns:
        Experiment results
    """
    if config is None:
        config = MLPConfig()

    print("=== MLP Baseline Experiment ===")

    train_inputs, train_targets = train_data
    test_inputs, test_targets = test_data

    # Infer dimensions
    config.input_dim = len(train_inputs[0])
    config.output_dim = len(train_targets[0])

    # Create model
    model = MLPBaseline(config)
    optimizer = Optimizer(model.parameters(), learning_rate=config.learning_rate)
    trainer = MLPTrainer(model, optimizer)

    # Train
    training_results = trainer.train(
        train_inputs, train_targets, test_inputs, test_targets, verbose=True
    )

    # Final test evaluation
    test_metrics = trainer.evaluate(test_inputs, test_targets)

    # Compile results
    results = {
        "model_type": "MLP",
        "config": asdict(config),
        "training_results": training_results,
        "test_metrics": test_metrics,
        "n_parameters": len(model.parameters()),
        "training_time": training_results["training_time"],
        "seed": seed,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "mlp_baseline_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV summary
    csv_file = os.path.join(output_dir, "mlp_baseline_summary.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Parameters", "Train_MSE", "Test_MSE", "Training_Time"])
        writer.writerow(
            [
                "MLP",
                len(model.parameters()),
                training_results["final_train_mse"],
                test_metrics["mse"],
                training_results["training_time"],
            ]
        )

    print(f"Results saved to {results_file}")
    print(f"Summary saved to {csv_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLP Baseline")
    parser.add_argument(
        "--hidden_dims", nargs="+", type=int, default=[64, 32], help="Hidden layer dimensions"
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh", "sigmoid"],
        default="relu",
        help="Activation function",
    )
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--l2_reg", type=float, default=0.0, help="L2 regularization")
    parser.add_argument("--output_dir", default="results/mlp_baseline", help="Output directory")

    args = parser.parse_args()

    # Create configuration
    config = MLPConfig(
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        l2_regularization=args.l2_reg,
    )

    print("MLP Baseline configuration:")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Activation: {config.activation}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
    print(f"  L2 regularization: {config.l2_regularization}")

    print("\nNote: This script requires training data to be provided.")
    print("Use this as a module: from mlp_baseline import run_mlp_baseline")
    print("Or integrate with your data loading pipeline.")
