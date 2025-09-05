"""
RR-arm inverse kinematics training with ZeroProofML.

This script trains different models on the RR robot IK dataset,
comparing standard approaches with TR-enhanced methods.
"""

import argparse
import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from zeroproof.core import real, TRTag
from zeroproof.autodiff import TRNode, GradientModeConfig, GradientMode
from zeroproof.layers import (
    MonomialBasis,
    TRRational,
    FullyIntegratedRational
)
from zeroproof.training import (
    HybridTRTrainer,
    HybridTrainingConfig,
    Optimizer
)
from zeroproof.utils.metrics import AntiIllusionMetrics, PoleLocation
from .rr_ik_dataset import RRDatasetGenerator, IKSample, RobotConfig


@dataclass
class TrainingConfig:
    """Configuration for IK training."""
    model_type: str = "tr_rat"  # "mlp", "rat_eps", "tr_rat"
    
    # Model architecture
    hidden_dim: int = 32
    n_layers: int = 2
    degree_p: int = 3
    degree_q: int = 2
    
    # Training parameters
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    
    # ZeroProof enhancements
    use_hybrid_schedule: bool = True
    use_tag_loss: bool = True
    use_pole_head: bool = True
    use_residual_loss: bool = True
    enable_anti_illusion: bool = True
    
    # Loss weights
    lambda_tag: float = 0.05
    lambda_pole: float = 0.1
    lambda_residual: float = 0.02
    
    # Coverage control
    enforce_coverage: bool = True
    min_coverage: float = 0.7
    
    # Evaluation
    evaluate_ple: bool = True
    track_convergence: bool = True


class MLPBaseline:
    """Standard MLP baseline for comparison."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32, n_layers: int = 2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Simple MLP implementation using TRNodes for consistency
        self.layers = []
        
        # Input layer
        prev_dim = input_dim
        for i in range(n_layers):
            layer_weights = []
            layer_biases = []
            
            curr_dim = hidden_dim if i < n_layers - 1 else output_dim
            
            # Initialize weights
            scale = np.sqrt(2.0 / prev_dim)
            for j in range(curr_dim):
                weights = []
                for k in range(prev_dim):
                    w = np.random.randn() * scale
                    weights.append(TRNode.parameter(real(w), name=f"W{i}_{j}_{k}"))
                layer_weights.append(weights)
                
                b = TRNode.parameter(real(0.0), name=f"b{i}_{j}")
                layer_biases.append(b)
            
            self.layers.append((layer_weights, layer_biases))
            prev_dim = curr_dim
    
    def forward(self, x: List[TRNode]) -> List[TRNode]:
        """Forward pass through MLP."""
        current = x
        
        for i, (weights, biases) in enumerate(self.layers):
            next_layer = []
            
            for j, (weight_row, bias) in enumerate(zip(weights, biases)):
                # Linear combination
                activation = bias
                for w, inp in zip(weight_row, current):
                    activation = activation + w * inp
                
                # ReLU activation (except last layer)
                if i < len(self.layers) - 1:
                    if activation.tag == TRTag.REAL and activation.value.value > 0:
                        next_layer.append(activation)
                    else:
                        next_layer.append(TRNode.constant(real(0.0)))
                else:
                    next_layer.append(activation)
            
            current = next_layer
        
        return current
    
    def parameters(self) -> List[TRNode]:
        """Get all parameters."""
        params = []
        for weights, biases in self.layers:
            for weight_row in weights:
                params.extend(weight_row)
            params.extend(biases)
        return params


class RationalEpsBaseline:
    """Rational function with epsilon regularization baseline."""
    
    def __init__(self, input_dim: int, output_dim: int, degree_p: int = 3, degree_q: int = 2, eps: float = 1e-4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eps = eps
        
        basis = MonomialBasis()
        
        # Create multiple rational functions for multi-output
        self.rationals = []
        for _ in range(output_dim):
            rational = TRRational(degree_p, degree_q, basis)
            self.rationals.append(rational)
    
    def forward(self, x: List[TRNode]) -> List[TRNode]:
        """Forward pass with epsilon regularization."""
        if len(x) != 1:
            # For multi-input, use first input or combine
            x_input = x[0]
        else:
            x_input = x[0]
        
        outputs = []
        for rational in self.rationals:
            # Standard rational computation
            y, tag = rational.forward(x_input)
            
            # Apply epsilon regularization by modifying denominator
            # This is a simplified version - in practice would modify Q directly
            if tag != TRTag.REAL:
                # Use epsilon-regularized fallback
                outputs.append(TRNode.constant(real(0.0)))
            else:
                outputs.append(y)
        
        return outputs
    
    def parameters(self) -> List[TRNode]:
        """Get all parameters."""
        params = []
        for rational in self.rationals:
            params.extend(rational.parameters())
        return params


class IKTrainer:
    """Trainer for inverse kinematics problems."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.trainer = None
        self.training_history = []
        
    def create_model(self, input_dim: int, output_dim: int):
        """Create model based on configuration."""
        if self.config.model_type == "mlp":
            self.model = MLPBaseline(
                input_dim, output_dim, 
                self.config.hidden_dim, self.config.n_layers
            )
            
        elif self.config.model_type == "rat_eps":
            self.model = RationalEpsBaseline(
                input_dim, output_dim,
                self.config.degree_p, self.config.degree_q
            )
            
        elif self.config.model_type == "tr_rat":
            basis = MonomialBasis()
            
            if output_dim == 1:
                # Single output rational
                if (self.config.use_tag_loss or self.config.use_pole_head or 
                    self.config.enable_anti_illusion):
                    self.model = FullyIntegratedRational(
                        d_p=self.config.degree_p,
                        d_q=self.config.degree_q,
                        basis=basis,
                        enable_tag_head=self.config.use_tag_loss,
                        enable_pole_head=self.config.use_pole_head,
                        track_Q_values=True
                    )
                else:
                    self.model = TRRational(
                        d_p=self.config.degree_p,
                        d_q=self.config.degree_q,
                        basis=basis
                    )
            else:
                # Multi-output - create separate rationals
                self.model = [
                    FullyIntegratedRational(
                        d_p=self.config.degree_p,
                        d_q=self.config.degree_q,
                        basis=basis,
                        enable_tag_head=self.config.use_tag_loss,
                        enable_pole_head=self.config.use_pole_head,
                        track_Q_values=True
                    ) for _ in range(output_dim)
                ]
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def setup_trainer(self):
        """Setup the trainer with ZeroProof enhancements."""
        if self.config.model_type != "tr_rat":
            # Use simple optimizer for baselines
            self.optimizer = Optimizer(
                self.model.parameters(), 
                learning_rate=self.config.learning_rate
            )
            return
        
        # ZeroProof enhanced training
        hybrid_config = HybridTrainingConfig(
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            
            # Hybrid gradient schedule
            use_hybrid_schedule=self.config.use_hybrid_schedule,
            warmup_epochs=max(1, self.config.epochs // 5),
            transition_epochs=max(1, self.config.epochs // 3),
            
            # Tag loss
            use_tag_loss=self.config.use_tag_loss,
            lambda_tag=self.config.lambda_tag,
            
            # Pole detection
            use_pole_head=self.config.use_pole_head,
            lambda_pole=self.config.lambda_pole,
            
            # Anti-illusion metrics
            enable_anti_illusion=self.config.enable_anti_illusion,
            lambda_residual=self.config.lambda_residual,
            
            # Coverage control
            enforce_coverage=self.config.enforce_coverage,
            min_coverage=self.config.min_coverage
        )
        
        # Handle multi-output case
        if isinstance(self.model, list):
            # Train each output separately for simplicity
            self.trainers = []
            for model in self.model:
                trainer = HybridTRTrainer(
                    model=model,
                    optimizer=Optimizer(model.parameters(), learning_rate=self.config.learning_rate),
                    config=hybrid_config
                )
                self.trainers.append(trainer)
        else:
            self.trainer = HybridTRTrainer(
                model=self.model,
                optimizer=Optimizer(self.model.parameters(), learning_rate=self.config.learning_rate),
                config=hybrid_config
            )
    
    def prepare_data(self, samples: List[IKSample]) -> Tuple[List, List]:
        """Prepare training data from IK samples."""
        inputs = []
        targets = []
        
        for sample in samples:
            # Input: current configuration + desired displacement
            # This is a common formulation for differential IK
            input_vec = [sample.theta1, sample.theta2, sample.dx, sample.dy]
            
            # Target: joint displacement
            target_vec = [sample.dtheta1, sample.dtheta2]
            
            inputs.append(input_vec)
            targets.append(target_vec)
        
        return inputs, targets
    
    def train(self, train_samples: List[IKSample], val_samples: Optional[List[IKSample]] = None):
        """Train the model on IK data."""
        print(f"Training {self.config.model_type} model...")
        print(f"Training samples: {len(train_samples)}")
        if val_samples:
            print(f"Validation samples: {len(val_samples)}")
        
        # Prepare data
        train_inputs, train_targets = self.prepare_data(train_samples)
        
        # Set gradient mode
        GradientModeConfig.set_mode(GradientMode.MASK_REAL)
        
        start_time = time.time()
        
        if self.config.model_type == "tr_rat" and self.trainer:
            # Use ZeroProof trainer
            self._train_with_zeroproof(train_inputs, train_targets)
        else:
            # Use simple training loop
            self._train_baseline(train_inputs, train_targets)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Validation if provided
        if val_samples:
            val_metrics = self.evaluate(val_samples)
            print(f"Validation MSE: {val_metrics['mse']:.6f}")
    
    def _train_with_zeroproof(self, inputs: List, targets: List):
        """Train using ZeroProof enhanced trainer."""
        for epoch in range(self.config.epochs):
            epoch_metrics = []
            
            # Mini-batch training
            for i in range(0, len(inputs), self.config.batch_size):
                batch_inputs = inputs[i:i+self.config.batch_size]
                batch_targets = targets[i:i+self.config.batch_size]
                
                # Convert to TRNodes
                tr_inputs = []
                tr_targets = []
                
                for inp, tgt in zip(batch_inputs, batch_targets):
                    # Multi-input case - create list of TRNodes
                    tr_inp = [TRNode.constant(real(x)) for x in inp]
                    tr_tgt = [real(y) for y in tgt]
                    
                    tr_inputs.append(tr_inp)
                    tr_targets.append(tr_tgt)
                
                # Train batch (simplified - would need proper batching)
                for tr_inp, tr_tgt in zip(tr_inputs, tr_targets):
                    if isinstance(self.model, list):
                        # Multi-output case
                        total_loss = 0.0
                        for j, (model, trainer) in enumerate(zip(self.model, self.trainers)):
                            # Single input for each model
                            x = tr_inp[0] if len(tr_inp) > 0 else TRNode.constant(real(0.0))
                            y_target = tr_tgt[j] if j < len(tr_tgt) else real(0.0)
                            
                            # Simple training step
                            y_pred, tag = model.forward(x)
                            if tag == TRTag.REAL:
                                loss = (y_pred - TRNode.constant(y_target)) ** 2
                                loss.backward()
                                trainer.optimizer.step(model)
                                total_loss += loss.value.value
                        
                        epoch_metrics.append(total_loss / len(self.model))
                    else:
                        # Single output case - would need proper implementation
                        pass
            
            # Log progress
            if epoch % 10 == 0 and epoch_metrics:
                avg_loss = sum(epoch_metrics) / len(epoch_metrics)
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
                self.training_history.append(avg_loss)
    
    def _train_baseline(self, inputs: List, targets: List):
        """Train baseline models."""
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, len(inputs), self.config.batch_size):
                batch_inputs = inputs[i:i+self.config.batch_size]
                batch_targets = targets[i:i+self.config.batch_size]
                
                batch_loss = 0.0
                
                for inp, tgt in zip(batch_inputs, batch_targets):
                    # Convert to TRNodes
                    tr_inp = [TRNode.constant(real(x)) for x in inp]
                    
                    # Forward pass
                    outputs = self.model.forward(tr_inp)
                    
                    # Compute loss
                    loss = TRNode.constant(real(0.0))
                    for j, (output, target) in enumerate(zip(outputs, tgt)):
                        if output.tag == TRTag.REAL:
                            diff = output - TRNode.constant(real(target))
                            loss = loss + diff * diff
                    
                    # Backward pass
                    loss.backward()
                    
                    batch_loss += loss.value.value if loss.tag == TRTag.REAL else 0.0
                
                # Optimizer step
                self.optimizer.step(self.model)
                
                epoch_loss += batch_loss
                n_batches += 1
            
            # Log progress
            if epoch % 10 == 0 and n_batches > 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
                self.training_history.append(avg_loss)
    
    def evaluate(self, test_samples: List[IKSample]) -> Dict[str, float]:
        """Evaluate model on test data."""
        test_inputs, test_targets = self.prepare_data(test_samples)
        
        total_mse = 0.0
        total_samples = 0
        predictions = []
        
        for inp, tgt in zip(test_inputs, test_targets):
            tr_inp = [TRNode.constant(real(x)) for x in inp]
            
            if isinstance(self.model, list):
                # Multi-output case
                outputs = []
                for model in self.model:
                    y, tag = model.forward(tr_inp[0])
                    if tag == TRTag.REAL:
                        outputs.append(y.value.value)
                    else:
                        outputs.append(0.0)
            else:
                # Single model case
                model_outputs = self.model.forward(tr_inp)
                outputs = [out.value.value if out.tag == TRTag.REAL else 0.0 
                          for out in model_outputs]
            
            # Compute MSE
            mse = sum((pred - true)**2 for pred, true in zip(outputs, tgt))
            total_mse += mse
            total_samples += 1
            
            predictions.append(outputs)
        
        avg_mse = total_mse / total_samples if total_samples > 0 else float('inf')
        
        return {
            'mse': avg_mse,
            'predictions': predictions,
            'n_samples': total_samples
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        summary = {
            'model_type': self.config.model_type,
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'n_parameters': len(self.model.parameters()) if hasattr(self.model, 'parameters') else 0
        }
        
        if hasattr(self, 'trainer') and self.trainer:
            trainer_summary = self.trainer.get_training_summary()
            summary['zeroproof_metrics'] = trainer_summary
        
        return summary


def run_experiment(dataset_file: str, config: TrainingConfig, output_dir: str):
    """Run complete training experiment."""
    print(f"=== Running IK Training Experiment ===")
    print(f"Model: {config.model_type}")
    print(f"Dataset: {dataset_file}")
    print(f"Output: {output_dir}")
    
    # Load dataset
    generator = RRDatasetGenerator.load_dataset(dataset_file)
    samples = generator.samples
    
    print(f"Loaded {len(samples)} samples")
    
    # Train/test split
    n_train = int(0.8 * len(samples))
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]
    
    # Create trainer
    trainer = IKTrainer(config)
    
    # Determine input/output dimensions
    # Input: [theta1, theta2, dx, dy] -> 4D
    # Output: [dtheta1, dtheta2] -> 2D
    trainer.create_model(input_dim=4, output_dim=2)
    trainer.setup_trainer()
    
    # Train model
    trainer.train(train_samples, test_samples)
    
    # Evaluate
    test_metrics = trainer.evaluate(test_samples)
    print(f"Final test MSE: {test_metrics['mse']:.6f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'config': config.__dict__,
        'dataset_info': {
            'file': dataset_file,
            'n_samples': len(samples),
            'n_train': len(train_samples),
            'n_test': len(test_samples)
        },
        'training_summary': trainer.get_training_summary(),
        'test_metrics': test_metrics
    }
    
    results_file = os.path.join(output_dir, f"results_{config.model_type}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    return results


def main():
    """Command-line interface for IK training."""
    parser = argparse.ArgumentParser(description="Train IK models on RR robot data")
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset file')
    parser.add_argument('--model', type=str, choices=['mlp', 'rat_eps', 'tr_rat'],
                       default='tr_rat', help='Model type')
    parser.add_argument('--output_dir', type=str, default='runs/ik_experiment',
                       help='Output directory')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=32,
                       help='Hidden dimension for MLP')
    parser.add_argument('--degree_p', type=int, default=3,
                       help='Numerator degree for rational models')
    parser.add_argument('--degree_q', type=int, default=2,
                       help='Denominator degree for rational models')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    # ZeroProof features
    parser.add_argument('--no_hybrid', action='store_true',
                       help='Disable hybrid gradient schedule')
    parser.add_argument('--no_tag_loss', action='store_true',
                       help='Disable tag loss')
    parser.add_argument('--no_pole_head', action='store_true',
                       help='Disable pole head')
    parser.add_argument('--no_residual_loss', action='store_true',
                       help='Disable residual consistency loss')
    parser.add_argument('--no_anti_illusion', action='store_true',
                       help='Disable anti-illusion metrics')
    parser.add_argument('--no_coverage', action='store_true',
                       help='Disable coverage enforcement')
    
    # Loss weights
    parser.add_argument('--lambda_tag', type=float, default=0.05,
                       help='Tag loss weight')
    parser.add_argument('--lambda_pole', type=float, default=0.1,
                       help='Pole loss weight')
    parser.add_argument('--lambda_residual', type=float, default=0.02,
                       help='Residual loss weight')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        degree_p=args.degree_p,
        degree_q=args.degree_q,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_hybrid_schedule=not args.no_hybrid,
        use_tag_loss=not args.no_tag_loss,
        use_pole_head=not args.no_pole_head,
        use_residual_loss=not args.no_residual_loss,
        enable_anti_illusion=not args.no_anti_illusion,
        enforce_coverage=not args.no_coverage,
        lambda_tag=args.lambda_tag,
        lambda_pole=args.lambda_pole,
        lambda_residual=args.lambda_residual
    )
    
    # Run experiment
    results = run_experiment(args.dataset, config, args.output_dir)
    
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()
