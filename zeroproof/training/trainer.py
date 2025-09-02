"""
Training utilities for models using transreal arithmetic.

This module provides a trainer class that handles the complexities of
training with transreal values, including adaptive loss policies and
proper gradient handling.
"""

from typing import List, Optional, Callable, Dict, Tuple, Any
from dataclasses import dataclass
import time

from ..core import TRScalar, TRTag, real
from ..autodiff import TRNode, backward_pass
from .adaptive_loss import AdaptiveLossPolicy, create_adaptive_loss
from .coverage import CoverageTracker


@dataclass
class TrainingConfig:
    """Configuration for transreal training."""
    # Optimization
    learning_rate: float = 0.001
    use_safe_lr: bool = False
    batch_size: int = 32
    max_epochs: int = 100
    
    # Adaptive loss
    use_adaptive_loss: bool = True
    target_coverage: float = 0.95
    lambda_learning_rate: float = 0.01
    initial_lambda: float = 1.0
    
    # Logging
    log_interval: int = 10
    verbose: bool = True
    
    # Early stopping
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4


class Optimizer:
    """Simple gradient descent optimizer for TR values."""
    
    def __init__(self, parameters: List[TRNode], learning_rate: float = 0.001):
        """
        Initialize optimizer.
        
        Args:
            parameters: List of parameter nodes to optimize
            learning_rate: Learning rate for updates
        """
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.step_count = 0
    
    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        for param in self.parameters:
            param.zero_grad()
    
    def step(self) -> None:
        """Perform optimization step."""
        from ..core import tr_mul, tr_sub
        
        for param in self.parameters:
            if param.gradient is not None and param.gradient.tag == TRTag.REAL:
                # Update: param = param - lr * grad
                lr_node = TRNode.constant(real(self.learning_rate))
                update = tr_mul(lr_node, param.gradient)
                # Subtract the TRScalar update from the TRScalar parameter value
                new_value = tr_sub(param.value, update)
                
                # Update parameter value in-place
                param._value = new_value
        
        self.step_count += 1


class TRTrainer:
    """
    Trainer for models using transreal arithmetic.
    
    Handles training loops, adaptive loss policies, and proper
    gradient computation with the Mask-REAL rule.
    """
    
    def __init__(self,
                 model: Any,
                 optimizer: Optional[Optimizer] = None,
                 config: Optional[TrainingConfig] = None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train (must have parameters() method)
            optimizer: Optimizer instance (creates default if None)
            config: Training configuration
        """
        self.model = model
        self.config = config or TrainingConfig()
        
        # Get model parameters
        if hasattr(model, 'parameters'):
            parameters = model.parameters()
        else:
            raise ValueError("Model must have parameters() method")
        
        # Create optimizer if not provided
        self.optimizer = optimizer or Optimizer(
            parameters, 
            learning_rate=self.config.learning_rate
        )
        
        # Create adaptive loss policy if enabled
        if self.config.use_adaptive_loss:
            self.loss_policy = create_adaptive_loss(
                target_coverage=self.config.target_coverage,
                learning_rate=self.config.lambda_learning_rate,
                initial_lambda=self.config.initial_lambda
            )
        else:
            self.loss_policy = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.training_history: Dict[str, List[float]] = {
            "loss": [],
            "coverage": [],
            "lambda_rej": [],
        }
        
        # Early stopping state
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def _compute_safe_lr(self, inputs: List[TRScalar], predictions: List[TRNode]) -> float:
        """Compute Phase-6 safe learning-rate clamp if enabled.

        Uses L_batch ≈ (Bpsi^2 / q_min^2) * (1 + y_max^2) + alpha.
        Requires model to expose basis bound and q_min computation.
        """
        # Try to get q_min and bounds from model if available
        try:
            # Compute q_min over inputs
            if hasattr(self.model, 'compute_q_min'):
                q_min = self.model.compute_q_min(inputs)
            else:
                return self.optimizer.learning_rate
            # Basis bound proxy
            Bpsi = getattr(getattr(self.model, 'basis', None), 'bound', None)
            if Bpsi is None:
                return self.optimizer.learning_rate
            # y_max over REAL predictions
            y_vals = [p.value.value for p in predictions if p.tag == TRTag.REAL]
            y_max = max([abs(v) for v in y_vals], default=0.0)
            # Regularization alpha on phi if present
            alpha = getattr(self.model, 'alpha_phi', 0.0) or 0.0
            if q_min == 0.0:
                return min(self.optimizer.learning_rate, 0.0)
            L_batch = (Bpsi * Bpsi) / (q_min * q_min) * (1.0 + y_max * y_max) + alpha
            if L_batch <= 0.0:
                return self.optimizer.learning_rate
            return min(self.optimizer.learning_rate, 1.0 / L_batch)
        except Exception:
            # On any issue, fall back to configured LR
            return self.optimizer.learning_rate

    def train_step(self,
                   inputs: List[TRScalar],
                   targets: List[TRScalar]) -> Dict[str, float]:
        """
        Perform single training step.
        
        Args:
            inputs: Batch of input values
            targets: Batch of target values
            
        Returns:
            Dictionary of metrics from this step
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = []
        for x in inputs:
            y = self.model(x)
            predictions.append(y)
        
        # Compute loss
        if self.loss_policy is not None:
            # Convert targets to nodes
            target_nodes = [TRNode.constant(t) for t in targets]
            loss = self.loss_policy.compute_batch_loss(predictions, target_nodes)
        else:
            # Simple MSE loss without adaptive policy
            from ..core import tr_sum, tr_div
            losses = []
            for pred, target in zip(predictions, targets):
                target_node = TRNode.constant(target)
                diff = pred - target_node
                sq_loss = TRNode.constant(real(0.5)) * diff * diff
                losses.append(sq_loss.value)
            
            total_loss = tr_sum(losses)
            avg_loss = tr_div(total_loss, real(float(len(inputs))))
            loss = TRNode.constant(avg_loss)
        
        # Backward pass
        loss.backward()
        
        # Optional safe LR clamp
        if self.config.use_safe_lr:
            safe_lr = self._compute_safe_lr(inputs, predictions)
            self.optimizer.learning_rate = safe_lr
        # Optimization step
        self.optimizer.step()
        
        # Collect metrics
        metrics = {
            "loss": loss.value.value if loss.value.tag == TRTag.REAL else float('nan'),
            "batch_size": len(inputs),
        }
        
        # Add coverage metrics if using adaptive loss
        if self.loss_policy is not None:
            stats = self.loss_policy.get_statistics()
            metrics.update({
                "coverage": stats["current_coverage"],
                "lambda_rej": stats["lambda_rej"],
                "coverage_gap": stats["coverage_gap"],
            })
        
        return metrics
    
    def train_epoch(self,
                    data_loader: List[Tuple[List[TRScalar], List[TRScalar]]]) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            data_loader: List of (inputs, targets) batches
            
        Returns:
            Average metrics for the epoch
        """
        epoch_metrics = {
            "loss": [],
            "coverage": [],
            "lambda_rej": [],
        }
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Training step
            metrics = self.train_step(inputs, targets)
            
            # Accumulate metrics
            epoch_metrics["loss"].append(metrics["loss"])
            if "coverage" in metrics:
                epoch_metrics["coverage"].append(metrics["coverage"])
                epoch_metrics["lambda_rej"].append(metrics["lambda_rej"])
            
            # Logging
            if self.config.verbose and batch_idx % self.config.log_interval == 0:
                self._log_batch(batch_idx, len(data_loader), metrics)
            
            self.global_step += 1
        
        # Compute epoch averages
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        return avg_metrics
    
    def train(self,
              train_data: List[Tuple[List[TRScalar], List[TRScalar]]],
              val_data: Optional[List[Tuple[List[TRScalar], List[TRScalar]]]] = None) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_data: Training data batches
            val_data: Optional validation data
            
        Returns:
            Training history
        """
        if self.config.verbose:
            print(f"Starting training for {self.config.max_epochs} epochs")
            if self.config.use_adaptive_loss:
                print(f"Using adaptive loss with target coverage: {self.config.target_coverage}")
        
        start_time = time.time()
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch + 1
            
            # Training epoch
            train_metrics = self.train_epoch(train_data)
            
            # Record history
            self.training_history["loss"].append(train_metrics["loss"])
            if "coverage" in train_metrics:
                self.training_history["coverage"].append(train_metrics["coverage"])
                self.training_history["lambda_rej"].append(train_metrics["lambda_rej"])
            
            # Validation
            if val_data is not None:
                val_metrics = self.evaluate(val_data)
                val_loss = val_metrics["loss"]
            else:
                val_loss = train_metrics["loss"]
            
            # Logging
            if self.config.verbose:
                self._log_epoch(epoch, train_metrics, val_metrics if val_data else None)
            
            # Early stopping
            if self.config.early_stopping:
                if val_loss < self.best_loss - self.config.min_delta:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.patience:
                        if self.config.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
        
        training_time = time.time() - start_time
        if self.config.verbose:
            print(f"Training completed in {training_time:.2f} seconds")
        
        return self.training_history
    
    def evaluate(self,
                 data_loader: List[Tuple[List[TRScalar], List[TRScalar]]]) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Args:
            data_loader: Evaluation data batches
            
        Returns:
            Average metrics
        """
        metrics = {
            "loss": [],
            "coverage": [],
        }
        
        coverage_tracker = CoverageTracker()
        
        for inputs, targets in data_loader:
            # Forward pass only
            predictions = []
            tags = []
            for x in inputs:
                y = self.model(x)
                predictions.append(y)
                tags.append(y.tag)
            
            # Track coverage
            coverage_tracker.update(tags)
            
            # Compute loss (no gradients needed)
            from ..core import tr_sum, tr_div
            losses = []
            for pred, target in zip(predictions, targets):
                if pred.tag == TRTag.REAL:
                    diff_val = pred.value.value - target.value
                    sq_loss = 0.5 * diff_val * diff_val
                    losses.append(sq_loss)
                else:
                    # Use current lambda for non-REAL
                    if self.loss_policy:
                        losses.append(self.loss_policy.adaptive_lambda.get_penalty())
                    else:
                        losses.append(1.0)  # Default penalty
            
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            metrics["loss"].append(avg_loss)
            metrics["coverage"].append(coverage_tracker.batch_coverage)
        
        # Compute averages
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        return avg_metrics
    
    def _log_batch(self, batch_idx: int, total_batches: int, metrics: Dict[str, float]) -> None:
        """Log batch metrics."""
        msg = f"Epoch {self.epoch} [{batch_idx}/{total_batches}]"
        msg += f" Loss: {metrics['loss']:.4f}"
        if "coverage" in metrics:
            msg += f" Coverage: {metrics['coverage']:.3f}"
            msg += f" λ_rej: {metrics['lambda_rej']:.3f}"
        print(msg)
    
    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                   val_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log epoch metrics."""
        msg = f"Epoch {epoch + 1}/{self.config.max_epochs}"
        msg += f" - Train Loss: {train_metrics['loss']:.4f}"
        
        if "coverage" in train_metrics:
            msg += f" Coverage: {train_metrics['coverage']:.3f}"
            msg += f" λ_rej: {train_metrics['lambda_rej']:.3f}"
        
        if val_metrics:
            msg += f" - Val Loss: {val_metrics['loss']:.4f}"
            if "coverage" in val_metrics:
                msg += f" Coverage: {val_metrics['coverage']:.3f}"
        
        print(msg)
