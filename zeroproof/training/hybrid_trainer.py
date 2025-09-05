"""
Enhanced trainer with hybrid gradient schedule support.

This module extends the basic trainer to support sophisticated training
strategies including hybrid gradient schedules, tag-loss, and pole learning.
"""

from typing import List, Optional, Dict, Tuple, Any, Union
from dataclasses import dataclass
import math
import time

from ..core import TRScalar, TRTag, real
from ..autodiff import TRNode, backward_pass
from ..autodiff.grad_mode import GradientMode, GradientModeConfig
from ..autodiff.hybrid_gradient import (
    HybridGradientContext, 
    HybridGradientSchedule,
    create_default_schedule
)
from ..utils.bridge import to_trnode_constant
from .trainer import TRTrainer, TrainingConfig, Optimizer
from .adaptive_loss import AdaptiveLossPolicy
from .coverage import CoverageTracker
from .enhanced_coverage import (
    EnhancedCoverageConfig,
    AdaptiveLambdaController,
    NearPoleSampler,
    CoverageEnforcementPolicy,
    CoverageStrategy
)
from .pole_detection import (
    PoleDetectionConfig,
    compute_pole_loss,
    DomainSpecificPoleDetector
)
from ..utils.metrics import (
    AntiIllusionMetrics,
    PoleLocation,
    ResidualConsistencyLoss
)
from ..utils.logging import (
    StructuredLogger,
    log_training_step
)


@dataclass
class HybridTrainingConfig(TrainingConfig):
    """Extended configuration for hybrid training."""
    
    # Hybrid gradient schedule
    use_hybrid_gradient: bool = False
    hybrid_warmup_epochs: int = 20
    hybrid_transition_epochs: int = 30
    hybrid_delta_init: float = 1e-2
    hybrid_delta_final: float = 1e-6
    hybrid_aggressive: bool = False
    
    # Tag loss (for non-REAL outputs)
    use_tag_loss: bool = False
    lambda_tag: float = 0.05
    
    # Pole detection head
    use_pole_head: bool = False
    lambda_pole: float = 0.1
    pole_head_degree: int = 3
    pole_config: Optional[PoleDetectionConfig] = None
    use_teacher_signals: bool = False
    pole_proximity_threshold: float = 0.1
    
    # Enhanced metrics
    track_pole_metrics: bool = False
    compute_ple: bool = False  # Pole Localization Error
    
    # Anti-illusion metrics
    enable_anti_illusion: bool = False
    lambda_residual: float = 0.01
    ground_truth_poles: Optional[List[Tuple[float, Optional[float]]]] = None
    ple_x_range: Tuple[float, float] = (-2.0, 2.0)
    residual_near_pole_threshold: float = 0.2
    
    # Coverage enforcement
    enforce_coverage: bool = False
    min_coverage: float = 0.7
    max_lambda_for_coverage: float = 10.0
    coverage_strategy: CoverageStrategy = CoverageStrategy.LAGRANGE
    coverage_window_size: int = 50
    oversample_near_pole: bool = False
    pole_sampling_threshold: float = 0.1
    
    # Logging and tracking
    enable_structured_logging: bool = True
    log_interval: int = 1  # Log every N epochs
    save_plots: bool = True
    run_dir: Optional[str] = None


class HybridTRTrainer(TRTrainer):
    """
    Enhanced trainer with hybrid gradient schedule support.
    
    This trainer supports:
    - Hybrid gradient schedules for near-pole learning
    - Tag-loss for non-REAL outputs
    - Pole detection heads
    - Advanced metrics for pole learning verification
    """
    
    def __init__(self,
                 model: Any,
                 optimizer: Optional[Optimizer] = None,
                 config: Optional[HybridTrainingConfig] = None):
        """
        Initialize hybrid trainer.
        
        Args:
            model: Model to train (should support hybrid features)
            optimizer: Optimizer instance
            config: Hybrid training configuration
        """
        # Use hybrid config or create default
        config = config or HybridTrainingConfig()
        super().__init__(model, optimizer, config)
        
        # Cast config to HybridTrainingConfig for type checking
        self.hybrid_config: HybridTrainingConfig = config  # type: ignore
        
        # Initialize hybrid gradient schedule if enabled
        self.hybrid_schedule = None
        if self.hybrid_config.use_hybrid_gradient:
            self.hybrid_schedule = self._create_hybrid_schedule()
            
            # Register with model if it supports hybrid
            if hasattr(model, 'hybrid_schedule'):
                model.hybrid_schedule = self.hybrid_schedule
        
        # Initialize tracking
        self.pole_metrics_history = []
        self.tag_statistics = []
        self.gradient_mode_history = []
        
        # Initialize domain-specific pole detector if using teacher signals
        self.pole_detector = None
        if self.hybrid_config.use_teacher_signals:
            self.pole_detector = DomainSpecificPoleDetector()
        
        # Initialize anti-illusion metrics
        self.anti_illusion_metrics = None
        self.residual_loss = None
        self.ground_truth_poles = []
        
        if self.hybrid_config.enable_anti_illusion:
            self.anti_illusion_metrics = AntiIllusionMetrics()
            self.residual_loss = ResidualConsistencyLoss(
                weight=self.hybrid_config.lambda_residual
            )
            
            # Convert ground truth poles format
            if self.hybrid_config.ground_truth_poles:
                for pole_data in self.hybrid_config.ground_truth_poles:
                    if len(pole_data) >= 2 and pole_data[1] is not None:
                        # 2D pole
                        self.ground_truth_poles.append(
                            PoleLocation(x=pole_data[0], y=pole_data[1])
                        )
                    else:
                        # 1D pole
                        self.ground_truth_poles.append(
                            PoleLocation(x=pole_data[0])
                        )
        
        # Initialize coverage enforcement if enabled
        self.coverage_policy = None
        if self.hybrid_config.enforce_coverage:
            coverage_config = EnhancedCoverageConfig(
                target_coverage=self.hybrid_config.target_coverage,
                min_coverage=self.hybrid_config.min_coverage,
                max_coverage=0.95,
                strategy=self.hybrid_config.coverage_strategy,
                lambda_init=self.hybrid_config.initial_lambda,
                lambda_min=self.hybrid_config.adaptive_lambda_min,
                lambda_max=self.hybrid_config.max_lambda_for_coverage,
                learning_rate=self.hybrid_config.lambda_learning_rate,
                window_size=self.hybrid_config.coverage_window_size,
                oversample_near_pole=self.hybrid_config.oversample_near_pole,
                pole_threshold=self.hybrid_config.pole_sampling_threshold
            )
            self.coverage_policy = CoverageEnforcementPolicy(coverage_config)
    
    def _create_hybrid_schedule(self) -> HybridGradientSchedule:
        """Create hybrid gradient schedule from config."""
        return HybridGradientSchedule(
            warmup_epochs=self.hybrid_config.hybrid_warmup_epochs,
            transition_epochs=self.hybrid_config.hybrid_transition_epochs,
            delta_init=self.hybrid_config.hybrid_delta_init,
            delta_final=self.hybrid_config.hybrid_delta_final,
            enable=True,
            saturating_bound=0.1 if self.hybrid_config.hybrid_aggressive else 1.0
        )
    
    def train_epoch(self,
                   data_loader: List[Tuple[List[TRScalar], List[TRScalar]]]
                   ) -> Dict[str, float]:
        """
        Train one epoch with hybrid gradient support.
        
        Args:
            data_loader: List of (inputs, targets) batches
            
        Returns:
            Dictionary of epoch metrics
        """
        # Update hybrid schedule if enabled
        if self.hybrid_schedule:
            HybridGradientContext.update_epoch(self.epoch)
            
            # Update model if it tracks epochs
            if hasattr(self.model, 'update_epoch'):
                self.model.update_epoch(self.epoch)
            
            # Set gradient mode
            delta = self.hybrid_schedule.get_delta(self.epoch)
            if delta is None:
                GradientModeConfig.set_mode(GradientMode.MASK_REAL)
            else:
                GradientModeConfig.set_mode(GradientMode.HYBRID)
                GradientModeConfig.set_local_threshold(delta)
        
        metrics = {
            "loss": [],
            "coverage": [],
            "lambda_rej": [],
            "tag_loss": [],
            "pole_loss": [],
            "near_pole_ratio": []
        }
        
        coverage_tracker = CoverageTracker()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            batch_metrics = self._train_batch(inputs, targets, coverage_tracker)
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key in metrics and value is not None:
                    metrics[key].append(value)
            
            # Log if needed
            if self.config.verbose and batch_idx % self.config.log_interval == 0:
                self._log_batch(batch_idx, len(data_loader), batch_metrics)
            
            self.global_step += 1
        
        # Compute epoch averages
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        # Apply coverage enforcement if enabled
        if self.coverage_policy and 'coverage' in avg_metrics:
            # Collect Q values if available
            Q_values = None
            if hasattr(self.model, 'q_min_history') and self.model.q_min_history:
                Q_values = self.model.q_min_history
            
            # Enforce coverage policy
            enforcement_actions = self.coverage_policy.enforce(
                avg_metrics['coverage'],
                self.epoch,
                Q_values
            )
            
            # Update lambda if changed
            if enforcement_actions['lambda_updated']:
                new_lambda = enforcement_actions['new_lambda']
                if self.loss_policy:
                    self.loss_policy.adaptive_lambda.lambda_rej = new_lambda
                avg_metrics['lambda_rej'] = new_lambda
                avg_metrics['coverage_enforced'] = True
                
                # Log intervention if triggered
                if enforcement_actions['intervention_triggered']:
                    print(f"[Coverage Intervention] Epoch {self.epoch}: "
                          f"Coverage {avg_metrics['coverage']:.3f} < {self.hybrid_config.min_coverage:.3f}, "
                          f"Lambda reduced to {new_lambda:.3f}")
        
        # Evaluate anti-illusion metrics if enabled
        if (self.hybrid_config.enable_anti_illusion and 
            self.anti_illusion_metrics and 
            self.ground_truth_poles and 
            epoch % 5 == 0):  # Evaluate every 5 epochs
            
            try:
                illusion_metrics = self.anti_illusion_metrics.evaluate_model(
                    self.model,
                    self.ground_truth_poles,
                    x_range=self.hybrid_config.ple_x_range
                )
                
                # Add to average metrics
                for key, value in illusion_metrics.items():
                    if not math.isnan(value) and not math.isinf(value):
                        avg_metrics[f'ai_{key}'] = value
                
                # Log key metrics
                if epoch % 10 == 0:
                    print(f"  Anti-illusion: PLE={illusion_metrics.get('ple', float('inf')):.4f}, "
                          f"Sign={illusion_metrics.get('sign_consistency', 0):.3f}, "
                          f"Score={illusion_metrics.get('anti_illusion_score', float('inf')):.4f}")
            
            except Exception as e:
                print(f"  Warning: Anti-illusion evaluation failed: {e}")
        
        # Add hybrid-specific metrics
        if self.hybrid_schedule:
            hybrid_stats = HybridGradientContext.get_statistics()
            avg_metrics['saturating_ratio'] = hybrid_stats.get('saturating_ratio', 0.0)
            avg_metrics['gradient_mode'] = self.hybrid_schedule.get_mode_description(self.epoch)
            
            # Track mode for history
            self.gradient_mode_history.append({
                'epoch': self.epoch,
                'mode': avg_metrics['gradient_mode'],
                'delta': self.hybrid_schedule.get_delta(self.epoch)
            })
        
        return avg_metrics
    
    def _train_batch(self,
                    inputs: List[TRScalar],
                    targets: List[TRScalar],
                    coverage_tracker: CoverageTracker) -> Dict[str, float]:
        """
        Train on a single batch with hybrid features.
        
        Args:
            inputs: Batch inputs
            targets: Batch targets
            coverage_tracker: Coverage tracking instance
            
        Returns:
            Batch metrics
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = []
        pole_scores = []
        tags = []
        all_tag_logits = []
        Q_values = []
        
        for x in inputs:
            # Check if model supports full integration
            if hasattr(self.model, 'forward_fully_integrated'):
                # Fully integrated model with all features
                result = self.model.forward_fully_integrated(x)
                predictions.append(result['output'])
                tags.append(result['tag'])
                if 'tag_logits' in result:
                    all_tag_logits.append(result['tag_logits'])
                if 'pole_score' in result:
                    pole_scores.append(result['pole_score'])
                if 'Q_abs' in result:
                    Q_values.append(result['Q_abs'])
            # Check if model supports tag prediction
            elif self.hybrid_config.use_tag_loss and hasattr(self.model, 'forward_with_tag_pred'):
                y, tag, tag_logits = self.model.forward_with_tag_pred(x)
                predictions.append(y)
                tags.append(tag)
                if tag_logits:
                    all_tag_logits.append(tag_logits)
            # Check if model supports pole head
            elif self.hybrid_config.use_pole_head and hasattr(self.model, 'forward_with_pole_score'):
                y, tag, pole_score = self.model.forward_with_pole_score(x)
                predictions.append(y)
                pole_scores.append(pole_score)
                tags.append(tag)
                # Try to get Q value if available
                if hasattr(self.model, 'get_Q_value'):
                    q_val = self.model.get_Q_value()
                    if q_val is not None:
                        Q_values.append(q_val)
            else:
                y = self.model(x)
                predictions.append(y)
                tags.append(y.tag)
        
        # Track coverage
        coverage_tracker.update(tags)
        
        # Compute main loss
        if self.loss_policy:
            batch_loss = self.loss_policy.compute_batch_loss(predictions, targets)
        else:
            # Simple MSE loss
            from ..core import tr_sum, tr_div
            losses = []
            for pred, target in zip(predictions, targets):
                if pred.tag == TRTag.REAL:
                    diff = pred - to_trnode_constant(target)
                    loss = TRNode.constant(real(0.5)) * diff * diff
                else:
                    loss = TRNode.constant(real(self.config.initial_lambda))
                losses.append(loss)
            
            total = tr_sum([l.value for l in losses])
            batch_loss = TRNode.constant(tr_div(total, real(float(len(losses)))))
        
        # Add tag loss if enabled
        tag_loss_value = 0.0
        if self.hybrid_config.use_tag_loss:
            tag_loss = self._compute_tag_loss(predictions, all_tag_logits)
            if tag_loss is not None:
                batch_loss = batch_loss + tag_loss  # Already weighted in compute_tag_loss
        
        # Compute pole detection loss if enabled
        if self.hybrid_config.use_pole_head and pole_scores:
            pole_loss = self._compute_pole_loss(
                predictions, pole_scores, Q_values, inputs
            )
            if pole_loss is not None:
                weighted_pole_loss = tr_mul(
                    TRNode.constant(real(self.hybrid_config.lambda_pole)),
                    pole_loss
                )
                batch_loss = batch_loss + weighted_pole_loss
                pole_loss_value = pole_loss.value.value if pole_loss.value.tag == TRTag.REAL else 0.0
        else:
            pole_loss_value = 0.0
        
        # Compute residual consistency loss if enabled
        residual_loss_value = 0.0
        if self.hybrid_config.enable_anti_illusion and self.residual_loss:
            # Extract input values for residual computation
            input_vals = []
            for x in inputs:
                if x.value.tag == TRTag.REAL:
                    input_vals.append(x.value.value)
            
            if input_vals:
                residual_loss = self.residual_loss.compute_loss(
                    self.model, input_vals, 
                    self.hybrid_config.residual_near_pole_threshold
                )
                batch_loss = batch_loss + residual_loss
                residual_loss_value = (
                    residual_loss.value.value if residual_loss.tag == TRTag.REAL else 0.0
                )
        
        # Add regularization
        if hasattr(self.model, 'regularization_loss'):
            reg_loss = self.model.regularization_loss()
            batch_loss = batch_loss + reg_loss
        
        # Backward pass
        batch_loss.backward()
        
        # Optimizer step
        self.optimizer.step(self.model)
        
        # Collect metrics
        metrics = {
            'loss': batch_loss.value.value if batch_loss.value.tag == TRTag.REAL else float('inf'),
            'coverage': coverage_tracker.batch_coverage,
            'tag_loss': tag_loss_value,
            'pole_loss': pole_loss_value,
            'residual_loss': residual_loss_value
        }
        
        # Add hybrid metrics
        if self.hybrid_schedule:
            hybrid_stats = HybridGradientContext.get_statistics()
            metrics['near_pole_ratio'] = hybrid_stats.get('near_pole_ratio', 0.0)
        
        # Add adaptive lambda if using policy
        if self.loss_policy:
            metrics['lambda_rej'] = self.loss_policy.adaptive_lambda.get_penalty()
        
        return metrics
    
    def _compute_tag_loss(self, 
                         predictions: List[TRNode],
                         tag_logits: List[List[TRNode]]) -> Optional[TRNode]:
        """
        Compute auxiliary loss for tag classification.
        
        This encourages the model to correctly predict the type
        of singularity (PINF vs NINF vs PHI).
        
        Args:
            predictions: Model predictions (for true tags)
            tag_logits: Predicted tag logits from tag head
            
        Returns:
            Tag classification loss or None
        """
        if not tag_logits:
            # Fallback to simple penalty if no tag head
            tags = [pred.tag for pred in predictions]
            non_real_count = sum(1 for tag in tags if tag != TRTag.REAL)
            if non_real_count > 0:
                penalty = float(non_real_count) / len(tags)
                return TRNode.constant(real(penalty))
            return None
        
        # Use proper tag loss computation
        from ..training.tag_loss import compute_tag_loss
        return compute_tag_loss(predictions, tag_logits, 
                               weight=self.hybrid_config.lambda_tag)
    
    def _compute_pole_loss(self,
                          predictions: List[TRNode],
                          pole_scores: List[TRNode],
                          Q_values: List[float],
                          inputs: List[TRNode]) -> Optional[TRNode]:
        """
        Compute pole detection loss.
        
        Args:
            predictions: Model outputs
            pole_scores: Predicted pole scores
            Q_values: Absolute Q values for self-supervision
            inputs: Input values for teacher signals
            
        Returns:
            Pole detection loss or None
        """
        if not pole_scores:
            return None
        
        # Get teacher labels if available
        teacher_labels = None
        if self.pole_detector and self.hybrid_config.use_teacher_signals:
            # Extract scalar values from inputs
            input_vals = []
            for x in inputs:
                if x.value.tag == TRTag.REAL:
                    input_vals.append(x.value.value)
                else:
                    input_vals.append(0.0)  # Default for non-REAL
            
            teacher_labels = self.pole_detector.generate_labels(input_vals)
        
        # Compute pole loss using the proper function
        pole_loss = compute_pole_loss(
            predictions,
            pole_scores,
            Q_values if Q_values else None,
            teacher_labels,
            self.hybrid_config.pole_config
        )
        
        return pole_loss
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary including hybrid metrics.
        
        Returns:
            Dictionary with training statistics
        """
        summary = {
            'epochs_trained': self.epoch,
            'global_steps': self.global_step,
            'final_metrics': self.history
        }
        
        # Add hybrid gradient history
        if self.gradient_mode_history:
            summary['gradient_modes'] = self.gradient_mode_history
            
            # Analyze transition
            warmup_end = next((i for i, m in enumerate(self.gradient_mode_history) 
                             if 'transitioning' in m['mode']), -1)
            transition_end = next((i for i, m in enumerate(self.gradient_mode_history)
                                 if 'converged' in m['mode']), -1)
            
            summary['warmup_epochs'] = warmup_end if warmup_end >= 0 else self.epoch
            summary['transition_complete'] = transition_end if transition_end >= 0 else None
        
        # Add model-specific metrics
        if hasattr(self.model, 'get_hybrid_statistics'):
            summary['model_statistics'] = self.model.get_hybrid_statistics()
        
        # Add coverage enforcement statistics
        if self.coverage_policy:
            summary['coverage_enforcement'] = self.coverage_policy.get_statistics()
        
        # Add anti-illusion metrics history
        if self.anti_illusion_metrics:
            summary['anti_illusion_trends'] = self.anti_illusion_metrics.get_trends()
            if self.anti_illusion_metrics.evaluation_history:
                latest = self.anti_illusion_metrics.evaluation_history[-1]
                summary['latest_anti_illusion'] = latest
        
        return summary
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint including hybrid state."""
        import pickle
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'history': self.history,
            'gradient_mode_history': self.gradient_mode_history,
            'model_state': self._get_model_state(),
            'optimizer_state': self._get_optimizer_state(),
            'hybrid_schedule': self.hybrid_schedule
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint including hybrid state."""
        import pickle
        
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint['history']
        self.gradient_mode_history = checkpoint.get('gradient_mode_history', [])
        
        if 'hybrid_schedule' in checkpoint:
            self.hybrid_schedule = checkpoint['hybrid_schedule']
            if hasattr(self.model, 'hybrid_schedule'):
                self.model.hybrid_schedule = self.hybrid_schedule
        
        # Restore model and optimizer states
        self._set_model_state(checkpoint['model_state'])
        self._set_optimizer_state(checkpoint['optimizer_state'])
        
        # Update hybrid context
        if self.hybrid_schedule:
            HybridGradientContext.set_schedule(self.hybrid_schedule)
            HybridGradientContext.update_epoch(self.epoch)
    
    def _get_model_state(self) -> Dict:
        """Get model state for checkpointing."""
        state = {}
        if hasattr(self.model, 'parameters'):
            for i, param in enumerate(self.model.parameters()):
                state[f'param_{i}'] = param.value
        return state
    
    def _set_model_state(self, state: Dict) -> None:
        """Set model state from checkpoint."""
        if hasattr(self.model, 'parameters'):
            for i, param in enumerate(self.model.parameters()):
                if f'param_{i}' in state:
                    param._value = state[f'param_{i}']
    
    def _get_optimizer_state(self) -> Dict:
        """Get optimizer state for checkpointing."""
        return {
            'learning_rate': self.optimizer.learning_rate,
            'step_count': self.optimizer.step_count
        }
    
    def _set_optimizer_state(self, state: Dict) -> None:
        """Set optimizer state from checkpoint."""
        self.optimizer.learning_rate = state['learning_rate']
        self.optimizer.step_count = state['step_count']
