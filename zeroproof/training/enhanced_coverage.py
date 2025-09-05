"""
Enhanced coverage control with adaptive lambda and near-pole sampling.

This module implements sophisticated coverage control mechanisms to prevent
the model from trivially rejecting too many near-pole points while maintaining
training stability.
"""

from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

from ..core import TRScalar, TRTag, real
from ..autodiff import TRNode


class CoverageStrategy(Enum):
    """Strategies for maintaining target coverage."""
    LAGRANGE = "lagrange"      # Lagrange multiplier approach
    PID = "pid"                # PID controller
    ADAPTIVE_RATE = "adaptive" # Adaptive learning rate
    DUAL_PHASE = "dual"        # Different strategies for warmup vs training


@dataclass
class EnhancedCoverageConfig:
    """
    Configuration for enhanced coverage control.
    
    Attributes:
        target_coverage: Desired proportion of REAL outputs
        min_coverage: Minimum acceptable coverage (triggers intervention)
        max_coverage: Maximum acceptable coverage (for balance)
        strategy: Control strategy to use
        lambda_init: Initial rejection penalty
        lambda_min: Minimum lambda value
        lambda_max: Maximum lambda value
        learning_rate: Learning rate for lambda updates
        momentum: Momentum for lambda updates
        pid_gains: PID controller gains (Kp, Ki, Kd)
        warmup_epochs: Epochs before enforcing coverage
        window_size: Window for moving average coverage
        oversample_near_pole: Whether to oversample near-pole regions
        pole_threshold: Threshold for identifying near-pole samples
    """
    target_coverage: float = 0.85
    min_coverage: float = 0.70
    max_coverage: float = 0.95
    strategy: CoverageStrategy = CoverageStrategy.LAGRANGE
    lambda_init: float = 1.0
    lambda_min: float = 0.0
    lambda_max: float = 10.0
    learning_rate: float = 0.01
    momentum: float = 0.9
    pid_gains: Tuple[float, float, float] = (1.0, 0.1, 0.01)
    warmup_epochs: int = 10
    window_size: int = 50
    oversample_near_pole: bool = True
    pole_threshold: float = 0.1


class AdaptiveLambdaController:
    """
    Advanced controller for rejection penalty lambda.
    
    This implements multiple control strategies to maintain target coverage
    while preventing trivial rejection of difficult samples.
    """
    
    def __init__(self, config: EnhancedCoverageConfig):
        """
        Initialize adaptive lambda controller.
        
        Args:
            config: Coverage control configuration
        """
        self.config = config
        self.lambda_value = config.lambda_init
        
        # State tracking
        self.epoch = 0
        self.coverage_history = []
        self.lambda_history = [self.lambda_value]
        self.error_history = []
        
        # Momentum state
        self.velocity = 0.0
        
        # PID state
        self.integral_error = 0.0
        self.prev_error = 0.0
        
        # Statistics
        self.adjustments_made = 0
        self.coverage_violations = 0
    
    def update(self, current_coverage: float) -> None:
        """
        Update lambda based on current coverage.
        
        Args:
            current_coverage: Current proportion of REAL outputs
        """
        self.coverage_history.append(current_coverage)
        
        # Check if we're still in warmup
        if self.epoch < self.config.warmup_epochs:
            self.epoch += 1
            return
        
        # Calculate error
        error = self.config.target_coverage - current_coverage
        self.error_history.append(error)
        
        # Check for coverage violations
        if current_coverage < self.config.min_coverage:
            self.coverage_violations += 1
        
        # Apply strategy-specific update
        if self.config.strategy == CoverageStrategy.LAGRANGE:
            self._lagrange_update(error)
        elif self.config.strategy == CoverageStrategy.PID:
            self._pid_update(error)
        elif self.config.strategy == CoverageStrategy.ADAPTIVE_RATE:
            self._adaptive_rate_update(error, current_coverage)
        elif self.config.strategy == CoverageStrategy.DUAL_PHASE:
            self._dual_phase_update(error, current_coverage)
        
        # Clamp lambda
        self.lambda_value = max(
            self.config.lambda_min,
            min(self.config.lambda_max, self.lambda_value)
        )
        
        # Record
        self.lambda_history.append(self.lambda_value)
        self.adjustments_made += 1
        self.epoch += 1
    
    def _lagrange_update(self, error: float) -> None:
        """
        Lagrange multiplier update: λ ← λ + η * (c* - c_actual)
        
        Args:
            error: Coverage error (target - actual)
        """
        # Update with momentum
        # Use sign so that when coverage is low (error>0) lambda decreases,
        # and when coverage is high (error<0) lambda increases.
        self.velocity = (self.config.momentum * self.velocity - 
                        self.config.learning_rate * error)
        self.lambda_value += self.velocity
        # Additional gentle damping to avoid overshoot in tests
        if self.lambda_value > self.config.lambda_max:
            self.lambda_value = self.config.lambda_max
        if self.lambda_value < self.config.lambda_min:
            self.lambda_value = self.config.lambda_min
    
    def _pid_update(self, error: float) -> None:
        """
        PID controller update.
        
        Args:
            error: Coverage error
        """
        Kp, Ki, Kd = self.config.pid_gains
        
        # Proportional term
        P = Kp * error
        
        # Integral term
        self.integral_error += error
        I = Ki * self.integral_error
        
        # Derivative term
        D = Kd * (error - self.prev_error)
        self.prev_error = error
        
        # Update lambda
        self.lambda_value += P + I + D
    
    def _adaptive_rate_update(self, error: float, coverage: float) -> None:
        """
        Adaptive learning rate based on coverage gap magnitude.
        
        Args:
            error: Coverage error
            coverage: Current coverage
        """
        # Adaptive learning rate: larger updates for larger gaps
        if coverage < self.config.min_coverage:
            # Emergency: large decrease in lambda
            adaptive_lr = self.config.learning_rate * 5.0
        elif coverage < self.config.target_coverage - 0.1:
            # Below target: moderate decrease
            adaptive_lr = self.config.learning_rate * 2.0
        elif coverage > self.config.max_coverage:
            # Above max: increase lambda to reduce coverage
            adaptive_lr = self.config.learning_rate * 2.0
        else:
            # Near target: normal rate
            adaptive_lr = self.config.learning_rate
        
        # Update with adaptive rate
        self.velocity = self.config.momentum * self.velocity + adaptive_lr * error
        self.lambda_value += self.velocity
    
    def _dual_phase_update(self, error: float, coverage: float) -> None:
        """
        Different strategies for different training phases.
        
        Args:
            error: Coverage error
            coverage: Current coverage
        """
        if self.epoch < self.config.warmup_epochs * 2:
            # Early phase: aggressive PID
            self._pid_update(error * 2.0)
        else:
            # Later phase: smooth Lagrange
            self._lagrange_update(error)
    
    def get_penalty(self) -> float:
        """Get current rejection penalty."""
        return self.lambda_value
    
    def get_statistics(self) -> Dict[str, float]:
        """Get controller statistics."""
        stats = {
            'lambda': self.lambda_value,
            'adjustments': self.adjustments_made,
            'violations': self.coverage_violations,
            'epoch': self.epoch
        }
        
        if self.coverage_history:
            stats['avg_coverage'] = np.mean(self.coverage_history)
            stats['min_coverage'] = np.min(self.coverage_history)
            stats['max_coverage'] = np.max(self.coverage_history)
            stats['coverage_std'] = np.std(self.coverage_history)
        
        if self.error_history:
            stats['avg_error'] = np.mean(self.error_history)
            stats['error_std'] = np.std(self.error_history)
        
        return stats
    
    def reset(self) -> None:
        """Reset controller state."""
        self.lambda_value = self.config.lambda_init
        self.epoch = 0
        self.coverage_history.clear()
        self.lambda_history = [self.lambda_value]
        self.error_history.clear()
        self.velocity = 0.0
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.adjustments_made = 0
        self.coverage_violations = 0


class NearPoleSampler:
    """
    Intelligent sampler that oversamples near-pole regions.
    
    This helps maintain coverage by ensuring the model sees enough
    near-pole samples to learn proper behavior rather than rejecting them.
    """
    
    def __init__(self,
                 pole_threshold: float = 0.1,
                 oversample_ratio: float = 2.0,
                 adaptive: bool = True):
        """
        Initialize near-pole sampler.
        
        Args:
            pole_threshold: |Q| threshold for near-pole identification
            oversample_ratio: How much to oversample near-pole regions
            adaptive: Whether to adapt ratio based on coverage
        """
        self.pole_threshold = pole_threshold
        self.base_oversample_ratio = oversample_ratio
        self.oversample_ratio = oversample_ratio
        self.adaptive = adaptive
        
        # Tracking
        self.near_pole_indices = []
        self.sample_weights = []
        self.coverage_history = []
    
    def compute_sample_weights(self,
                              Q_values: List[float],
                              current_coverage: Optional[float] = None) -> np.ndarray:
        """
        Compute sampling weights based on proximity to poles.
        
        Args:
            Q_values: Denominator values |Q(x)| for each sample
            current_coverage: Current coverage for adaptive adjustment
            
        Returns:
            Sample weights for weighted sampling
        """
        weights = np.ones(len(Q_values))
        
        # Identify near-pole samples
        self.near_pole_indices = []
        for i, q_val in enumerate(Q_values):
            if abs(q_val) <= self.pole_threshold:
                self.near_pole_indices.append(i)
                weights[i] = self.oversample_ratio
        
        # Adaptive adjustment
        if self.adaptive and current_coverage is not None:
            self.coverage_history.append(current_coverage)
            
            # Increase oversampling if coverage is too low
            if current_coverage < 0.7:
                self.oversample_ratio = self.base_oversample_ratio * 1.5
            elif current_coverage < 0.8:
                self.oversample_ratio = self.base_oversample_ratio * 1.2
            else:
                self.oversample_ratio = self.base_oversample_ratio
            
            # Update weights
            for i in self.near_pole_indices:
                weights[i] = self.oversample_ratio
        
        # Normalize weights
        weights = weights / weights.sum()
        self.sample_weights = weights
        
        return weights
    
    def sample_batch(self,
                    data: List[Tuple],
                    batch_size: int,
                    Q_values: Optional[List[float]] = None) -> List[Tuple]:
        """
        Sample a batch with oversampling of near-pole regions.
        
        Args:
            data: List of (input, target) tuples
            batch_size: Batch size
            Q_values: Optional pre-computed |Q| values
            
        Returns:
            Sampled batch
        """
        if Q_values is None:
            # Uniform sampling if no Q values provided
            indices = np.random.choice(len(data), batch_size, replace=True)
        else:
            # Weighted sampling based on Q values
            weights = self.compute_sample_weights(Q_values)
            indices = np.random.choice(
                len(data), batch_size, replace=True, p=weights
            )
        
        return [data[i] for i in indices]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get sampler statistics."""
        stats = {
            'oversample_ratio': self.oversample_ratio,
            'near_pole_count': len(self.near_pole_indices),
            'near_pole_ratio': len(self.near_pole_indices) / len(self.sample_weights) 
                              if self.sample_weights else 0
        }
        
        if self.coverage_history:
            stats['avg_coverage_seen'] = np.mean(self.coverage_history)
        
        return stats


class CoverageEnforcementPolicy:
    """
    High-level policy for enforcing coverage constraints.
    
    Combines lambda control, sampling strategy, and intervention mechanisms.
    """
    
    def __init__(self,
                 config: EnhancedCoverageConfig,
                 lambda_controller: Optional[AdaptiveLambdaController] = None,
                 sampler: Optional[NearPoleSampler] = None):
        """
        Initialize enforcement policy.
        
        Args:
            config: Coverage configuration
            lambda_controller: Lambda controller (creates default if None)
            sampler: Near-pole sampler (creates default if None)
        """
        self.config = config
        
        self.lambda_controller = lambda_controller or AdaptiveLambdaController(config)
        self.sampler = sampler or NearPoleSampler(
            pole_threshold=config.pole_threshold,
            oversample_ratio=2.0
        ) if config.oversample_near_pole else None
        
        # Intervention tracking
        self.interventions = []
        self.coverage_restored_epoch = None
    
    def enforce(self,
               current_coverage: float,
               epoch: int,
               Q_values: Optional[List[float]] = None) -> Dict[str, any]:
        """
        Apply coverage enforcement policy.
        
        Args:
            current_coverage: Current coverage
            epoch: Current epoch
            Q_values: Optional denominator values
            
        Returns:
            Dictionary of actions taken
        """
        actions = {
            'lambda_updated': False,
            'sampling_adjusted': False,
            'intervention_triggered': False,
            'new_lambda': self.lambda_controller.get_penalty()
        }
        
        # Update lambda controller
        self.lambda_controller.update(current_coverage)
        actions['lambda_updated'] = True
        actions['new_lambda'] = self.lambda_controller.get_penalty()
        
        # Check for critical coverage violation
        if current_coverage < self.config.min_coverage:
            actions['intervention_triggered'] = True
            self._trigger_intervention(current_coverage, epoch)
        
        # Adjust sampling if enabled
        if self.sampler and Q_values:
            weights = self.sampler.compute_sample_weights(
                Q_values, current_coverage
            )
            actions['sampling_adjusted'] = True
            actions['sample_weights'] = weights
        
        # Check if coverage restored
        if (self.coverage_restored_epoch is None and 
            current_coverage >= self.config.target_coverage):
            self.coverage_restored_epoch = epoch
        
        return actions
    
    def _trigger_intervention(self, coverage: float, epoch: int) -> None:
        """
        Trigger emergency intervention for critical coverage loss.
        
        Args:
            coverage: Current coverage
            epoch: Current epoch
        """
        intervention = {
            'epoch': epoch,
            'coverage': coverage,
            'action': 'emergency_lambda_reduction'
        }
        
        # Emergency lambda reduction
        self.lambda_controller.lambda_value *= 0.5
        
        # Reset PID if using PID strategy
        if self.config.strategy == CoverageStrategy.PID:
            self.lambda_controller.integral_error = 0
        
        self.interventions.append(intervention)
    
    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive policy statistics."""
        stats = {
            'lambda_stats': self.lambda_controller.get_statistics(),
            'interventions': len(self.interventions),
            'coverage_restored_epoch': self.coverage_restored_epoch
        }
        
        if self.sampler:
            stats['sampler_stats'] = self.sampler.get_statistics()
        
        return stats
    
    def reset(self) -> None:
        """Reset policy state."""
        self.lambda_controller.reset()
        if self.sampler:
            self.sampler.coverage_history.clear()
        self.interventions.clear()
        self.coverage_restored_epoch = None
