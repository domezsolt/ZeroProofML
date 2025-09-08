"""
Advanced control strategies for loss and curriculum learning.

This module implements PI controllers, dead-band control, and curriculum
learning strategies for improved training stability near poles.
"""

from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import math

from ..core import TRTag


class ControlStrategy(Enum):
    """Available control strategies for λ_rej."""
    PROPORTIONAL = "proportional"
    PI = "pi"
    PID = "pid"
    DEAD_BAND = "dead_band"
    ADAPTIVE = "adaptive"


@dataclass
class PIControllerConfig:
    """Configuration for PI controller."""
    # Controller gains
    kp: float = 1.0  # Proportional gain
    ki: float = 0.1  # Integral gain
    kd: float = 0.0  # Derivative gain (for PID)
    
    # Limits
    output_min: float = 0.1  # Minimum λ_rej
    output_max: float = 10.0  # Maximum λ_rej
    integral_limit: float = 5.0  # Anti-windup limit
    
    # Dead-band
    dead_band: float = 0.02  # Dead-band around target
    
    # Smoothing
    error_smoothing: float = 0.0  # EMA smoothing for error (0 = no smoothing)
    output_smoothing: float = 0.0  # EMA smoothing for output
    
    # Target
    target_coverage: float = 0.85
    
    # Adaptive features
    adaptive_gains: bool = False  # Adapt gains based on performance
    gain_adaptation_rate: float = 0.01


class PIController:
    """
    PI/PID controller for λ_rej adjustment.
    
    Provides more stable control than pure proportional adjustment
    by incorporating integral (and optionally derivative) terms.
    """
    
    def __init__(self, config: Optional[PIControllerConfig] = None):
        """
        Initialize PI controller.
        
        Args:
            config: Controller configuration
        """
        self.config = config or PIControllerConfig()
        
        # Controller state
        self.integral = 0.0
        self.last_error = 0.0
        self.smoothed_error = 0.0
        self.smoothed_output = self.config.output_min
        
        # History for analysis
        self.error_history = deque(maxlen=100)
        self.output_history = deque(maxlen=100)
        self.coverage_history = deque(maxlen=100)
        
        # Performance metrics
        self.total_adjustments = 0
        self.oscillation_count = 0
        self.time_in_dead_band = 0
        
    def compute(self, current_coverage: float) -> Tuple[float, Dict[str, float]]:
        """
        Compute control output using PI/PID algorithm.
        
        Args:
            current_coverage: Current REAL coverage
            
        Returns:
            Tuple of (λ_rej value, debug info)
        """
        # Compute error (negative because we want to increase λ when coverage is high)
        error = current_coverage - self.config.target_coverage
        
        # Apply dead-band
        if abs(error) < self.config.dead_band:
            self.time_in_dead_band += 1
            # Don't update integral in dead-band to prevent windup
            return self.smoothed_output, {
                'error': error,
                'in_dead_band': True,
                'p_term': 0.0,
                'i_term': 0.0,
                'd_term': 0.0,
            }
        
        # Smooth error if configured
        if self.config.error_smoothing > 0:
            self.smoothed_error = (self.config.error_smoothing * self.smoothed_error + 
                                  (1 - self.config.error_smoothing) * error)
            error = self.smoothed_error
        
        # Proportional term
        p_term = self.config.kp * error
        
        # Integral term with anti-windup
        self.integral += error
        self.integral = np.clip(self.integral, 
                               -self.config.integral_limit, 
                               self.config.integral_limit)
        i_term = self.config.ki * self.integral
        
        # Derivative term (if PID)
        d_term = 0.0
        if self.config.kd > 0:
            if self.last_error != 0:  # Skip first iteration
                d_term = self.config.kd * (error - self.last_error)
        
        # Compute output
        output = self.smoothed_output + p_term + i_term + d_term
        
        # Apply limits
        output = np.clip(output, self.config.output_min, self.config.output_max)
        
        # Smooth output if configured
        if self.config.output_smoothing > 0:
            self.smoothed_output = (self.config.output_smoothing * self.smoothed_output + 
                                   (1 - self.config.output_smoothing) * output)
            output = self.smoothed_output
        else:
            self.smoothed_output = output
        
        # Update state
        self.last_error = error
        
        # Update history
        self.error_history.append(error)
        self.output_history.append(output)
        self.coverage_history.append(current_coverage)
        
        # Detect oscillations
        if len(self.error_history) > 3:
            recent_errors = list(self.error_history)[-4:]
            sign_changes = sum(1 for i in range(1, len(recent_errors)) 
                             if np.sign(recent_errors[i]) != np.sign(recent_errors[i-1]))
            if sign_changes >= 3:
                self.oscillation_count += 1
                
                # Adaptive gain reduction if oscillating
                if self.config.adaptive_gains:
                    self.config.kp *= (1 - self.config.gain_adaptation_rate)
                    self.config.ki *= (1 - self.config.gain_adaptation_rate)
        
        self.total_adjustments += 1
        
        return output, {
            'error': error,
            'in_dead_band': False,
            'p_term': p_term,
            'i_term': i_term,
            'd_term': d_term,
            'integral': self.integral,
            'oscillations': self.oscillation_count,
        }
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.smoothed_error = 0.0
        self.smoothed_output = self.config.output_min
        self.error_history.clear()
        self.output_history.clear()
        self.coverage_history.clear()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get controller performance statistics."""
        if not self.error_history:
            return {}
        
        errors = list(self.error_history)
        outputs = list(self.output_history)
        
        return {
            'mean_error': np.mean(np.abs(errors)),
            'std_error': np.std(errors),
            'mean_output': np.mean(outputs),
            'std_output': np.std(outputs),
            'oscillation_rate': self.oscillation_count / max(1, self.total_adjustments),
            'dead_band_rate': self.time_in_dead_band / max(1, self.total_adjustments),
            'integral_saturation': abs(self.integral) / self.config.integral_limit,
        }


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    # Difficulty stages
    n_stages: int = 5  # Number of difficulty stages
    stage_duration: int = 20  # Epochs per stage
    
    # Pole distance thresholds
    easy_distance: float = 1.0  # Far from poles
    hard_distance: float = 0.01  # Very close to poles
    
    # Sampling strategy
    initial_easy_ratio: float = 0.9  # Start with 90% easy samples
    final_easy_ratio: float = 0.2  # End with 20% easy samples
    
    # Progression
    linear_progression: bool = False  # Linear vs exponential progression
    warmup_epochs: int = 5  # Initial warmup with only easy samples
    
    # Adaptive progression
    adaptive: bool = True  # Adapt based on performance
    min_coverage_for_advance: float = 0.7  # Min coverage to advance stage
    max_loss_for_advance: float = 1.0  # Max loss to advance stage
    
    # Sample weighting
    use_importance_weights: bool = True  # Weight hard samples more
    importance_temperature: float = 1.0  # Temperature for importance weights


class CurriculumScheduler:
    """
    Curriculum learning scheduler for gradual pole introduction.
    
    Starts with easy samples (far from poles) and gradually
    introduces harder samples (near poles) as training progresses.
    """
    
    def __init__(self, 
                 config: Optional[CurriculumConfig] = None,
                 pole_locations: Optional[List[float]] = None):
        """
        Initialize curriculum scheduler.
        
        Args:
            config: Curriculum configuration
            pole_locations: Known pole locations for distance computation
        """
        self.config = config or CurriculumConfig()
        self.pole_locations = pole_locations or []
        
        # State
        self.current_epoch = 0
        self.current_stage = 0
        self.stage_start_epoch = 0
        
        # Performance tracking
        self.stage_performance = []
        self.coverage_history = []
        self.loss_history = []
        
    def get_difficulty_threshold(self) -> float:
        """
        Get current difficulty threshold (distance from poles).
        
        Returns:
            Maximum distance from poles for "hard" samples
        """
        if self.current_epoch < self.config.warmup_epochs:
            # Warmup period - only easy samples
            return self.config.easy_distance
        
        # Compute progression
        total_stages = self.config.n_stages
        
        if self.config.linear_progression:
            # Linear progression
            progress = self.current_stage / max(1, total_stages - 1)
        else:
            # Exponential progression (faster initial progress)
            progress = (np.exp(self.current_stage / (total_stages - 1)) - 1) / (np.e - 1)
        
        # Interpolate threshold
        threshold = (self.config.easy_distance * (1 - progress) + 
                    self.config.hard_distance * progress)
        
        return threshold
    
    def get_sample_weights(self, 
                          x_values: List[float],
                          q_values: Optional[List[float]] = None) -> np.ndarray:
        """
        Get sampling weights based on curriculum.
        
        Args:
            x_values: Input values
            q_values: Optional Q values for pole proximity
            
        Returns:
            Sampling weights for each sample
        """
        n_samples = len(x_values)
        weights = np.ones(n_samples)
        
        if not self.pole_locations and q_values is None:
            return weights / n_samples  # Uniform if no pole info
        
        # Compute distances to poles
        distances = []
        for x in x_values:
            if self.pole_locations:
                min_dist = min(abs(x - pole) for pole in self.pole_locations)
            elif q_values:
                # Use Q values as proxy for pole distance
                idx = x_values.index(x)
                min_dist = abs(q_values[idx]) if idx < len(q_values) else 1.0
            else:
                min_dist = 1.0
            distances.append(min_dist)
        
        distances = np.array(distances)
        
        # Get current difficulty threshold
        threshold = self.get_difficulty_threshold()
        
        # Classify samples
        hard_mask = distances <= threshold
        easy_mask = ~hard_mask
        
        # Compute target ratio
        progress = self.current_stage / max(1, self.config.n_stages - 1)
        easy_ratio = (self.config.initial_easy_ratio * (1 - progress) + 
                     self.config.final_easy_ratio * progress)
        hard_ratio = 1 - easy_ratio
        
        # Assign weights
        n_hard = hard_mask.sum()
        n_easy = easy_mask.sum()
        
        if n_hard > 0:
            weights[hard_mask] = hard_ratio / n_hard
        if n_easy > 0:
            weights[easy_mask] = easy_ratio / n_easy
        
        # Apply importance weighting if configured
        if self.config.use_importance_weights and n_hard > 0:
            # Increase weight of harder samples within hard category
            hard_indices = np.where(hard_mask)[0]
            hard_distances = distances[hard_mask]
            
            # Compute importance (inverse distance with temperature)
            importance = np.exp(-hard_distances / self.config.importance_temperature)
            importance = importance / importance.sum()
            
            # Modulate hard weights by importance
            base_weight = weights[hard_indices[0]]
            for idx, imp in zip(hard_indices, importance):
                weights[idx] = base_weight * imp * len(hard_indices)
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    def should_advance_stage(self, 
                            current_coverage: float,
                            current_loss: float) -> bool:
        """
        Check if ready to advance to next stage.
        
        Args:
            current_coverage: Current REAL coverage
            current_loss: Current training loss
            
        Returns:
            True if should advance to next stage
        """
        if not self.config.adaptive:
            # Fixed progression
            epochs_in_stage = self.current_epoch - self.stage_start_epoch
            return epochs_in_stage >= self.config.stage_duration
        
        # Adaptive progression based on performance
        if current_coverage < self.config.min_coverage_for_advance:
            return False
        
        if current_loss > self.config.max_loss_for_advance:
            return False
        
        # Minimum time in stage
        epochs_in_stage = self.current_epoch - self.stage_start_epoch
        min_epochs = self.config.stage_duration // 2
        
        return epochs_in_stage >= min_epochs
    
    def update(self, 
              epoch: int,
              coverage: float,
              loss: float) -> Dict[str, Any]:
        """
        Update curriculum state.
        
        Args:
            epoch: Current epoch
            coverage: Current REAL coverage
            loss: Current loss
            
        Returns:
            Curriculum status info
        """
        self.current_epoch = epoch
        self.coverage_history.append(coverage)
        self.loss_history.append(loss)
        
        # Check for stage advancement
        advanced = False
        if (self.current_stage < self.config.n_stages - 1 and
            self.should_advance_stage(coverage, loss)):
            self.current_stage += 1
            self.stage_start_epoch = epoch
            advanced = True
            
            # Record stage performance
            self.stage_performance.append({
                'stage': self.current_stage - 1,
                'duration': epoch - self.stage_start_epoch,
                'final_coverage': coverage,
                'final_loss': loss,
            })
        
        # Get current state
        threshold = self.get_difficulty_threshold()
        progress = self.current_stage / max(1, self.config.n_stages - 1)
        
        return {
            'stage': self.current_stage,
            'progress': progress,
            'difficulty_threshold': threshold,
            'advanced': advanced,
            'epochs_in_stage': epoch - self.stage_start_epoch,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get curriculum summary statistics."""
        if not self.coverage_history:
            return {}
        
        return {
            'current_stage': self.current_stage,
            'total_stages': self.config.n_stages,
            'mean_coverage': np.mean(self.coverage_history),
            'coverage_trend': np.polyfit(range(len(self.coverage_history)), 
                                        self.coverage_history, 1)[0] if len(self.coverage_history) > 1 else 0,
            'stage_performance': self.stage_performance,
        }


class HybridController:
    """
    Hybrid controller combining multiple strategies.
    
    Can combine PI control with dead-band and curriculum learning
    for comprehensive training control.
    """
    
    def __init__(self,
                 pi_config: Optional[PIControllerConfig] = None,
                 curriculum_config: Optional[CurriculumConfig] = None,
                 pole_locations: Optional[List[float]] = None):
        """
        Initialize hybrid controller.
        
        Args:
            pi_config: PI controller configuration
            curriculum_config: Curriculum configuration
            pole_locations: Known pole locations
        """
        self.pi_controller = PIController(pi_config) if pi_config else None
        self.curriculum = CurriculumScheduler(curriculum_config, pole_locations) if curriculum_config else None
        
        # Combined history
        self.epoch_history = []
        
    def update(self,
              epoch: int,
              coverage: float,
              loss: float,
              near_pole_coverage: Optional[float] = None) -> Dict[str, Any]:
        """
        Update all controllers.
        
        Args:
            epoch: Current epoch
            coverage: Global REAL coverage
            loss: Current loss
            near_pole_coverage: Optional near-pole coverage
            
        Returns:
            Combined control outputs
        """
        result = {
            'epoch': epoch,
            'coverage': coverage,
            'loss': loss,
        }
        
        # PI control for λ_rej
        if self.pi_controller:
            lambda_rej, pi_info = self.pi_controller.compute(coverage)
            result['lambda_rej'] = lambda_rej
            result['pi_control'] = pi_info
        
        # Curriculum control
        if self.curriculum:
            curriculum_info = self.curriculum.update(epoch, coverage, loss)
            result['curriculum'] = curriculum_info
        
        # Record history
        self.epoch_history.append(result)
        
        return result
    
    def get_sample_weights(self, 
                          x_values: List[float],
                          q_values: Optional[List[float]] = None) -> np.ndarray:
        """
        Get curriculum-based sample weights.
        
        Args:
            x_values: Input values
            q_values: Optional Q values
            
        Returns:
            Sample weights
        """
        if self.curriculum:
            return self.curriculum.get_sample_weights(x_values, q_values)
        else:
            # Uniform weights
            n = len(x_values)
            return np.ones(n) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = {}
        
        if self.pi_controller:
            stats['pi_control'] = self.pi_controller.get_statistics()
        
        if self.curriculum:
            stats['curriculum'] = self.curriculum.get_summary()
        
        if self.epoch_history:
            # Overall trends
            coverages = [h['coverage'] for h in self.epoch_history]
            losses = [h['loss'] for h in self.epoch_history]
            
            stats['overall'] = {
                'mean_coverage': np.mean(coverages),
                'final_coverage': coverages[-1] if coverages else 0,
                'coverage_improvement': coverages[-1] - coverages[0] if len(coverages) > 1 else 0,
                'mean_loss': np.mean(losses),
                'final_loss': losses[-1] if losses else 0,
            }
        
        return stats


def create_advanced_controller(control_type: str = "hybrid",
                              target_coverage: float = 0.85,
                              pole_locations: Optional[List[float]] = None,
                              **kwargs) -> HybridController:
    """
    Factory function to create advanced controller.
    
    Args:
        control_type: Type of control ("pi", "curriculum", "hybrid")
        target_coverage: Target REAL coverage
        pole_locations: Known pole locations
        **kwargs: Additional configuration
        
    Returns:
        Configured controller
    """
    pi_config = None
    curriculum_config = None
    
    if control_type in ["pi", "hybrid"]:
        pi_config = PIControllerConfig(
            target_coverage=target_coverage,
            kp=kwargs.get('kp', 1.0),
            ki=kwargs.get('ki', 0.1),
            kd=kwargs.get('kd', 0.0),
            dead_band=kwargs.get('dead_band', 0.02),
            adaptive_gains=kwargs.get('adaptive_gains', True),
        )
    
    if control_type in ["curriculum", "hybrid"]:
        curriculum_config = CurriculumConfig(
            n_stages=kwargs.get('n_stages', 5),
            stage_duration=kwargs.get('stage_duration', 20),
            adaptive=kwargs.get('adaptive_curriculum', True),
            use_importance_weights=kwargs.get('use_importance_weights', True),
        )
    
    return HybridController(pi_config, curriculum_config, pole_locations)
