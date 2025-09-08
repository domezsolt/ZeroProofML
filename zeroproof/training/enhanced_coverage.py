"""
Enhanced coverage tracking with near-pole monitoring.

This module extends the basic coverage tracking with additional metrics
for near-pole regions and singularity distances.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import deque

from ..core import TRTag


@dataclass
class EnhancedCoverageMetrics:
    """Extended metrics for coverage tracking."""
    # Basic tag counts
    total_samples: int = 0
    real_samples: int = 0
    pinf_samples: int = 0
    ninf_samples: int = 0
    phi_samples: int = 0
    
    # Near-pole tracking
    near_pole_samples: int = 0
    near_pole_real: int = 0
    near_pole_nonreal: int = 0
    
    # Distance tracking
    min_q_value: Optional[float] = None
    mean_q_value: Optional[float] = None
    q_values: List[float] = field(default_factory=list)
    
    # Actual non-REAL outputs tracking
    actual_nonreal_outputs: List[Tuple[int, TRTag]] = field(default_factory=list)
    
    @property
    def coverage(self) -> float:
        """Proportion of REAL samples."""
        if self.total_samples == 0:
            return 1.0
        return self.real_samples / self.total_samples
    
    @property
    def near_pole_coverage(self) -> float:
        """Coverage specifically in near-pole regions."""
        if self.near_pole_samples == 0:
            return 1.0
        return self.near_pole_real / self.near_pole_samples
    
    @property
    def actual_nonreal_rate(self) -> float:
        """Actual rate of non-REAL outputs."""
        if self.total_samples == 0:
            return 0.0
        return len(self.actual_nonreal_outputs) / self.total_samples
    
    @property
    def tag_distribution(self) -> Dict[str, float]:
        """Distribution of tags as proportions."""
        if self.total_samples == 0:
            return {"REAL": 1.0, "PINF": 0.0, "NINF": 0.0, "PHI": 0.0}
        
        return {
            "REAL": self.real_samples / self.total_samples,
            "PINF": self.pinf_samples / self.total_samples,
            "NINF": self.ninf_samples / self.total_samples,
            "PHI": self.phi_samples / self.total_samples,
        }
    
    def update(self, 
               tags: List[TRTag],
               q_values: Optional[List[float]] = None,
               pole_threshold: float = 0.1) -> None:
        """
        Update metrics with new batch of tags and Q values.
        
        Args:
            tags: List of output tags
            q_values: Optional list of |Q(x)| values
            pole_threshold: Threshold for considering as near-pole
        """
        for i, tag in enumerate(tags):
            self.total_samples += 1
            
            # Update tag counts
            if tag == TRTag.REAL:
                self.real_samples += 1
            elif tag == TRTag.PINF:
                self.pinf_samples += 1
                self.actual_nonreal_outputs.append((self.total_samples - 1, tag))
            elif tag == TRTag.NINF:
                self.ninf_samples += 1
                self.actual_nonreal_outputs.append((self.total_samples - 1, tag))
            elif tag == TRTag.PHI:
                self.phi_samples += 1
                self.actual_nonreal_outputs.append((self.total_samples - 1, tag))
            
            # Update Q-value tracking if provided
            if q_values and i < len(q_values):
                q_val = abs(q_values[i])
                self.q_values.append(q_val)
                
                # Update min Q
                if self.min_q_value is None or q_val < self.min_q_value:
                    self.min_q_value = q_val
                
                # Check if near pole
                if q_val <= pole_threshold:
                    self.near_pole_samples += 1
                    if tag == TRTag.REAL:
                        self.near_pole_real += 1
                    else:
                        self.near_pole_nonreal += 1
        
        # Update mean Q value
        if self.q_values:
            self.mean_q_value = np.mean(self.q_values)
    
    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.total_samples = 0
        self.real_samples = 0
        self.pinf_samples = 0
        self.ninf_samples = 0
        self.phi_samples = 0
        self.near_pole_samples = 0
        self.near_pole_real = 0
        self.near_pole_nonreal = 0
        self.min_q_value = None
        self.mean_q_value = None
        self.q_values = []
        self.actual_nonreal_outputs = []


class EnhancedCoverageTracker:
    """
    Enhanced coverage tracker with near-pole monitoring and distance tracking.
    
    This tracker provides:
    - Global REAL coverage
    - Near-pole specific coverage
    - Actual non-REAL output tracking
    - Distance to nearest singularity monitoring
    """
    
    def __init__(self,
                 target_coverage: float = 0.95,
                 pole_threshold: float = 0.1,
                 window_size: Optional[int] = None,
                 track_pole_distances: bool = True):
        """
        Initialize enhanced coverage tracker.
        
        Args:
            target_coverage: Desired proportion of REAL outputs
            pole_threshold: |Q| threshold for near-pole classification
            window_size: Size of sliding window (None for cumulative)
            track_pole_distances: Whether to track distances to poles
        """
        if not 0 <= target_coverage <= 1:
            raise ValueError(f"Target coverage must be in [0,1], got {target_coverage}")
        
        self.target_coverage = target_coverage
        self.pole_threshold = pole_threshold
        self.window_size = window_size
        self.track_pole_distances = track_pole_distances
        
        # Current batch metrics
        self.current_batch = EnhancedCoverageMetrics()
        
        # Cumulative metrics
        self.cumulative = EnhancedCoverageMetrics()
        
        # History for sliding window
        self.history: deque = deque(maxlen=window_size) if window_size else deque()
        
        # Moving averages
        self.window_coverage: Optional[float] = None
        self.window_near_pole_coverage: Optional[float] = None
        
        # Pole location tracking
        self.detected_pole_locations: List[float] = []
        self.pole_encounter_history: List[Tuple[int, float]] = []  # (step, |Q|)
    
    def update(self,
               tags: List[TRTag],
               q_values: Optional[List[float]] = None,
               x_values: Optional[List[float]] = None) -> None:
        """
        Update coverage statistics with a batch.
        
        Args:
            tags: List of output tags
            q_values: Optional list of |Q(x)| values
            x_values: Optional list of input x values
        """
        # Update current batch
        self.current_batch.reset()
        self.current_batch.update(tags, q_values, self.pole_threshold)
        
        # Update cumulative
        self.cumulative.update(tags, q_values, self.pole_threshold)
        
        # Track pole encounters
        if q_values and self.track_pole_distances:
            for i, q_val in enumerate(q_values):
                if abs(q_val) <= self.pole_threshold:
                    step = self.cumulative.total_samples - len(tags) + i
                    self.pole_encounter_history.append((step, abs(q_val)))
                    
                    # Track pole location if x_values provided
                    if x_values and i < len(x_values):
                        x_val = x_values[i]
                        if not any(abs(x_val - p) < 0.01 for p in self.detected_pole_locations):
                            self.detected_pole_locations.append(x_val)
        
        # Update window if applicable
        if self.window_size is not None:
            self._update_window()
    
    def _update_window(self) -> None:
        """Update sliding window statistics."""
        # Add current batch to history
        batch_copy = EnhancedCoverageMetrics(
            total_samples=self.current_batch.total_samples,
            real_samples=self.current_batch.real_samples,
            pinf_samples=self.current_batch.pinf_samples,
            ninf_samples=self.current_batch.ninf_samples,
            phi_samples=self.current_batch.phi_samples,
            near_pole_samples=self.current_batch.near_pole_samples,
            near_pole_real=self.current_batch.near_pole_real,
            near_pole_nonreal=self.current_batch.near_pole_nonreal,
            min_q_value=self.current_batch.min_q_value,
            mean_q_value=self.current_batch.mean_q_value,
        )
        self.history.append(batch_copy)
        
        # Compute window coverage
        total = sum(m.total_samples for m in self.history)
        real = sum(m.real_samples for m in self.history)
        self.window_coverage = real / total if total > 0 else 1.0
        
        # Compute window near-pole coverage
        near_total = sum(m.near_pole_samples for m in self.history)
        near_real = sum(m.near_pole_real for m in self.history)
        self.window_near_pole_coverage = near_real / near_total if near_total > 0 else 1.0
    
    @property
    def coverage(self) -> float:
        """Get current coverage (window or cumulative)."""
        if self.window_size is not None and self.window_coverage is not None:
            return self.window_coverage
        return self.cumulative.coverage
    
    @property
    def near_pole_coverage(self) -> float:
        """Get near-pole coverage (window or cumulative)."""
        if self.window_size is not None and self.window_near_pole_coverage is not None:
            return self.window_near_pole_coverage
        return self.cumulative.near_pole_coverage
    
    @property
    def batch_coverage(self) -> float:
        """Coverage of the most recent batch."""
        return self.current_batch.coverage
    
    @property
    def batch_near_pole_coverage(self) -> float:
        """Near-pole coverage of the most recent batch."""
        return self.current_batch.near_pole_coverage
    
    @property
    def min_distance_to_pole(self) -> Optional[float]:
        """Minimum |Q| value encountered (distance to nearest pole)."""
        return self.cumulative.min_q_value
    
    @property
    def mean_distance_to_pole(self) -> Optional[float]:
        """Mean |Q| value (average distance to poles)."""
        return self.cumulative.mean_q_value
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coverage statistics."""
        stats = {
            # Basic coverage
            "target_coverage": self.target_coverage,
            "current_coverage": self.coverage,
            "batch_coverage": self.batch_coverage,
            "cumulative_coverage": self.cumulative.coverage,
            
            # Near-pole coverage
            "near_pole_coverage": self.near_pole_coverage,
            "batch_near_pole_coverage": self.batch_near_pole_coverage,
            "near_pole_samples": self.cumulative.near_pole_samples,
            "near_pole_real": self.cumulative.near_pole_real,
            "near_pole_nonreal": self.cumulative.near_pole_nonreal,
            
            # Non-REAL tracking
            "actual_nonreal_rate": self.cumulative.actual_nonreal_rate,
            "nonreal_count": len(self.cumulative.actual_nonreal_outputs),
            
            # Distance metrics
            "min_q_value": self.min_distance_to_pole,
            "mean_q_value": self.mean_distance_to_pole,
            
            # Pole detection
            "detected_poles": len(self.detected_pole_locations),
            "pole_locations": self.detected_pole_locations.copy(),
            "pole_encounters": len(self.pole_encounter_history),
            
            # Tag distribution
            "tag_distribution": self.cumulative.tag_distribution,
            "total_samples": self.cumulative.total_samples,
        }
        
        # Add window stats if applicable
        if self.window_size is not None:
            stats["window_coverage"] = self.window_coverage
            stats["window_near_pole_coverage"] = self.window_near_pole_coverage
        
        return stats
    
    def get_nonreal_samples(self, last_n: Optional[int] = None) -> List[Tuple[int, TRTag]]:
        """
        Get actual non-REAL output samples.
        
        Args:
            last_n: Return only last N non-REAL samples (None for all)
            
        Returns:
            List of (sample_index, tag) tuples
        """
        outputs = self.cumulative.actual_nonreal_outputs
        if last_n is not None:
            return outputs[-last_n:]
        return outputs.copy()
    
    def reset(self) -> None:
        """Reset all tracking statistics."""
        self.current_batch.reset()
        self.cumulative.reset()
        self.history.clear()
        self.window_coverage = None
        self.window_near_pole_coverage = None
        self.detected_pole_locations = []
        self.pole_encounter_history = []


class CoverageEnforcementPolicy:
    """
    Policy for enforcing coverage targets with improved control.
    
    Features:
    - Asymmetric updates (faster increase, slower decrease)
    - Dead-band to prevent oscillation
    - Special handling for near-pole coverage
    """
    
    def __init__(self,
                 target_coverage: float = 0.90,
                 near_pole_target: float = 0.70,
                 dead_band: float = 0.02,
                 increase_rate: float = 2.0,
                 decrease_rate: float = 0.5,
                 min_lambda: float = 0.1,
                 max_lambda: float = 10.0):
        """
        Initialize coverage enforcement policy.
        
        Args:
            target_coverage: Target global coverage
            near_pole_target: Target coverage in near-pole regions
            dead_band: Tolerance band around target (±)
            increase_rate: Multiplier for lambda increases
            decrease_rate: Multiplier for lambda decreases
            min_lambda: Minimum lambda value
            max_lambda: Maximum lambda value
        """
        self.target_coverage = target_coverage
        self.near_pole_target = near_pole_target
        self.dead_band = dead_band
        self.increase_rate = increase_rate
        self.decrease_rate = decrease_rate
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        
        # State tracking
        self.lambda_history: List[float] = []
        self.coverage_history: List[float] = []
        self.intervention_count = 0
    
    def compute_lambda_update(self,
                             current_coverage: float,
                             current_lambda: float,
                             near_pole_coverage: Optional[float] = None) -> float:
        """
        Compute lambda update based on coverage.
        
        Args:
            current_coverage: Current global coverage
            current_lambda: Current lambda value
            near_pole_coverage: Optional near-pole coverage
            
        Returns:
            Updated lambda value
        """
        # Check if within dead-band
        coverage_gap = self.target_coverage - current_coverage
        if abs(coverage_gap) < self.dead_band:
            # Within acceptable range, no change
            return current_lambda
        
        # Base update magnitude
        base_update = abs(coverage_gap) * 0.1  # Base learning rate
        
        # Apply asymmetric rates with correct direction
        if coverage_gap < 0:  # Coverage too high, need to increase lambda
            # Make update positive to increase lambda
            effective_update = base_update * self.increase_rate
        else:  # Coverage too low, need to decrease lambda
            # Make update negative to decrease lambda
            effective_update = -base_update * self.decrease_rate
        
        # Special adjustment for poor near-pole coverage
        if near_pole_coverage is not None and near_pole_coverage < self.near_pole_target:
            # Boost lambda to encourage exploration
            pole_gap = self.near_pole_target - near_pole_coverage
            pole_adjustment = pole_gap * 0.05 * self.increase_rate
            effective_update += pole_adjustment
        
        # Apply update
        new_lambda = current_lambda + effective_update
        
        # Apply bounds
        new_lambda = max(self.min_lambda, min(self.max_lambda, new_lambda))
        
        # Track history
        self.lambda_history.append(new_lambda)
        self.coverage_history.append(current_coverage)
        
        return new_lambda
    
    def should_intervene(self, 
                        current_coverage: float,
                        steps_since_last: int = 0) -> bool:
        """
        Determine if intervention is needed.
        
        Args:
            current_coverage: Current coverage
            steps_since_last: Steps since last intervention
            
        Returns:
            True if intervention needed
        """
        # Critical intervention if coverage way off
        if current_coverage > 0.99:  # Almost no exploration
            return True
        if current_coverage < 0.5:  # Too much rejection
            return True
        
        # Regular intervention if outside dead-band and enough time passed
        if abs(self.target_coverage - current_coverage) > self.dead_band * 2:
            if steps_since_last >= 10:  # Wait at least 10 steps
                return True
        
        return False
    
    def enforce(self,
               current_coverage: float,
               current_lambda: float,
               near_pole_coverage: Optional[float] = None,
               Q_values: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Enforce coverage policy.
        
        Args:
            current_coverage: Current global coverage
            current_lambda: Current lambda value
            near_pole_coverage: Optional near-pole coverage
            Q_values: Optional list of Q values for analysis
            
        Returns:
            Dictionary with enforcement results
        """
        # Check if intervention needed
        should_update = abs(self.target_coverage - current_coverage) > self.dead_band
        
        if not should_update:
            return {
                "lambda_updated": False,
                "new_lambda": current_lambda,
                "coverage_gap": self.target_coverage - current_coverage,
                "intervention_triggered": False
            }
        
        # Compute new lambda
        new_lambda = self.compute_lambda_update(
            current_coverage, current_lambda, near_pole_coverage
        )
        
        # Check if this is an intervention
        intervention = abs(new_lambda - current_lambda) > current_lambda * 0.2
        if intervention:
            self.intervention_count += 1
        
        return {
            "lambda_updated": True,
            "new_lambda": new_lambda,
            "old_lambda": current_lambda,
            "coverage_gap": self.target_coverage - current_coverage,
            "near_pole_gap": (self.near_pole_target - near_pole_coverage) if near_pole_coverage else None,
            "intervention_triggered": intervention,
            "intervention_count": self.intervention_count
        }


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
            pole_threshold: |Q| threshold for near-pole classification
            oversample_ratio: How much to oversample near-pole regions
            adaptive: Whether to adapt ratio based on coverage
        """
        self.pole_threshold = pole_threshold
        self.oversample_ratio = oversample_ratio
        self.base_oversample_ratio = oversample_ratio
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
        Sample a batch with oversampling near poles.
        
        Args:
            data: List of data tuples
            batch_size: Size of batch to sample
            Q_values: Optional Q values for weighted sampling
            
        Returns:
            Sampled batch
        """
        if Q_values is None:
            # Random sampling if no Q values
            indices = np.random.choice(len(data), batch_size, replace=True)
        else:
            # Weighted sampling based on Q values
            weights = self.compute_sample_weights(Q_values)
            indices = np.random.choice(len(data), batch_size, p=weights, replace=True)
        
        return [data[i] for i in indices]


class AdaptiveGridSampler:
    """
    Adaptive grid refinement for sampling near detected poles.
    
    This sampler dynamically refines the sampling grid around
    detected singularities to ensure adequate exploration.
    """
    
    def __init__(self,
                 initial_grid_size: int = 100,
                 refinement_factor: int = 5,
                 pole_radius: float = 0.1):
        """
        Initialize adaptive grid sampler.
        
        Args:
            initial_grid_size: Initial number of grid points
            refinement_factor: How many points to add near poles
            pole_radius: Radius around poles for refinement
        """
        self.initial_grid_size = initial_grid_size
        self.refinement_factor = refinement_factor
        self.pole_radius = pole_radius
        
        # Grid state
        self.grid_points = []
        self.detected_poles = []
        self.refinement_history = []
    
    def initialize_grid(self, x_min: float, x_max: float) -> np.ndarray:
        """
        Initialize uniform grid.
        
        Args:
            x_min: Minimum x value
            x_max: Maximum x value
            
        Returns:
            Initial grid points
        """
        self.grid_points = np.linspace(x_min, x_max, self.initial_grid_size)
        return self.grid_points
    
    def refine_near_pole(self, pole_location: float) -> np.ndarray:
        """
        Refine grid near a detected pole.
        
        Args:
            pole_location: Location of detected pole
            
        Returns:
            New grid points added
        """
        # Add refined points around pole
        new_points = []
        for offset in np.linspace(-self.pole_radius, self.pole_radius, 
                                 self.refinement_factor):
            point = pole_location + offset
            if point not in self.grid_points:
                new_points.append(point)
        
        # Update grid
        if new_points:
            self.grid_points = np.sort(np.concatenate([self.grid_points, new_points]))
            self.refinement_history.append((pole_location, len(new_points)))
        
        return np.array(new_points)
    
    def update_poles(self, Q_values: List[float], 
                    x_values: List[float],
                    threshold: float = 0.1) -> List[float]:
        """
        Detect and refine around new poles.
        
        Args:
            Q_values: |Q(x)| values
            x_values: Corresponding x values
            threshold: Threshold for pole detection
            
        Returns:
            List of newly detected pole locations
        """
        new_poles = []
        
        for q_val, x_val in zip(Q_values, x_values):
            if abs(q_val) <= threshold:
                # Check if this is a new pole
                if not any(abs(x_val - p) < self.pole_radius/2 for p in self.detected_poles):
                    self.detected_poles.append(x_val)
                    new_poles.append(x_val)
                    # Refine grid around new pole
                    self.refine_near_pole(x_val)
        
        return new_poles
    
    def get_weighted_samples(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get weighted samples with emphasis near poles.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (sample_points, weights)
        """
        # Compute weights based on proximity to poles
        weights = np.ones(len(self.grid_points))
        
        for point in self.grid_points:
            for pole in self.detected_poles:
                dist = abs(point - pole)
                if dist <= self.pole_radius:
                    # Weight inversely proportional to distance
                    weight_boost = 1.0 / (dist + 0.01)
                    weights[self.grid_points == point] *= (1 + weight_boost)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Sample points
        indices = np.random.choice(len(self.grid_points), n_samples, p=weights)
        samples = self.grid_points[indices]
        
        return samples, weights[indices]