"""
Hybrid gradient schedule and context.

Provides a schedule for switching between Mask-REAL and Saturating gradients
near poles, along with a global context to coordinate per-epoch thresholds and
basic usage statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from math import cos, pi
from typing import Optional, Dict, Any, List, Set, Tuple
import numpy as np

from .grad_mode import GradientModeConfig


class ScheduleType(Enum):
    LINEAR = auto()
    EXPONENTIAL = auto()
    COSINE = auto()


@dataclass
class HybridGradientSchedule:
    warmup_epochs: int = 0
    transition_epochs: int = 20
    delta_init: float = 1e-2
    delta_final: float = 1e-6
    schedule_type: ScheduleType = ScheduleType.EXPONENTIAL
    enable: bool = True
    saturating_bound: float = 1.0
    
    # New: Force exploration parameters
    force_pole_exploration: bool = True
    pole_exploration_radius: float = 0.05  # δ-neighborhood radius
    pole_exploration_epochs: int = 5  # Epochs to explore each detected pole
    pole_detection_threshold: float = 0.1  # Threshold to consider as pole
    adaptive_delta: bool = True  # Adapt delta based on q_min
    min_delta: float = 1e-8  # Minimum delta value
    
    # New: Detected poles tracking
    detected_poles: List[float] = field(default_factory=list)
    pole_exploration_schedule: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)

    def is_warmup(self, epoch: int) -> bool:
        return self.enable and epoch < max(0, self.warmup_epochs)

    def is_transitioning(self, epoch: int) -> bool:
        if not self.enable:
            return False
        return (not self.is_warmup(epoch)) and (self.transition_epochs > 0) and (
            epoch < self.warmup_epochs + self.transition_epochs
        )

    def _progress(self, epoch: int) -> float:
        if self.transition_epochs <= 0:
            return 1.0
        p = (epoch - self.warmup_epochs) / float(self.transition_epochs)
        if p < 0.0:
            p = 0.0
        if p > 1.0:
            p = 1.0
        return p

    def get_delta(self, epoch: int) -> Optional[float]:
        if not self.enable:
            return None
        if self.is_warmup(epoch):
            return None
        
        # Base delta from schedule
        p = self._progress(epoch)
        if self.schedule_type == ScheduleType.LINEAR:
            base_delta = self.delta_init + (self.delta_final - self.delta_init) * p
        elif self.schedule_type == ScheduleType.COSINE:
            # Cosine anneal from init → final
            base_delta = self.delta_final + 0.5 * (self.delta_init - self.delta_final) * (1.0 + cos(pi * p))
        else:
            # EXPONENTIAL (default)
            if self.delta_init <= 0.0:
                base_delta = self.delta_final
            else:
                ratio = self.delta_final / self.delta_init
                base_delta = self.delta_init * (ratio ** p)
        
        # Adapt based on q_min if enabled
        if self.adaptive_delta:
            q_min = HybridGradientContext.get_q_min()
            if q_min is not None and q_min > 0:
                # Increase delta when q_min is small (near poles)
                adapted_delta = base_delta * max(1.0, 0.1 / q_min)
                base_delta = min(adapted_delta, self.delta_init * 2.0)  # Cap at 2x initial
        
        # Apply minimum threshold
        return max(base_delta, self.min_delta)

    def get_mode_description(self, epoch: int) -> str:
        if not self.enable:
            return "disabled"
        if self.is_warmup(epoch):
            return f"warmup (mask-real only, {epoch}/{self.warmup_epochs})"
        if self.is_transitioning(epoch):
            delta = self.get_delta(epoch)
            pole_info = ""
            if self.force_pole_exploration and epoch in self.pole_exploration_schedule:
                n_poles = len(self.pole_exploration_schedule[epoch])
                pole_info = f", exploring {n_poles} poles"
            return f"transitioning (delta={delta:.3e}{pole_info})"
        
        delta = self.get_delta(epoch)
        return f"converged (delta={delta:.3e})"
    
    # Lightweight context manager API used by tests
    def apply(self, epoch: int):
        """Context manager to apply this schedule for a given epoch.

        Usage:
            with schedule.apply(epoch=5):
                # do backward passes
        """
        from contextlib import contextmanager
        @contextmanager
        def _ctx():
            # Register schedule and epoch
            HybridGradientContext.set_schedule(self)
            HybridGradientContext.update_epoch(epoch)
            try:
                yield
            finally:
                # No-op on exit; keep stats for inspection
                pass
        return _ctx()
    
    def update_detected_poles(self, new_poles: List[float], epoch: int) -> None:
        """Update detected poles and schedule exploration."""
        if not self.force_pole_exploration:
            return
        
        # Add new unique poles
        for pole in new_poles:
            if not any(abs(pole - p) < self.pole_exploration_radius for p in self.detected_poles):
                self.detected_poles.append(pole)
                
                # Schedule exploration for next few epochs
                for e in range(epoch + 1, min(epoch + 1 + self.pole_exploration_epochs, 
                                             self.warmup_epochs + self.transition_epochs)):
                    if e not in self.pole_exploration_schedule:
                        self.pole_exploration_schedule[e] = []
                    # Add pole neighborhood to explore
                    self.pole_exploration_schedule[e].append(
                        (pole - self.pole_exploration_radius, pole + self.pole_exploration_radius)
                    )
    
    def get_exploration_regions(self, epoch: int) -> List[Tuple[float, float]]:
        """Get pole neighborhoods to explore in this epoch."""
        return self.pole_exploration_schedule.get(epoch, [])


class HybridGradientContext:
    """Global controller for hybrid gradient thresholds and stats."""

    _schedule: Optional[HybridGradientSchedule] = None
    _current_epoch: int = 0
    _local_threshold: Optional[float] = None

    _stats_total_calls: int = 0
    _stats_saturating: int = 0
    _stats_mask_real: int = 0
    
    # New: q_min tracking
    _q_min_batch: Optional[float] = None
    _q_min_epoch: Optional[float] = None
    _q_values_batch: List[float] = []
    _near_pole_samples: Set[int] = set()
    _exploration_regions: List[Tuple[float, float]] = []

    @classmethod
    def set_schedule(cls, schedule: HybridGradientSchedule) -> None:
        cls._schedule = schedule

    @classmethod
    def get_schedule(cls) -> Optional[HybridGradientSchedule]:
        return cls._schedule

    @classmethod
    def update_epoch(cls, epoch: int) -> None:
        cls._current_epoch = epoch
        cls.reset_epoch_statistics()
        
        if cls._schedule is None or not cls._schedule.enable:
            cls._local_threshold = None
        else:
            cls._local_threshold = cls._schedule.get_delta(epoch)
            # Set exploration regions for this epoch
            cls._exploration_regions = cls._schedule.get_exploration_regions(epoch)
            
        # Expose threshold to grad mode config for callers that consult it
        GradientModeConfig.set_local_threshold(cls._local_threshold)

    @classmethod
    def should_use_saturating(cls, abs_q_value: float, x_value: Optional[float] = None) -> bool:
        """Determine if saturating gradient should be used.
        
        Args:
            abs_q_value: Absolute value of Q(x)
            x_value: Optional input value for pole exploration check
        """
        cls._stats_total_calls += 1
        
        # Track Q values
        cls._q_values_batch.append(abs_q_value)
        if cls._q_min_batch is None or abs_q_value < cls._q_min_batch:
            cls._q_min_batch = abs_q_value
        
        # Check if in forced exploration region
        if x_value is not None and cls._exploration_regions:
            for region_min, region_max in cls._exploration_regions:
                if region_min <= x_value <= region_max:
                    cls._stats_saturating += 1
                    cls._near_pole_samples.add(cls._stats_total_calls)
                    return True
        
        # Standard threshold check
        thr = cls._local_threshold
        if thr is not None and abs_q_value <= thr:
            cls._stats_saturating += 1
            cls._near_pole_samples.add(cls._stats_total_calls)
            return True
        
        cls._stats_mask_real += 1
        return False

    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        total = cls._stats_total_calls
        sat = cls._stats_saturating
        mask = cls._stats_mask_real
        ratio = (sat / total) if total > 0 else 0.0
        
        # Compute q statistics
        q_stats = {}
        if cls._q_values_batch:
            q_stats = {
                "q_min_batch": cls._q_min_batch,
                "q_mean_batch": np.mean(cls._q_values_batch),
                "q_median_batch": np.median(cls._q_values_batch),
                "near_pole_ratio": len(cls._near_pole_samples) / len(cls._q_values_batch)
            }
        
        return {
            "current_epoch": cls._current_epoch,
            "local_threshold": cls._local_threshold,
            "total_gradient_calls": total,
            "saturating_activations": sat,
            "mask_real_activations": mask,
            "saturating_ratio": ratio,
            "q_min_epoch": cls._q_min_epoch,
            "exploration_regions": len(cls._exploration_regions),
            **q_stats
        }

    @classmethod
    def reset_statistics(cls) -> None:
        """Reset per-batch statistics."""
        cls._stats_total_calls = 0
        cls._stats_saturating = 0
        cls._stats_mask_real = 0
        cls._q_values_batch = []
        cls._near_pole_samples = set()
        
        # Update epoch minimum
        if cls._q_min_batch is not None:
            if cls._q_min_epoch is None or cls._q_min_batch < cls._q_min_epoch:
                cls._q_min_epoch = cls._q_min_batch
        cls._q_min_batch = None
    
    @classmethod
    def reset_epoch_statistics(cls) -> None:
        """Reset per-epoch statistics."""
        cls._q_min_epoch = None
        cls._exploration_regions = []
    
    @classmethod
    def get_q_min(cls) -> Optional[float]:
        """Get current batch q_min."""
        return cls._q_min_batch
    
    @classmethod
    def get_q_min_epoch(cls) -> Optional[float]:
        """Get epoch q_min."""
        return cls._q_min_epoch
    
    @classmethod
    def update_q_value(cls, abs_q: float) -> None:
        """Update q_min tracking."""
        if cls._q_min_batch is None or abs_q < cls._q_min_batch:
            cls._q_min_batch = abs_q
    
    @classmethod
    def set_exploration_regions(cls, regions: List[Tuple[float, float]]) -> None:
        """Set pole exploration regions for current epoch."""
        cls._exploration_regions = regions
    
    @classmethod
    def detect_poles(cls, threshold: Optional[float] = None) -> List[int]:
        """Detect samples that are likely near poles.
        
        Args:
            threshold: Q-value threshold for pole detection
            
        Returns:
            List of sample indices that are near poles
        """
        if not cls._q_values_batch:
            return []
        
        threshold = threshold or (cls._local_threshold if cls._local_threshold else 0.1)
        near_poles = []
        
        for i, q_val in enumerate(cls._q_values_batch):
            if q_val <= threshold:
                near_poles.append(i)
        
        return near_poles

    @classmethod
    def reset(cls) -> None:
        cls._schedule = None
        cls._current_epoch = 0
        cls._local_threshold = None
        cls.reset_statistics()
        cls.reset_epoch_statistics()
        GradientModeConfig.reset()


def create_default_schedule(aggressive: bool = False, 
                           warmup_epochs: int = 0,
                           force_exploration: bool = True) -> HybridGradientSchedule:
    """Create a default hybrid gradient schedule.
    
    Args:
        aggressive: If True, use more aggressive parameters
        warmup_epochs: Number of warmup epochs with Mask-REAL only
        force_exploration: If True, enable forced pole exploration
        
    Returns:
        HybridGradientSchedule with appropriate parameters
    """
    if aggressive:
        return HybridGradientSchedule(
            warmup_epochs=warmup_epochs,
            transition_epochs=20,
            delta_init=1e-1,
            delta_final=1e-8,
            schedule_type=ScheduleType.EXPONENTIAL,
            enable=True,
            saturating_bound=0.1,
            force_pole_exploration=force_exploration,
            pole_exploration_radius=0.1,
            pole_exploration_epochs=10,
            adaptive_delta=True,
            min_delta=1e-10
        )
    return HybridGradientSchedule(
        warmup_epochs=warmup_epochs,
        transition_epochs=20,
        delta_init=1e-2,
        delta_final=1e-6,
        schedule_type=ScheduleType.EXPONENTIAL,
        enable=True,
        saturating_bound=1.0,
        force_pole_exploration=force_exploration,
        pole_exploration_radius=0.05,
        pole_exploration_epochs=5,
        adaptive_delta=True,
        min_delta=1e-8
    )