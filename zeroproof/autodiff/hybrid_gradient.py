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
from typing import Any, Dict, List, Optional, Set, Tuple

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
        return (
            (not self.is_warmup(epoch))
            and (self.transition_epochs > 0)
            and (epoch < self.warmup_epochs + self.transition_epochs)
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
            base_delta = self.delta_final + 0.5 * (self.delta_init - self.delta_final) * (
                1.0 + cos(pi * p)
            )
        else:
            # EXPONENTIAL (default)
            if self.delta_init <= 0.0:
                base_delta = self.delta_final
            else:
                ratio = self.delta_final / self.delta_init
                base_delta = self.delta_init * (ratio**p)

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
            # Ensure gradient mode reflects hybrid scheduling and bound
            from .grad_mode import GradientMode, GradientModeConfig

            GradientModeConfig.set_mode(GradientMode.HYBRID)
            GradientModeConfig.set_saturation_bound(self.saturating_bound)
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
                for e in range(
                    epoch + 1,
                    min(
                        epoch + 1 + self.pole_exploration_epochs,
                        self.warmup_epochs + self.transition_epochs,
                    ),
                ):
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
    # Flip tracking across batches in an epoch
    _policy_flip_count_epoch: int = 0
    _batch_count_epoch: int = 0

    # Policy-driven hysteresis state
    _policy_enabled: bool = False
    _policy_mode_sat: bool = False  # False: MR, True: SAT
    _tau_q_on: Optional[float] = None
    _tau_q_off: Optional[float] = None
    _g_on: Optional[float] = None
    _g_off: Optional[float] = None

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

        # Configure policy-based thresholds if a policy is active
        try:
            from ..policy import TRPolicyConfig

            pol = TRPolicyConfig.get_policy()
            if pol is not None:
                cls._policy_enabled = True
                cls._tau_q_on = float(pol.tau_Q_on)
                cls._tau_q_off = float(pol.tau_Q_off)
                cls._g_on = float(pol.g_on) if pol.g_on is not None else None
                cls._g_off = float(pol.g_off) if pol.g_off is not None else None
            else:
                cls._policy_enabled = False
                cls._tau_q_on = cls._tau_q_off = None
                cls._g_on = cls._g_off = None
        except Exception:
            cls._policy_enabled = False
            cls._tau_q_on = cls._tau_q_off = None
            cls._g_on = cls._g_off = None

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

        # If policy hysteresis says SAT globally, honor it
        if cls._policy_enabled and cls._policy_mode_sat:
            cls._stats_saturating += 1
            cls._near_pole_samples.add(cls._stats_total_calls)
            return True

        # Standard threshold check (schedule or policy ON threshold)
        thr = cls._local_threshold
        # If policy exists, prefer its ON threshold when schedule threshold is absent
        if thr is None and cls._policy_enabled and cls._tau_q_on is not None:
            thr = cls._tau_q_on

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
            q_arr = np.array(cls._q_values_batch, dtype=float)
            # Robust batch stats and quantiles available regardless of policy
            try:
                p10 = float(np.percentile(q_arr, 10))
                p50 = float(np.percentile(q_arr, 50))
                p90 = float(np.percentile(q_arr, 90))
            except Exception:
                p10 = float(np.min(q_arr))
                p50 = float(np.median(q_arr))
                p90 = float(np.max(q_arr))
            # Derive sensitivity proxy g = 1/|Q|
            try:
                g_arr = 1.0 / np.maximum(q_arr, 1e-18)
                g_p10 = float(np.percentile(g_arr, 10))
                g_p50 = float(np.percentile(g_arr, 50))
                g_p90 = float(np.percentile(g_arr, 90))
            except Exception:
                # Fallbacks
                g_p10 = float(1.0 / max(p90, 1e-18))
                g_p50 = float(1.0 / max(p50, 1e-18))
                g_p90 = float(1.0 / max(p10, 1e-18))

            q_stats = {
                "q_min_batch": cls._q_min_batch,
                "q_mean_batch": float(np.mean(q_arr)),
                "q_median_batch": float(np.median(q_arr)),
                "q_p10": p10,
                "q_p50": p50,
                "q_p90": p90,
                "g_p10": g_p10,
                "g_p50": g_p50,
                "g_p90": g_p90,
                "near_pole_ratio": len(cls._near_pole_samples) / len(cls._q_values_batch),
            }

        stats = {
            "current_epoch": cls._current_epoch,
            "local_threshold": cls._local_threshold,
            "total_gradient_calls": total,
            "saturating_activations": sat,
            "mask_real_activations": mask,
            "saturating_ratio": ratio,
            "q_min_epoch": cls._q_min_epoch,
            "exploration_regions": len(cls._exploration_regions),
            **q_stats,
        }

        # Add policy/hysteresis summary and thresholds if available
        if cls._policy_enabled:
            try:
                stats.update(
                    {
                        "policy_mode": "SAT" if cls._policy_mode_sat else "MR",
                        "tau_q_on": cls._tau_q_on,
                        "tau_q_off": cls._tau_q_off,
                    }
                )
            except Exception:
                stats.update(
                    {
                        "policy_mode": "SAT" if cls._policy_mode_sat else "MR",
                        "tau_q_on": cls._tau_q_on,
                        "tau_q_off": cls._tau_q_off,
                    }
                )

        # Flip stats across batches in this epoch (if tracked)
        if hasattr(cls, "_policy_flip_count_epoch") and hasattr(cls, "_batch_count_epoch"):
            flips = getattr(cls, "_policy_flip_count_epoch", 0) or 0
            batches = getattr(cls, "_batch_count_epoch", 0) or 0
            stats["policy_flip_count"] = flips
            stats["flip_rate"] = (float(flips) / float(batches)) if batches > 0 else 0.0

        return stats

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
        cls._policy_flip_count_epoch = 0
        cls._batch_count_epoch = 0

    @classmethod
    def end_batch_policy_update(cls) -> None:
        """Update hybrid mode using policy hysteresis based on batch quantiles.

        - Enter SAT if q_p10 <= tau_Q_on or g90 >= g_on
        - Return to MR if q_p10 >= tau_Q_off and (g90 <= g_off if provided)
        Then reset per-batch statistics.
        """
        if not cls._policy_enabled or not cls._q_values_batch:
            cls.reset_statistics()
            return

        q = np.array(cls._q_values_batch, dtype=float)
        try:
            q_p10 = float(np.percentile(q, 10))
        except Exception:
            q_p10 = float(np.min(q))

        # Sensitivity proxy g ≈ 1/|Q|; use 90th percentile for robustness
        with np.errstate(divide="ignore", invalid="ignore"):
            g_vals = 1.0 / np.maximum(q, 1e-18)
        try:
            g90 = float(np.percentile(g_vals, 90))
        except Exception:
            g90 = float(np.max(g_vals))

        # Hysteresis decisions
        enter_sat = False
        exit_mr = False
        if cls._tau_q_on is not None and q_p10 <= cls._tau_q_on:
            enter_sat = True
        if cls._g_on is not None and g90 >= cls._g_on:
            enter_sat = True
        if cls._tau_q_off is not None and q_p10 >= cls._tau_q_off:
            exit_mr = True
        if cls._g_off is not None and g90 > cls._g_off:
            exit_mr = False

        prev_mode = cls._policy_mode_sat
        if not cls._policy_mode_sat and enter_sat:
            cls._policy_mode_sat = True
        elif cls._policy_mode_sat and exit_mr:
            cls._policy_mode_sat = False
        # Update flip counters
        if cls._policy_mode_sat != prev_mode:
            cls._policy_flip_count_epoch += 1
        cls._batch_count_epoch += 1

        # Reset per-batch stats after update
        cls.reset_statistics()

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
        # Also record for quantiles even if saturating decision path is not hit
        try:
            cls._q_values_batch.append(float(abs_q))
        except Exception:
            pass

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


def create_default_schedule(
    aggressive: bool = False, warmup_epochs: int = 0, force_exploration: bool = True
) -> HybridGradientSchedule:
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
            min_delta=1e-10,
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
        min_delta=1e-8,
    )
