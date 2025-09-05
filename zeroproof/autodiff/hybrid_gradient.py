"""
Hybrid gradient schedule and context.

Provides a schedule for switching between Mask-REAL and Saturating gradients
near poles, along with a global context to coordinate per-epoch thresholds and
basic usage statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import cos, pi
from typing import Optional, Dict, Any

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
        p = self._progress(epoch)
        if self.schedule_type == ScheduleType.LINEAR:
            return self.delta_init + (self.delta_final - self.delta_init) * p
        if self.schedule_type == ScheduleType.COSINE:
            # Cosine anneal from init → final
            return self.delta_final + 0.5 * (self.delta_init - self.delta_final) * (1.0 + cos(pi * p))
        # EXPONENTIAL (default)
        if self.delta_init <= 0.0:
            return self.delta_final
        ratio = self.delta_final / self.delta_init
        return self.delta_init * (ratio ** p)

    def get_mode_description(self, epoch: int) -> str:
        if not self.enable:
            return "disabled"
        if self.is_warmup(epoch):
            return "warmup (MASK_REAL)"
        if self.is_transitioning(epoch):
            return f"transitioning (delta={self.get_delta(epoch):.3e})"
        return f"converged (delta={self.get_delta(epoch):.3e})"


class HybridGradientContext:
    """Global controller for hybrid gradient thresholds and stats."""

    _schedule: Optional[HybridGradientSchedule] = None
    _current_epoch: int = 0
    _local_threshold: Optional[float] = None

    _stats_total_calls: int = 0
    _stats_saturating: int = 0
    _stats_mask_real: int = 0

    @classmethod
    def set_schedule(cls, schedule: HybridGradientSchedule) -> None:
        cls._schedule = schedule

    @classmethod
    def get_schedule(cls) -> Optional[HybridGradientSchedule]:
        return cls._schedule

    @classmethod
    def update_epoch(cls, epoch: int) -> None:
        cls._current_epoch = epoch
        if cls._schedule is None or not cls._schedule.enable:
            cls._local_threshold = None
        else:
            cls._local_threshold = cls._schedule.get_delta(epoch)
        # Expose threshold to grad mode config for callers that consult it
        GradientModeConfig.set_local_threshold(cls._local_threshold)

    @classmethod
    def should_use_saturating(cls, abs_q_value: float) -> bool:
        cls._stats_total_calls += 1
        thr = cls._local_threshold
        if thr is not None and abs_q_value <= thr:
            cls._stats_saturating += 1
            return True
        cls._stats_mask_real += 1
        return False

    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        total = cls._stats_total_calls
        sat = cls._stats_saturating
        mask = cls._stats_mask_real
        ratio = (sat / total) if total > 0 else 0.0
        return {
            "current_epoch": cls._current_epoch,
            "local_threshold": cls._local_threshold,
            "total_gradient_calls": total,
            "saturating_activations": sat,
            "mask_real_activations": mask,
            "saturating_ratio": ratio,
        }

    @classmethod
    def reset_statistics(cls) -> None:
        cls._stats_total_calls = 0
        cls._stats_saturating = 0
        cls._stats_mask_real = 0

    @classmethod
    def reset(cls) -> None:
        cls._schedule = None
        cls._current_epoch = 0
        cls._local_threshold = None
        cls.reset_statistics()
        GradientModeConfig.reset()


def create_default_schedule(aggressive: bool = False, warmup_epochs: int = 0) -> HybridGradientSchedule:
    if aggressive:
        return HybridGradientSchedule(
            warmup_epochs=warmup_epochs,
            transition_epochs=20,
            delta_init=1e-1,
            delta_final=1e-8,
            schedule_type=ScheduleType.EXPONENTIAL,
            enable=True,
            saturating_bound=0.1,
        )
    return HybridGradientSchedule(
        warmup_epochs=warmup_epochs,
        transition_epochs=20,
        delta_init=1e-2,
        delta_final=1e-6,
        schedule_type=ScheduleType.EXPONENTIAL,
        enable=True,
        saturating_bound=1.0,
    )


