"""
Gradient computation modes for transreal autodifferentiation.

This module defines the available gradient modes (Mask-REAL and Saturating)
and provides utilities for managing the current mode.
"""

from contextlib import contextmanager
from enum import Enum, auto
from typing import Optional


class GradientMode(Enum):
    """
    Available gradient computation modes.

    Attributes:
        MASK_REAL: Default mode where non-REAL forward values produce zero gradients
        SATURATING: Alternative mode with bounded gradients near singularities
        HYBRID: Adaptive mode that switches based on local pole detection
    """

    MASK_REAL = auto()
    SATURATING = auto()
    HYBRID = auto()  # New mode for hybrid schedule

    def __str__(self) -> str:
        return self.name.lower().replace("_", "-")


class GradientModeConfig:
    """Global configuration for gradient computation mode."""

    _mode: GradientMode = GradientMode.MASK_REAL
    _saturation_bound: float = 1.0  # Default bound for saturating mode
    _local_threshold: Optional[float] = None  # For hybrid mode pole detection

    @classmethod
    def set_mode(cls, mode: GradientMode) -> None:
        """
        Set the global gradient computation mode.

        Args:
            mode: The gradient mode to use
        """
        if not isinstance(mode, GradientMode):
            raise ValueError(f"Invalid gradient mode: {mode}")
        cls._mode = mode

    @classmethod
    def get_mode(cls) -> GradientMode:
        """Get the current gradient computation mode."""
        return cls._mode

    @classmethod
    def set_saturation_bound(cls, bound: float) -> None:
        """
        Set the saturation bound for SATURATING mode.

        Args:
            bound: The bound value (must be positive)
        """
        if bound <= 0:
            raise ValueError(f"Saturation bound must be positive, got {bound}")
        cls._saturation_bound = bound

    @classmethod
    def get_saturation_bound(cls) -> float:
        """Get the current saturation bound."""
        return cls._saturation_bound

    @classmethod
    def is_mask_real(cls) -> bool:
        """Check if using MASK_REAL mode."""
        return cls._mode == GradientMode.MASK_REAL

    @classmethod
    def is_saturating(cls) -> bool:
        """Check if using SATURATING mode."""
        return cls._mode == GradientMode.SATURATING

    @classmethod
    def is_hybrid(cls) -> bool:
        """Check if using HYBRID mode."""
        return cls._mode == GradientMode.HYBRID

    @classmethod
    def set_local_threshold(cls, threshold: Optional[float]) -> None:
        """
        Set the local threshold for hybrid mode pole detection.

        Args:
            threshold: Threshold for |Q| to trigger saturating mode
        """
        cls._local_threshold = threshold

    @classmethod
    def get_local_threshold(cls) -> Optional[float]:
        """Get the current local threshold."""
        return cls._local_threshold

    @classmethod
    def reset(cls) -> None:
        """Reset to default configuration."""
        cls._mode = GradientMode.MASK_REAL
        cls._saturation_bound = 1.0
        cls._local_threshold = None


@contextmanager
def gradient_mode(mode: GradientMode, saturation_bound: Optional[float] = None):
    """
    Context manager for temporary gradient mode changes.

    Args:
        mode: The gradient mode to use
        saturation_bound: Optional saturation bound for SATURATING mode

    Example:
        with gradient_mode(GradientMode.SATURATING, saturation_bound=10.0):
            # Compute gradients with saturation
            loss.backward()
    """
    old_mode = GradientModeConfig.get_mode()
    old_bound = GradientModeConfig.get_saturation_bound()

    try:
        GradientModeConfig.set_mode(mode)
        if saturation_bound is not None and mode == GradientMode.SATURATING:
            GradientModeConfig.set_saturation_bound(saturation_bound)
        yield
    finally:
        GradientModeConfig.set_mode(old_mode)
        GradientModeConfig.set_saturation_bound(old_bound)


# Convenience functions
def use_mask_real() -> None:
    """Switch to MASK_REAL gradient mode."""
    GradientModeConfig.set_mode(GradientMode.MASK_REAL)


def use_saturating(bound: float = 1.0) -> None:
    """
    Switch to SATURATING gradient mode.

    Args:
        bound: Saturation bound (default: 1.0)
    """
    GradientModeConfig.set_mode(GradientMode.SATURATING)
    GradientModeConfig.set_saturation_bound(bound)
