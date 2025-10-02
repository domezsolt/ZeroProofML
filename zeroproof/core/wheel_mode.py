"""
Wheel mode configuration for transreal arithmetic.

In Wheel mode, certain operations that produce PHI in standard TR mode
instead produce a bottom element (⊥), providing stricter algebraic control.

Key differences:
- 0 × ∞ = ⊥ (instead of PHI)
- ∞ + ∞ = ⊥ (instead of PHI)
- ∞ - ∞ = ⊥ (instead of PHI)
"""

from contextlib import contextmanager
from enum import Enum, auto


class ArithmeticMode(Enum):
    """
    Available arithmetic modes.

    Attributes:
        TRANSREAL: Standard transreal arithmetic with PHI for indeterminate forms
        WHEEL: Stricter algebra with bottom (⊥) for certain operations
    """

    TRANSREAL = auto()
    WHEEL = auto()

    def __str__(self) -> str:
        return self.name.lower()


class WheelModeConfig:
    """Global configuration for arithmetic mode (TR vs Wheel)."""

    _mode: ArithmeticMode = ArithmeticMode.TRANSREAL

    @classmethod
    def set_mode(cls, mode: ArithmeticMode) -> None:
        """
        Set the global arithmetic mode.

        Args:
            mode: The arithmetic mode to use

        Raises:
            ValueError: If mode is invalid
        """
        if not isinstance(mode, ArithmeticMode):
            raise ValueError(f"Invalid arithmetic mode: {mode}")
        cls._mode = mode

    @classmethod
    def get_mode(cls) -> ArithmeticMode:
        """Get the current arithmetic mode."""
        return cls._mode

    @classmethod
    def is_transreal(cls) -> bool:
        """Check if using standard transreal mode."""
        return cls._mode == ArithmeticMode.TRANSREAL

    @classmethod
    def is_wheel(cls) -> bool:
        """Check if using wheel mode."""
        return cls._mode == ArithmeticMode.WHEEL

    @classmethod
    def reset(cls) -> None:
        """Reset to default mode (transreal)."""
        cls._mode = ArithmeticMode.TRANSREAL


@contextmanager
def wheel_mode():
    """
    Context manager for temporary wheel mode.

    Example:
        with wheel_mode():
            # Operations use wheel algebra
            result = tr_mul(zero, infinity)  # Returns bottom
    """
    old_mode = WheelModeConfig.get_mode()

    try:
        WheelModeConfig.set_mode(ArithmeticMode.WHEEL)
        yield
    finally:
        WheelModeConfig.set_mode(old_mode)


@contextmanager
def arithmetic_mode(mode: ArithmeticMode):
    """
    Context manager for temporary arithmetic mode change.

    Args:
        mode: The arithmetic mode to use

    Example:
        with arithmetic_mode(ArithmeticMode.WHEEL):
            # Use wheel algebra
            pass
    """
    old_mode = WheelModeConfig.get_mode()

    try:
        WheelModeConfig.set_mode(mode)
        yield
    finally:
        WheelModeConfig.set_mode(old_mode)


# Convenience functions
def use_transreal() -> None:
    """Switch to standard transreal mode."""
    WheelModeConfig.set_mode(ArithmeticMode.TRANSREAL)


def use_wheel() -> None:
    """Switch to wheel mode."""
    WheelModeConfig.set_mode(ArithmeticMode.WHEEL)
