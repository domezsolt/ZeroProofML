"""Wheel algebra mode convenience exports.

This module provides a thin compatibility layer that re-exports the
wheel-mode controls from ``zeroproof.core.wheel_mode`` under common names.

Exports:
- ``wheel_mode``: Context manager for temporary wheel mode
- ``enable_wheel_mode``: Switch global mode to wheel
- ``disable_wheel_mode``: Switch global mode back to transreal
- ``is_wheel_mode``: Check if wheel mode is active
- ``Bottom``: Constructor alias for creating a bottom (⊥) value
"""

from ..core import bottom  # Bottom value constructor
from ..core.wheel_mode import WheelModeConfig, use_transreal, use_wheel, wheel_mode


def enable_wheel_mode() -> None:
    """Enable wheel mode globally."""
    use_wheel()


def disable_wheel_mode() -> None:
    """Disable wheel mode and switch back to transreal globally."""
    use_transreal()


def is_wheel_mode() -> bool:
    """Return True if wheel mode is currently active."""
    return WheelModeConfig.is_wheel()


def Bottom():
    """Create a bottom (⊥) value. Alias for ``zeroproof.core.bottom()``."""
    return bottom()


__all__ = [
    "wheel_mode",
    "enable_wheel_mode",
    "disable_wheel_mode",
    "is_wheel_mode",
    "Bottom",
]
