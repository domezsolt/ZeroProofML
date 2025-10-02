"""
Global precision configuration for ZeroProof.

This module manages the default numeric precision used throughout the library.
By default, float64 is used for maximum precision in transreal arithmetic.

NumPy is optional for import-time; when NumPy is unavailable we provide a
minimal float64-only fallback so the top-level package can import with
minimal dependencies. Functions that require non-float64 precisions will
raise informative errors when NumPy is not installed.
"""

import sys
from enum import Enum
from typing import Union

try:  # Optional dependency for dtype management
    import numpy as np  # type: ignore

    NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    np = None  # type: ignore
    NUMPY_AVAILABLE = False


class PrecisionMode(Enum):
    """Supported precision modes.

    Uses string labels to avoid importing NumPy at module import time.
    """

    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    @property
    def numpy_dtype(self):
        """Get the NumPy dtype for this precision (or a fallback for float64)."""
        if NUMPY_AVAILABLE:
            # mypy: getattr is fine here for known dtype names
            return getattr(np, self.value)  # type: ignore[attr-defined]
        # Fallback: only float64 is supported without NumPy
        if self is PrecisionMode.FLOAT64:
            return float
        raise ImportError(
            "NumPy is required for precision modes other than float64. "
            "Install with: pip install numpy"
        )

    @property
    def bits(self) -> int:
        """Get the number of bits for this precision."""
        if NUMPY_AVAILABLE:
            return int(np.dtype(self.numpy_dtype).itemsize * 8)  # type: ignore[arg-type]
        return {PrecisionMode.FLOAT16: 16, PrecisionMode.FLOAT32: 32, PrecisionMode.FLOAT64: 64}[
            self
        ]


class PrecisionConfig:
    """
    Global precision configuration.

    By default, ZeroProof uses float64 for all computations to ensure
    maximum precision in transreal arithmetic operations.
    """

    _default_mode: PrecisionMode = PrecisionMode.FLOAT64
    _enforce_precision: bool = True

    @classmethod
    def set_precision(cls, mode: Union[PrecisionMode, str]) -> None:
        """
        Set the default precision mode.

        Args:
            mode: PrecisionMode enum or string ('float16', 'float32', 'float64')

        Raises:
            ValueError: If mode is not supported
        """
        if isinstance(mode, str):
            mode_map = {
                "float16": PrecisionMode.FLOAT16,
                "float32": PrecisionMode.FLOAT32,
                "float64": PrecisionMode.FLOAT64,
            }
            if mode not in mode_map:
                raise ValueError(f"Unsupported precision mode: {mode}")
            mode = mode_map[mode]

        if not isinstance(mode, PrecisionMode):
            raise ValueError(f"Invalid precision mode: {mode}")

        cls._default_mode = mode

    @classmethod
    def get_precision(cls) -> PrecisionMode:
        """Get the current default precision mode."""
        return cls._default_mode

    @classmethod
    def get_dtype(cls):  # -> Type[np.floating] | Type[float]
        """Get the dtype for the current precision.

        Returns a NumPy dtype when NumPy is available; otherwise returns
        Python's ``float`` for float64.
        """
        return cls._default_mode.numpy_dtype

    @classmethod
    def enforce_precision(cls, value: Union[float, int]) -> float:
        """
        Convert a value to the current default precision.

        Args:
            value: Numeric value to convert

        Returns:
            Value with enforced precision as Python float
        """
        if not cls._enforce_precision:
            return float(value)

        if NUMPY_AVAILABLE:
            dtype = cls.get_dtype()
            # Convert to numpy dtype then back to Python float
            return float(dtype(value))  # type: ignore[operator]
        # Fallback (float64 only)
        if cls._default_mode is not PrecisionMode.FLOAT64:
            raise ImportError(
                "NumPy is required for precision conversion to non-float64 modes. "
                "Install with: pip install numpy"
            )
        return float(value)

    @classmethod
    def set_enforcement(cls, enforce: bool) -> None:
        """
        Enable or disable precision enforcement.

        Args:
            enforce: Whether to enforce precision conversion
        """
        cls._enforce_precision = enforce

    @classmethod
    def is_enforcing(cls) -> bool:
        """Check if precision enforcement is enabled."""
        return cls._enforce_precision

    # Conservative overflow margin to avoid borderline magnitudes near dtype max
    _overflow_safety_margin: float = 1.0  # strict check; do not preemptively overflow

    @classmethod
    def check_overflow(cls, value: float) -> bool:
        """
        Check if a value would overflow in the current precision.

        Args:
            value: Value to check

        Returns:
            True if value would overflow
        """
        if NUMPY_AVAILABLE:
            dtype = cls.get_dtype()
            finfo = np.finfo(dtype)  # type: ignore[arg-type]
            # Be conservative near the limit to ensure deterministic overflow
            safety = cls._overflow_safety_margin
            try:
                threshold = float(finfo.max) * float(safety)
            except Exception:
                threshold = float(finfo.max)
            return abs(value) > threshold
        # Fallback: use Python float info for float64
        if cls._default_mode is not PrecisionMode.FLOAT64:
            raise ImportError(
                "NumPy is required for overflow checks on non-float64 modes. "
                "Install with: pip install numpy"
            )
        # sys.float_info.max corresponds to double precision
        return abs(value) > sys.float_info.max * cls._overflow_safety_margin

    @classmethod
    def get_epsilon(cls) -> float:
        """Get machine epsilon for current precision."""
        if NUMPY_AVAILABLE:
            return float(np.finfo(cls.get_dtype()).eps)  # type: ignore[arg-type]
        return sys.float_info.epsilon

    @classmethod
    def get_max(cls) -> float:
        """Get maximum representable value for current precision."""
        if NUMPY_AVAILABLE:
            return float(np.finfo(cls.get_dtype()).max)  # type: ignore[arg-type]
        return sys.float_info.max

    @classmethod
    def get_min(cls) -> float:
        """Get minimum positive value for current precision."""
        if NUMPY_AVAILABLE:
            return float(np.finfo(cls.get_dtype()).tiny)  # type: ignore[arg-type]
        # Python float has no distinct tiny; approximate with min positive
        return sys.float_info.min


# Context manager for temporary precision changes
class precision_context:
    """
    Context manager for temporary precision changes.

    Example:
        with precision_context('float32'):
            # Operations use float32
            x = real(1.0)
        # Back to previous precision
    """

    def __init__(self, mode: Union[PrecisionMode, str]):
        self.new_mode = mode
        self.old_mode = None
        self.old_enforcement = None

    def __enter__(self):
        self.old_mode = PrecisionConfig.get_precision()
        self.old_enforcement = PrecisionConfig.is_enforcing()
        PrecisionConfig.set_precision(self.new_mode)
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        PrecisionConfig.set_precision(self.old_mode)
        PrecisionConfig.set_enforcement(self.old_enforcement)
