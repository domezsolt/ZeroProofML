"""
Precision utilities for transreal arithmetic.

This module provides support for different floating-point precisions
(float16, float32, float64) and precision-aware conversions.
"""

import struct
import warnings
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from ..core import TRScalar, TRTag, ninf, phi, pinf, real


class Precision(Enum):
    """Supported floating-point precisions."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"  # Brain float (Google/Intel)


# Precision specifications
PRECISION_INFO = {
    Precision.FLOAT16: {
        "bits": 16,
        "mantissa_bits": 10,
        "exponent_bits": 5,
        "max_value": 65504.0,
        "min_normal": 6.104e-5,
        "epsilon": 9.77e-4,
    },
    Precision.FLOAT32: {
        "bits": 32,
        "mantissa_bits": 23,
        "exponent_bits": 8,
        "max_value": 3.4028235e38,
        "min_normal": 1.175494e-38,
        "epsilon": 1.192093e-7,
    },
    Precision.FLOAT64: {
        "bits": 64,
        "mantissa_bits": 52,
        "exponent_bits": 11,
        "max_value": 1.7976931348623157e308,
        "min_normal": 2.2250738585072014e-308,
        "epsilon": 2.220446049250313e-16,
    },
    Precision.BFLOAT16: {
        "bits": 16,
        "mantissa_bits": 7,
        "exponent_bits": 8,
        "max_value": 3.3895e38,  # Similar range to float32
        "min_normal": 1.175494e-38,
        "epsilon": 7.81e-3,
    },
}


def get_precision_info(precision: Precision) -> Dict[str, Any]:
    """Get information about a precision level."""
    return PRECISION_INFO[precision]


def cast_to_precision(value: float, precision: Precision) -> float:
    """
    Cast a float64 value to specified precision and back.

    This simulates the effect of storing in lower precision.

    Args:
        value: Float64 value
        precision: Target precision

    Returns:
        Value after round-trip through target precision
    """
    if precision == Precision.FLOAT64:
        return value

    info = PRECISION_INFO[precision]

    # Handle special values
    if value == float("inf"):
        return float("inf")
    elif value == float("-inf"):
        return float("-inf")
    elif value != value:  # NaN
        return float("nan")

    # Check for overflow
    if abs(value) > info["max_value"]:
        return float("inf") if value > 0 else float("-inf")

    # Check for underflow to zero
    if abs(value) < info["min_normal"] * info["epsilon"]:
        return 0.0

    # Simulate precision loss
    if precision == Precision.FLOAT32:
        # Use struct to do actual float32 conversion
        return struct.unpack("f", struct.pack("f", value))[0]

    elif precision == Precision.FLOAT16:
        # Simplified float16 simulation
        # A full implementation would use numpy or a similar library
        scale = 2.0 ** (info["mantissa_bits"] + 1)
        return round(value * scale) / scale

    elif precision == Precision.BFLOAT16:
        # BFloat16: keep exponent range of float32 but reduce mantissa
        # Truncate mantissa bits
        scale = 2.0 ** (info["mantissa_bits"] + 1)
        return round(value * scale) / scale

    else:
        raise ValueError(f"Unsupported precision: {precision}")


def tr_scalar_with_precision(value: float, precision: Precision = Precision.FLOAT64) -> TRScalar:
    """
    Create TRScalar with specified precision.

    Args:
        value: Input value
        precision: Desired precision

    Returns:
        TRScalar with value cast to specified precision
    """
    cast_value = cast_to_precision(value, precision)

    # Check if casting produced special values
    if cast_value == float("inf"):
        return pinf()
    elif cast_value == float("-inf"):
        return ninf()
    elif cast_value != cast_value:  # NaN
        return phi()
    else:
        return real(cast_value)


class PrecisionContext:
    """
    Context manager for precision-aware operations.

    Example:
        with PrecisionContext(Precision.FLOAT32):
            # All TR operations will use float32 precision
            result = tr_add(a, b)
    """

    _precision_stack: list[Precision] = [Precision.FLOAT64]

    def __init__(self, precision: Precision):
        self.precision = precision
        self._prev_precision = None

    def __enter__(self):
        self._prev_precision = self._precision_stack[-1]
        self._precision_stack.append(self.precision)
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self._precision_stack.pop()
        return False

    @classmethod
    def get_current_precision(cls) -> Precision:
        """Get the current precision context."""
        return cls._precision_stack[-1]

    @classmethod
    def reset(cls):
        """Reset precision stack to default."""
        cls._precision_stack = [Precision.FLOAT64]


def with_precision(tr_value: TRScalar, precision: Optional[Precision] = None) -> TRScalar:
    """
    Apply precision casting to a transreal value.

    Args:
        tr_value: Input transreal scalar
        precision: Target precision (uses context if None)

    Returns:
        TRScalar with precision applied
    """
    if precision is None:
        precision = PrecisionContext.get_current_precision()

    if tr_value.tag != TRTag.REAL:
        # Non-REAL values are unchanged by precision
        return tr_value

    return tr_scalar_with_precision(tr_value.value, precision)


# Precision-aware operations
def check_precision_overflow(value: float, precision: Precision) -> Optional[TRTag]:
    """
    Check if a value would overflow at given precision.

    Args:
        value: Value to check
        precision: Target precision

    Returns:
        TRTag.PINF/NINF if overflow would occur, None otherwise
    """
    info = PRECISION_INFO[precision]

    if value > info["max_value"]:
        return TRTag.PINF
    elif value < -info["max_value"]:
        return TRTag.NINF
    else:
        return None


def precision_safe_operation(
    op_name: str, *args: TRScalar, precision: Optional[Precision] = None
) -> TRScalar:
    """
    Perform operation with precision safety checks.

    Args:
        op_name: Operation name ('add', 'mul', etc.)
        args: TRScalar arguments
        precision: Target precision

    Returns:
        Result with precision-aware overflow handling
    """
    if precision is None:
        precision = PrecisionContext.get_current_precision()

    # Import operations
    from ..core import tr_add, tr_div, tr_mul, tr_sub

    ops = {
        "add": tr_add,
        "sub": tr_sub,
        "mul": tr_mul,
        "div": tr_div,
    }

    if op_name not in ops:
        raise ValueError(f"Unknown operation: {op_name}")

    # Perform operation
    result = ops[op_name](*args)

    # Apply precision to result
    return with_precision(result, precision)


# Mixed precision utilities
class MixedPrecisionStrategy:
    """
    Strategy for mixed precision computation.

    Allows different precisions for different parts of computation.
    """

    def __init__(
        self,
        compute_precision: Precision = Precision.FLOAT32,
        accumulate_precision: Precision = Precision.FLOAT64,
        output_precision: Precision = Precision.FLOAT32,
    ):
        """
        Initialize mixed precision strategy.

        Args:
            compute_precision: Precision for computations
            accumulate_precision: Precision for accumulations (reductions)
            output_precision: Precision for final outputs
        """
        self.compute_precision = compute_precision
        self.accumulate_precision = accumulate_precision
        self.output_precision = output_precision

    def compute(self, op_fn, *args):
        """Perform computation in compute precision."""
        with PrecisionContext(self.compute_precision):
            return op_fn(*args)

    def accumulate(self, values: list[TRScalar]) -> TRScalar:
        """Accumulate values in higher precision."""
        with PrecisionContext(self.accumulate_precision):
            from ..core import real, tr_add

            if not values:
                return real(0.0)

            result = values[0]
            for val in values[1:]:
                result = tr_add(result, val)

            return result

    def finalize(self, value: TRScalar) -> TRScalar:
        """Convert to output precision."""
        return with_precision(value, self.output_precision)


# Precision analysis utilities
def analyze_precision_requirements(values: list[float]) -> Dict[str, Any]:
    """
    Analyze precision requirements for a set of values.

    Args:
        values: List of float values

    Returns:
        Dictionary with precision analysis
    """
    import math

    if not values:
        return {
            "min_precision": Precision.FLOAT16,
            "recommended_precision": Precision.FLOAT32,
            "range": (0.0, 0.0),
            "needs_float64": False,
        }

    # Find range
    min_val = min(abs(v) for v in values if v != 0.0) if any(v != 0.0 for v in values) else 0.0
    max_val = max(abs(v) for v in values)

    # Check if values exceed float32 range
    float32_info = PRECISION_INFO[Precision.FLOAT32]
    needs_float64 = max_val > float32_info["max_value"] or (
        min_val > 0 and min_val < float32_info["min_normal"]
    )

    # Check if values fit in float16
    float16_info = PRECISION_INFO[Precision.FLOAT16]
    fits_float16 = max_val <= float16_info["max_value"] and (
        min_val == 0 or min_val >= float16_info["min_normal"]
    )

    # Determine minimal precision to satisfy relative error tolerance
    # Target: subsequent casting error should be <= 1e-6 relative
    target_rel_tol = 1e-6
    # Map precision ordering (lower index means lower precision)
    precision_order = [Precision.FLOAT16, Precision.FLOAT32, Precision.FLOAT64]

    # Smallest precision meeting the tolerance based on machine epsilon
    def meets_tol(p: Precision) -> bool:
        return PRECISION_INFO[p]["epsilon"] <= target_rel_tol

    if meets_tol(Precision.FLOAT16):
        min_by_error = Precision.FLOAT16
    elif meets_tol(Precision.FLOAT32):
        min_by_error = Precision.FLOAT32
    else:
        min_by_error = Precision.FLOAT64

    # Determine minimum and recommended precision
    # Determine baseline precision from range/denormal constraints
    if needs_float64:
        baseline_min = Precision.FLOAT64
    elif fits_float16:
        baseline_min = Precision.FLOAT16
    else:
        baseline_min = Precision.FLOAT32

    # Final min precision is the higher of baseline and error-based requirement
    def rank(p: Precision) -> int:
        return precision_order.index(p)

    min_precision = precision_order[max(rank(baseline_min), rank(min_by_error))]

    # Recommended precision favors float32 for stability unless float64 is required
    recommended = Precision.FLOAT64 if min_precision == Precision.FLOAT64 else Precision.FLOAT32

    return {
        "min_precision": min_precision,
        "recommended_precision": recommended,
        "range": (min_val, max_val),
        "needs_float64": needs_float64,
        "fits_float16": fits_float16,
    }


# Precision conversion warnings
def warn_precision_loss(from_precision: Precision, to_precision: Precision):
    """Issue warning about potential precision loss."""
    from_bits = PRECISION_INFO[from_precision]["bits"]
    to_bits = PRECISION_INFO[to_precision]["bits"]

    if from_bits > to_bits:
        warnings.warn(
            f"Converting from {from_precision.value} to {to_precision.value} "
            f"may result in precision loss ({from_bits} bits to {to_bits} bits)",
            category=UserWarning,
        )
