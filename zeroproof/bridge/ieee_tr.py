"""
IEEE-754 ↔ Transreal bridge functions.

This module provides conversion between IEEE-754 floating point values
and transreal values, preserving semantics across representations.
"""

import math
from typing import TYPE_CHECKING

from ..core import TRScalar, TRTag, ninf, phi, pinf, real
from ..core.precision_config import PrecisionConfig

if TYPE_CHECKING:
    pass


def from_ieee(value: float) -> TRScalar:
    """
    Convert IEEE-754 float to transreal value.

    Mapping:
        - finite float → (value, REAL)
        - +∞ → (—, PINF)
        - -∞ → (—, NINF)
        - NaN → (—, PHI)

    Args:
        value: IEEE-754 float value

    Returns:
        Corresponding transreal scalar
    """
    if math.isnan(value):
        return phi()
    elif math.isinf(value):
        if value > 0:
            return pinf()
        else:
            return ninf()
    else:
        # Finite value (including ±0.0, subnormals)
        # Enforce precision before creating TR value
        value_precise = PrecisionConfig.enforce_precision(value)
        return real(value_precise)


def to_ieee(tr_value: TRScalar) -> float:
    """
    Convert transreal value to IEEE-754 float.

    Mapping:
        - (value, REAL) → value
        - (—, PINF) → +∞
        - (—, NINF) → -∞
        - (—, PHI) → NaN
        - (—, BOTTOM) → NaN (wheel mode)

    Args:
        tr_value: Transreal scalar

    Returns:
        Corresponding IEEE-754 float
    """
    if tr_value.tag == TRTag.REAL:
        # Return value with enforced precision
        return float(PrecisionConfig.enforce_precision(tr_value.value))
    elif tr_value.tag == TRTag.PINF:
        return float("inf")
    elif tr_value.tag == TRTag.NINF:
        return float("-inf")
    elif tr_value.tag == TRTag.PHI:
        return float("nan")
    else:  # BOTTOM
        return float("nan")


# Import numpy bridge if available
try:
    from .numpy_bridge import NUMPY_AVAILABLE, TRArray
    from .numpy_bridge import from_numpy as from_numpy_impl
    from .numpy_bridge import to_numpy as to_numpy_impl
except ImportError:
    NUMPY_AVAILABLE = False
    TRArray = None
    from_numpy_impl = None
    to_numpy_impl = None


def from_numpy(arr):
    """
    Convert NumPy array to transreal representation.

    Args:
        arr: NumPy array or scalar

    Returns:
        TRArray for arrays, TRScalar for scalars

    Raises:
        ImportError: If NumPy is not available
    """
    if not NUMPY_AVAILABLE or from_numpy_impl is None:
        raise ImportError("NumPy is required for from_numpy(). Install with: pip install numpy")

    return from_numpy_impl(arr)


def to_numpy(tr_obj):
    """
    Convert transreal object to NumPy representation.

    Args:
        tr_obj: TRScalar or TRArray

    Returns:
        float for TRScalar, NumPy array for TRArray

    Raises:
        ImportError: If NumPy is not available
    """
    if not NUMPY_AVAILABLE or to_numpy_impl is None:
        raise ImportError("NumPy is required for to_numpy(). Install with: pip install numpy")

    return to_numpy_impl(tr_obj)
