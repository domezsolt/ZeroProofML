"""
Reduction operations for transreal values.

This module implements aggregation operations (sum, mean, etc.) that handle
non-REAL values according to the specified reduction mode.
"""

from typing import List

from .precision_config import PrecisionConfig
from .reduction import ReductionMode
from .tr_ops import tr_add, tr_div
from .tr_scalar import TRScalar, TRTag, bottom, ninf, phi, pinf, real

# Global toggle for deterministic compensated reductions (set by policy)
_DETERMINISTIC_REDUCTION: bool = False


def set_deterministic_reduction(enabled: bool) -> None:
    """
    Enable/disable deterministic compensated reductions (Kahan/Neumaier).

    This is intended to be driven by the active TR policy. When enabled,
    REAL-only sum/mean use a compensated summation to reduce rounding error
    while preserving TR semantics for non-REAL cases handled above.
    """
    global _DETERMINISTIC_REDUCTION
    _DETERMINISTIC_REDUCTION = bool(enabled)


def _compensated_sum_real(values: List[TRScalar]) -> TRScalar:
    """
    Neumaier compensated summation on REAL TRScalars.

    Assumes all inputs have tag == TRTag.REAL (callers must filter/guard).
    Returns a REAL TRScalar with improved numerical stability.
    """
    if not values:
        return real(0.0)
    # Neumaier algorithm
    s = 0.0
    c = 0.0
    for v in values:
        if v.tag != TRTag.REAL:
            # Safety: should not occur when caller guards, but fall back
            return tr_sum(values, mode=ReductionMode.STRICT)
        x = float(v.value)
        t = s + x
        # |s| >= |x| ?
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return real(s + c)


def tr_sum(values: List[TRScalar], mode: ReductionMode = ReductionMode.STRICT) -> TRScalar:
    """
    Sum reduction over transreal values.

    Args:
        values: List of transreal scalars to sum
        mode: Reduction mode (STRICT or DROP_NULL)

    Returns:
        Sum of values according to the specified mode

    Behavior:
        STRICT mode:
        - If any PHI present → PHI
        - If any BOTTOM present → BOTTOM (wheel mode)
        - If both PINF and NINF present → PHI
        - If any infinity → that infinity
        - Otherwise → sum of REAL values

        DROP_NULL mode:
        - Ignores PHI and BOTTOM values
        - If no non-null values → PHI (or BOTTOM if all were BOTTOM)
        - Otherwise applies STRICT rules to non-null values
    """
    if not values:
        return real(0.0)

    if mode == ReductionMode.DROP_NULL:
        # Filter out PHI and BOTTOM values
        non_null_values = [v for v in values if v.tag not in (TRTag.PHI, TRTag.BOTTOM)]
        if not non_null_values:
            # If all values were null, return appropriate null value
            if any(v.tag == TRTag.BOTTOM for v in values):
                return bottom()
            else:
                return phi()
        values = non_null_values
    elif mode == ReductionMode.STRICT:
        # Check for any PHI or BOTTOM
        if any(v.tag == TRTag.PHI for v in values):
            return phi()
        if any(v.tag == TRTag.BOTTOM for v in values):
            return bottom()

    # Check for infinities
    has_pinf = any(v.tag == TRTag.PINF for v in values)
    has_ninf = any(v.tag == TRTag.NINF for v in values)

    if has_pinf and has_ninf:
        return phi()  # ∞ + (-∞) = PHI
    elif has_pinf:
        return pinf()
    elif has_ninf:
        return ninf()

    # Sum REAL values (compensated if enabled)
    if _DETERMINISTIC_REDUCTION:
        real_values = [v for v in values if v.tag == TRTag.REAL]
        return _compensated_sum_real(real_values)
    else:
        result = real(0.0)
        for v in values:
            if v.tag == TRTag.REAL:
                result = tr_add(result, v)
        return result


def tr_mean(values: List[TRScalar], mode: ReductionMode = ReductionMode.STRICT) -> TRScalar:
    """
    Mean reduction over transreal values.

    Args:
        values: List of transreal scalars
        mode: Reduction mode (STRICT or DROP_NULL)

    Returns:
        Mean of values according to the specified mode

    Behavior:
        - Computes sum according to mode
        - Divides by count of included values
        - Empty input or all-PHI with DROP_NULL → PHI
    """
    if not values:
        return phi()

    if mode == ReductionMode.DROP_NULL:
        # Count non-PHI values
        non_phi_values = [v for v in values if v.tag != TRTag.PHI]
        if not non_phi_values:
            return phi()
        count = len(non_phi_values)
    else:
        count = len(values)

    # Compute sum
    total = tr_sum(values, mode)

    # If sum is non-REAL, return it
    if total.tag != TRTag.REAL:
        return total

    # Divide by count
    return tr_div(total, real(float(count)))


def tr_prod(values: List[TRScalar], mode: ReductionMode = ReductionMode.STRICT) -> TRScalar:
    """
    Product reduction over transreal values.

    Args:
        values: List of transreal scalars
        mode: Reduction mode (STRICT or DROP_NULL)

    Returns:
        Product of values according to the specified mode
    """
    if not values:
        return real(1.0)

    if mode == ReductionMode.DROP_NULL:
        # Filter out PHI values
        non_phi_values = [v for v in values if v.tag != TRTag.PHI]
        if not non_phi_values:
            return phi()
        values = non_phi_values
    elif mode == ReductionMode.STRICT:
        # Check for any PHI
        if any(v.tag == TRTag.PHI for v in values):
            return phi()

    # Deterministic product if enabled and all REAL: use pairwise multiplication
    if _DETERMINISTIC_REDUCTION:
        all_real = all(v.tag == TRTag.REAL for v in values)
        if all_real:
            return _pairwise_prod_real(values)

    # Fallback: sequential TR multiply (preserves TR semantics)
    result = real(1.0)
    for v in values:
        result = tr_mul(result, v)
        if result.tag == TRTag.PHI:
            return result
    return result


def _pairwise_prod_real(values: List[TRScalar]) -> TRScalar:
    """
    Pairwise product for REAL TRScalars (deterministic tree).

    Preserves TR semantics via tr_mul; exact for many simple products.
    """
    if not values:
        return real(1.0)
    # Any zero yields zero immediately
    for v in values:
        if v.tag == TRTag.REAL and float(v.value) == 0.0:
            return real(0.0)

    def _prod(lo: int, hi: int) -> TRScalar:
        if hi - lo == 1:
            return values[lo]
        mid = (lo + hi) // 2
        left = _prod(lo, mid)
        right = _prod(mid, hi)
        return tr_mul(left, right)

    return _prod(0, len(values))


def tr_min(values: List[TRScalar], mode: ReductionMode = ReductionMode.STRICT) -> TRScalar:
    """
    Minimum reduction over transreal values.

    Args:
        values: List of transreal scalars
        mode: Reduction mode (STRICT or DROP_NULL)

    Returns:
        Minimum value according to the specified mode

    Order: NINF < REAL values < PINF, PHI is incomparable
    """
    if not values:
        return phi()

    if mode == ReductionMode.DROP_NULL:
        # Filter out PHI values
        non_phi_values = [v for v in values if v.tag != TRTag.PHI]
        if not non_phi_values:
            return phi()
        values = non_phi_values
    elif mode == ReductionMode.STRICT:
        # Check for any PHI
        if any(v.tag == TRTag.PHI for v in values):
            return phi()

    # Check for NINF (smallest value)
    if any(v.tag == TRTag.NINF for v in values):
        return ninf()

    # Find minimum among REAL and PINF
    min_val = None
    for v in values:
        if v.tag == TRTag.REAL:
            if min_val is None or v.value < min_val.value:
                min_val = v
        elif v.tag == TRTag.PINF and min_val is None:
            min_val = v

    return min_val if min_val is not None else phi()


def tr_max(values: List[TRScalar], mode: ReductionMode = ReductionMode.STRICT) -> TRScalar:
    """
    Maximum reduction over transreal values.

    Args:
        values: List of transreal scalars
        mode: Reduction mode (STRICT or DROP_NULL)

    Returns:
        Maximum value according to the specified mode

    Order: NINF < REAL values < PINF, PHI is incomparable
    """
    if not values:
        return phi()

    if mode == ReductionMode.DROP_NULL:
        # Filter out PHI values
        non_phi_values = [v for v in values if v.tag != TRTag.PHI]
        if not non_phi_values:
            return phi()
        values = non_phi_values
    elif mode == ReductionMode.STRICT:
        # Check for any PHI
        if any(v.tag == TRTag.PHI for v in values):
            return phi()

    # Check for PINF (largest value)
    if any(v.tag == TRTag.PINF for v in values):
        return pinf()

    # Find maximum among REAL and NINF
    max_val = None
    for v in values:
        if v.tag == TRTag.REAL:
            if max_val is None or v.value > max_val.value:
                max_val = v
        elif v.tag == TRTag.NINF and max_val is None:
            max_val = v

    return max_val if max_val is not None else phi()


# Import tr_mul at the end to avoid circular import
from .tr_ops import tr_mul  # noqa: E402
