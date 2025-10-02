"""
Transreal arithmetic operations.

This module implements the core arithmetic operations for transreal numbers,
following the tag tables specified in the TR specification. All operations
are total (never throw exceptions) and deterministic.
"""

from __future__ import annotations

import math

from .precision_config import PrecisionConfig  # noqa: E402
from .tr_scalar import TRScalar, TRTag, bottom, ninf, phi, pinf, real
from .wheel_mode import WheelModeConfig  # noqa: E402


# Internal helper: accept callers that accidentally pass TRNode
def _ensure_trscalar(x):
    # If it's already TRScalar, return
    if isinstance(x, TRScalar):
        return x
    # If it looks like a TRNode (has .value which is TRScalar), unwrap once
    if hasattr(x, "value") and isinstance(getattr(x, "value"), TRScalar):
        return getattr(x, "value")
    return x  # Let downstream raise if unsupported


# Addition operation


def tr_add(a: TRScalar, b: TRScalar) -> TRScalar:
    """
    Transreal addition (a + b).

    Tag table for addition:
        | a \\ b | REAL | PINF | NINF | PHI |
        |-------|------|------|------|-----|
        | REAL  | REAL | PINF | NINF | PHI |
        | PINF  | PINF | PINF | PHI  | PHI |
        | NINF  | NINF | PHI  | NINF | PHI |
        | PHI   | PHI  | PHI  | PHI  | PHI |

    Args:
        a, b: Transreal scalars to add

    Returns:
        Transreal scalar representing a + b
    """
    a = _ensure_trscalar(a)
    b = _ensure_trscalar(b)
    # PHI propagates through everything
    if a.tag == TRTag.PHI or b.tag == TRTag.PHI:
        return phi()

    # Handle BOTTOM - it propagates (wheel mode)
    if a.tag == TRTag.BOTTOM or b.tag == TRTag.BOTTOM:
        return bottom()

    # Handle REAL + REAL
    if a.tag == TRTag.REAL and b.tag == TRTag.REAL:
        # Perform addition with precision enforcement on numeric payloads
        result = PrecisionConfig.enforce_precision(float(a.value) + float(b.value))

        # Check for overflow to infinity
        if math.isinf(result) or PrecisionConfig.check_overflow(result):
            if result > 0 or (a.value > 0 and b.value > 0):
                return pinf()
            else:
                return ninf()
        else:
            return real(result)

    # Handle infinity cases
    if a.tag == TRTag.PINF:
        if b.tag == TRTag.NINF:
            # +∞ + -∞ = Φ (or ⊥ in wheel mode)
            return bottom() if WheelModeConfig.is_wheel() else phi()
        elif b.tag == TRTag.PINF and WheelModeConfig.is_wheel():
            # +∞ + +∞ = ⊥ in wheel mode
            return bottom()
        else:
            return pinf()  # +∞ + REAL = +∞, +∞ + +∞ = +∞ (in TR mode)

    if a.tag == TRTag.NINF:
        if b.tag == TRTag.PINF:
            # -∞ + +∞ = Φ (or ⊥ in wheel mode)
            return bottom() if WheelModeConfig.is_wheel() else phi()
        elif b.tag == TRTag.NINF and WheelModeConfig.is_wheel():
            # -∞ + -∞ = ⊥ in wheel mode
            return bottom()
        else:
            return ninf()  # -∞ + REAL = -∞, -∞ + -∞ = -∞ (in TR mode)

    # Handle REAL + infinity (remaining cases)
    if a.tag == TRTag.REAL:
        if b.tag == TRTag.PINF:
            return pinf()
        else:  # b.tag == TRTag.NINF
            return ninf()

    # Should never reach here
    raise RuntimeError(f"Unhandled case in tr_add: {a.tag}, {b.tag}")


def tr_sub(a: TRScalar, b: TRScalar) -> TRScalar:
    """
    Transreal subtraction (a - b).

    Implemented as a + (-b).

    Args:
        a, b: Transreal scalars

    Returns:
        Transreal scalar representing a - b
    """
    return tr_add(a, tr_neg(b))


# Multiplication operation


def tr_mul(a: TRScalar, b: TRScalar) -> TRScalar:
    """
    Transreal multiplication (a × b).

    Tag table for multiplication:
        | a \\ b | REAL≠0 | 0  | PINF | NINF | PHI |
        |-------|--------|----|------|------|-----|
        | REAL≠0| REAL   | 0  | ±∞*  | ±∞*  | PHI |
        | 0     | 0      | 0  | PHI  | PHI  | PHI |
        | PINF  | ±∞*    | PHI| PINF | NINF | PHI |
        | NINF  | ±∞*    | PHI| NINF | PINF | PHI |
        | PHI   | PHI    | PHI| PHI  | PHI  | PHI |

    * Sign determined by sign rule

    Args:
        a, b: Transreal scalars to multiply

    Returns:
        Transreal scalar representing a × b
    """
    a = _ensure_trscalar(a)
    b = _ensure_trscalar(b)
    # PHI propagates through everything
    if a.tag == TRTag.PHI or b.tag == TRTag.PHI:
        return phi()

    # Handle BOTTOM - it propagates (wheel mode)
    if a.tag == TRTag.BOTTOM or b.tag == TRTag.BOTTOM:
        return bottom()

    # Handle REAL × REAL
    if a.tag == TRTag.REAL and b.tag == TRTag.REAL:
        # Perform multiplication (check overflow before precision enforcement)
        raw_result = float(a.value) * float(b.value)

        # Check for overflow to infinity first
        if math.isinf(raw_result) or PrecisionConfig.check_overflow(raw_result):
            # Determine sign from operands
            sign = 1 if (a.value >= 0) == (b.value >= 0) else -1
            if sign > 0:
                return pinf()
            else:
                return ninf()
        else:
            # Apply precision enforcement only if no overflow
            result = PrecisionConfig.enforce_precision(raw_result)
            return real(result)

    # Check for zero operands (critical for 0 × ∞ = Φ)
    a_is_zero = a.tag == TRTag.REAL and a.value == 0.0
    b_is_zero = b.tag == TRTag.REAL and b.value == 0.0

    # 0 × infinity = PHI (or BOTTOM in wheel mode)
    if a_is_zero and b.tag in (TRTag.PINF, TRTag.NINF):
        return bottom() if WheelModeConfig.is_wheel() else phi()
    if b_is_zero and a.tag in (TRTag.PINF, TRTag.NINF):
        return bottom() if WheelModeConfig.is_wheel() else phi()

    # 0 × anything else = 0
    if a_is_zero or b_is_zero:
        return real(0.0)

    # Determine sign for infinity results
    def get_sign(x: TRScalar) -> int:
        """Get sign of a transreal value (-1, 0, or 1)."""
        if x.tag == TRTag.REAL:
            if x.value > 0:
                return 1
            elif x.value < 0:
                return -1
            else:
                return 0
        elif x.tag == TRTag.PINF:
            return 1
        elif x.tag == TRTag.NINF:
            return -1
        else:  # PHI
            return 0  # Should not reach here

    sign_a = get_sign(a)
    sign_b = get_sign(b)
    result_sign = sign_a * sign_b

    # Handle remaining infinity cases
    if a.tag in (TRTag.PINF, TRTag.NINF) or b.tag in (TRTag.PINF, TRTag.NINF):
        if result_sign > 0:
            return pinf()
        else:
            return ninf()

    # Should never reach here
    raise RuntimeError(f"Unhandled case in tr_mul: {a.tag}, {b.tag}")


# Division operation


def tr_div(a: TRScalar, b: TRScalar) -> TRScalar:
    """
    Transreal division (a ÷ b).

    Tag table for division:
        | a \\ b | REAL>0 | REAL<0 | 0   | PINF | NINF | PHI |
        |-------|--------|--------|-----|------|------|-----|
        | REAL>0| REAL   | REAL   | PINF| 0    | 0    | PHI |
        | REAL<0| REAL   | REAL   | NINF| 0    | 0    | PHI |
        | 0     | 0      | 0      | PHI | 0    | 0    | PHI |
        | PINF  | PINF   | NINF   | PINF| PHI  | PHI  | PHI |
        | NINF  | NINF   | PINF   | NINF| PHI  | PHI  | PHI |
        | PHI   | PHI    | PHI    | PHI | PHI  | PHI  | PHI |

    Args:
        a, b: Transreal scalars (numerator, denominator)

    Returns:
        Transreal scalar representing a ÷ b
    """
    # PHI propagates through everything
    if a.tag == TRTag.PHI or b.tag == TRTag.PHI:
        return phi()

    # Handle BOTTOM - it propagates (wheel mode)
    if a.tag == TRTag.BOTTOM or b.tag == TRTag.BOTTOM:
        return bottom()

    a = _ensure_trscalar(a)
    b = _ensure_trscalar(b)
    # Handle denominator = 0
    if b.tag == TRTag.REAL and b.value == 0.0:
        # Need to check the sign of zero
        b_is_negative_zero = math.copysign(1.0, b.value) < 0

        if a.tag == TRTag.REAL:
            if a.value > 0:
                # positive / +0 = +∞, positive / -0 = -∞
                return ninf() if b_is_negative_zero else pinf()
            elif a.value < 0:
                # negative / +0 = -∞, negative / -0 = +∞
                return pinf() if b_is_negative_zero else ninf()
            else:
                return phi()  # 0 / 0 = Φ
        elif a.tag == TRTag.PINF:
            # +∞ / +0 = +∞, +∞ / -0 = -∞
            return ninf() if b_is_negative_zero else pinf()
        elif a.tag == TRTag.NINF:
            # -∞ / +0 = -∞, -∞ / -0 = +∞
            return pinf() if b_is_negative_zero else ninf()

    # Handle infinity ÷ infinity
    if a.tag in (TRTag.PINF, TRTag.NINF) and b.tag in (TRTag.PINF, TRTag.NINF):
        # ∞ / ∞ = Φ (or ⊥ in wheel mode)
        return bottom() if WheelModeConfig.is_wheel() else phi()

    # Handle finite ÷ infinity
    if a.tag == TRTag.REAL and b.tag in (TRTag.PINF, TRTag.NINF):
        return real(0.0)  # finite / ∞ = 0

    # Handle infinity ÷ finite
    if a.tag in (TRTag.PINF, TRTag.NINF) and b.tag == TRTag.REAL:
        if b.value > 0:
            return a  # ±∞ / positive = ±∞
        else:  # b.value < 0
            # Flip sign of infinity
            if a.tag == TRTag.PINF:
                return ninf()
            else:
                return pinf()

    # Handle REAL ÷ REAL (remaining case)
    if a.tag == TRTag.REAL and b.tag == TRTag.REAL:
        # Perform division with precision enforcement on numeric payloads
        result = PrecisionConfig.enforce_precision(float(a.value) / float(b.value))

        # Check for overflow to infinity
        if math.isinf(result) or PrecisionConfig.check_overflow(result):
            # Determine sign from operands
            sign = 1 if (a.value >= 0) == (b.value >= 0) else -1
            if sign > 0:
                return pinf()
            else:
                return ninf()
        else:
            return real(result)

    # Should never reach here
    raise RuntimeError(f"Unhandled case in tr_div: {a.tag}, {b.tag}")


# Unary operations


def tr_neg(x: TRScalar) -> TRScalar:
    """
    Transreal negation (-x).

    NOTE: This function expects a TRScalar. Call sites that pass a TRNode
    should pass .value. This ensures correct behavior in autodiff contexts
    and avoids mishandling TRNode vs TRScalar types.

    Rules:
        - -REAL = REAL with negated value
        - -PINF = NINF
        - -NINF = PINF
        - -PHI = PHI
        - -BOTTOM = BOTTOM (wheel mode)
    """
    if x.tag == TRTag.REAL:
        return real(-x.value)
    if x.tag == TRTag.PINF:
        return ninf()
    if x.tag == TRTag.NINF:
        return pinf()
    if x.tag == TRTag.PHI:
        return phi()
    return bottom()


def tr_abs(x: TRScalar) -> TRScalar:
    """
    Transreal absolute value (|x|).

    Rules:
        - |REAL| = REAL with |value|
        - |PINF| = PINF
        - |NINF| = PINF
        - |PHI| = PHI
        - |BOTTOM| = BOTTOM (wheel mode)

    Args:
        x: Transreal scalar

    Returns:
        Transreal scalar representing |x|
    """
    if x.tag == TRTag.REAL:
        return real(abs(x.value))
    elif x.tag == TRTag.PINF:
        return pinf()
    elif x.tag == TRTag.NINF:
        return pinf()
    elif x.tag == TRTag.PHI:
        return phi()
    else:  # BOTTOM
        return bottom()


def tr_sign(x: TRScalar) -> TRScalar:
    """
    Transreal sign function.

    Rules:
        - sign(REAL) = REAL in {-1, 0, 1}
        - sign(PINF) = REAL +1
        - sign(NINF) = REAL -1
        - sign(PHI) = PHI
        - sign(BOTTOM) = BOTTOM (wheel mode)

    Args:
        x: Transreal scalar

    Returns:
        Transreal scalar representing sign(x)
    """
    if x.tag == TRTag.REAL:
        if x.value > 0:
            return real(1.0)
        elif x.value < 0:
            return real(-1.0)
        else:
            return real(0.0)
    elif x.tag == TRTag.PINF:
        return real(1.0)
    elif x.tag == TRTag.NINF:
        return real(-1.0)
    elif x.tag == TRTag.PHI:
        return phi()
    else:  # BOTTOM
        return bottom()


# Domain-aware operations


def tr_log(x: TRScalar) -> TRScalar:
    """
    Transreal natural logarithm.

    Rules:
        - log(REAL x>0) = REAL ln(x)
        - log(REAL x≤0) = PHI
        - log(PINF) = PINF
        - log(NINF) = PHI
        - log(PHI) = PHI
        - log(BOTTOM) = BOTTOM (wheel mode)

    Args:
        x: Transreal scalar

    Returns:
        Transreal scalar representing log(x)
    """
    if x.tag == TRTag.PHI:
        return phi()

    if x.tag == TRTag.BOTTOM:
        return bottom()

    if x.tag == TRTag.REAL:
        if x.value > 0:
            # Compute log with precision enforcement
            result = PrecisionConfig.enforce_precision(math.log(x.value))
            return real(result)
        else:
            return phi()  # log(x≤0) = PHI
    elif x.tag == TRTag.PINF:
        return pinf()  # log(+∞) = +∞
    else:  # NINF
        return phi()  # log(-∞) = PHI


def tr_sqrt(x: TRScalar) -> TRScalar:
    """
    Transreal square root.

    Rules:
        - sqrt(REAL x≥0) = REAL √x
        - sqrt(REAL x<0) = PHI
        - sqrt(PINF) = PINF
        - sqrt(NINF) = PHI
        - sqrt(PHI) = PHI
        - sqrt(BOTTOM) = BOTTOM (wheel mode)

    Args:
        x: Transreal scalar

    Returns:
        Transreal scalar representing √x
    """
    if x.tag == TRTag.PHI:
        return phi()

    if x.tag == TRTag.BOTTOM:
        return bottom()

    if x.tag == TRTag.REAL:
        if x.value >= 0:
            # Compute sqrt with precision enforcement
            result = PrecisionConfig.enforce_precision(math.sqrt(x.value))
            return real(result)
        else:
            return phi()  # sqrt(negative) = PHI
    elif x.tag == TRTag.PINF:
        return pinf()  # sqrt(+∞) = +∞
    else:  # NINF
        return phi()  # sqrt(-∞) = PHI


def tr_pow_int(x: TRScalar, k: int) -> TRScalar:
    """
    Transreal integer power (x^k).

    Computed by repeated multiplication and division under TR rules.

    Special cases:
        - 0^0 = PHI
        - (±∞)^0 = PHI
        - 0^k = 0 for k > 0
        - 0^k = ±∞ for k < 0 (using division rules)

    Args:
        x: Transreal scalar base
        k: Integer exponent

    Returns:
        Transreal scalar representing x^k
    """
    # Handle BOTTOM propagation
    if x.tag == TRTag.BOTTOM:
        return bottom()

    # Handle k = 0 special cases
    if k == 0:
        if x.tag == TRTag.REAL and x.value == 0.0:
            return phi()  # 0^0 = PHI
        elif x.tag in (TRTag.PINF, TRTag.NINF):
            return phi()  # (±∞)^0 = PHI
        elif x.tag == TRTag.PHI:
            return phi()  # PHI^0 = PHI
        else:
            return real(1.0)  # x^0 = 1 for finite x ≠ 0

    # Handle k = 1
    if k == 1:
        return x

    # Handle negative exponents via reciprocal
    if k < 0:
        # x^(-k) = 1 / x^k
        return tr_div(real(1.0), tr_pow_int(x, -k))

    # Handle positive exponents by repeated multiplication
    result = real(1.0)
    base = x
    exp = k

    # Binary exponentiation for efficiency
    while exp > 0:
        if exp % 2 == 1:
            result = tr_mul(result, base)
        base = tr_mul(base, base)
        exp //= 2

    return result
