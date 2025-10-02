"""
Transreal scalar type implementation.

This module defines the core TRScalar type that extends real arithmetic with
special values for infinity and undefined forms, making all operations total.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Optional, Union

from .precision_config import PrecisionConfig


class _FloatWithValue(float):
    """Float subclass that also exposes a `.value` attribute.

    This helps tests that sometimes access `x.value.value` when `x` is a
    TRScalar (so `x.value` is a float). Returning this wrapper keeps float
    semantics while allowing a second `.value` dereference.
    """

    @property
    def value(self) -> float:
        return float(self)


class TRTag(IntEnum):
    """Tags for transreal values."""

    # Explicit codes align with NumPy bridge encoding
    REAL = 0  # Finite real number
    PINF = 1  # Positive infinity
    NINF = 2  # Negative infinity
    PHI = 3  # Nullity (undefined/indeterminate)
    BOTTOM = 4  # Bottom element (⊥) for wheel mode

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.name


class TRScalar:
    """
    Transreal scalar value.

    A transreal number consists of a value (float64) and a tag indicating
    its type (REAL, PINF, NINF, or PHI). The value is only meaningful
    when tag=REAL.

    Attributes:
        value: The numeric value (only valid when tag=REAL)
        tag: The transreal tag indicating the type
    """

    __slots__ = ("_value", "_tag")

    def __init__(self, value: float, tag: TRTag) -> None:
        """
        Initialize a transreal scalar.

        Args:
            value: Numeric value (only used when tag=REAL)
            tag: The transreal tag

        Note:
            Direct construction is discouraged. Use factory functions
            real(), pinf(), ninf(), phi() instead.
        """
        if tag == TRTag.REAL:
            # Enforce precision for REAL values
            value = PrecisionConfig.enforce_precision(value)

            if math.isnan(value) or math.isinf(value):
                raise ValueError(
                    f"REAL tag requires finite value, got {value}. "
                    "Use from_ieee() to convert IEEE special values."
                )
            self._value = value
        else:
            # Use NaN for non-REAL values, with proper precision
            self._value = PrecisionConfig.enforce_precision(float("nan"))

        self._tag = tag

    @property
    def value(self) -> float:
        """Get the numeric value (only meaningful when tag=REAL)."""
        # Return a float-compatible object with a `.value` attribute for
        # robustness in contexts that access `x.value.value`.
        return _FloatWithValue(self._value)

    @property
    def tag(self) -> TRTag:
        """Get the transreal tag."""
        return self._tag

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        if self._tag == TRTag.REAL:
            return f"TRScalar({self._value}, {self._tag})"
        else:
            return f"TRScalar({self._tag})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self._tag == TRTag.REAL:
            return str(self._value)
        elif self._tag == TRTag.PINF:
            return "+∞"
        elif self._tag == TRTag.NINF:
            return "-∞"
        elif self._tag == TRTag.PHI:
            return "Φ"
        else:  # BOTTOM
            return "⊥"

    def __format__(self, format_spec: str) -> str:
        """Formatted string representation, compatible with floats for REAL values."""
        try:
            return format(float(self), format_spec)
        except Exception:
            return str(self)

    def __eq__(self, other: Any) -> bool:
        """
        Check equality of transreal values.

        Two transreal values are equal if they have the same tag and,
        for REAL values, the same numeric value.
        """
        if not isinstance(other, TRScalar):
            return NotImplemented

        if self._tag != other._tag:
            return False

        if self._tag == TRTag.REAL:
            # Handle floating point comparison carefully
            # Exact equality for transreal semantics
            return self._value == other._value

        # Non-REAL values with same tag are equal
        return True

    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        if self._tag == TRTag.REAL:
            return hash((self._tag, self._value))
        return hash(self._tag)

    def __float__(self) -> float:
        """Allow implicit conversion to float for REAL values.

        For non-REAL tags, returns NaN to preserve safety in aggregations.
        """
        if self._tag == TRTag.REAL:
            return float(self._value)
        return float("nan")

    def __bool__(self) -> bool:
        """
        Truth value testing.

        - REAL: True if value != 0
        - PINF, NINF: True (infinities are "truthy")
        - PHI: False (nullity is "falsy")
        """
        if self._tag == TRTag.REAL:
            return bool(self._value != 0.0)
        elif self._tag in (TRTag.PINF, TRTag.NINF):
            return True
        else:  # PHI
            return False

    # Rich comparisons (only meaningful for REAL values)
    def _as_real_float(self, other: Any) -> Optional[float]:
        try:
            if isinstance(other, TRScalar):
                return float(other)
            return float(other)
        except Exception:
            return None

    def __lt__(self, other: Any) -> bool:
        other_val = self._as_real_float(other)
        if self._tag != TRTag.REAL or other_val is None or math.isnan(other_val):
            return False
        return float(self._value) < other_val

    def __le__(self, other: Any) -> bool:
        other_val = self._as_real_float(other)
        if self._tag != TRTag.REAL or other_val is None or math.isnan(other_val):
            return False
        return float(self._value) <= other_val

    def __gt__(self, other: Any) -> bool:
        other_val = self._as_real_float(other)
        if self._tag != TRTag.REAL or other_val is None or math.isnan(other_val):
            return False
        return float(self._value) > other_val

    def __ge__(self, other: Any) -> bool:
        other_val = self._as_real_float(other)
        if self._tag != TRTag.REAL or other_val is None or math.isnan(other_val):
            return False
        return float(self._value) >= other_val

    def is_real(self) -> bool:
        """Check if this is a finite real value."""
        return self._tag == TRTag.REAL

    def is_pinf(self) -> bool:
        """Check if this is positive infinity."""
        return self._tag == TRTag.PINF

    def is_ninf(self) -> bool:
        """Check if this is negative infinity."""
        return self._tag == TRTag.NINF

    def is_phi(self) -> bool:
        """Check if this is nullity (undefined)."""
        return self._tag == TRTag.PHI

    def is_finite(self) -> bool:
        """Check if this is a finite value (REAL)."""
        return self._tag == TRTag.REAL

    def is_infinite(self) -> bool:
        """Check if this is an infinity (PINF or NINF)."""
        return self._tag in (TRTag.PINF, TRTag.NINF)

    # Arithmetic operator overloading

    def __add__(self, other: Union[TRScalar, float, int]) -> TRScalar:
        """Addition operator (+)."""
        from .tr_ops import tr_add

        if not isinstance(other, TRScalar):
            try:
                other = real(float(other))
            except Exception:
                pass
        return tr_add(self, other)

    def __radd__(self, other: Union[float, int]) -> TRScalar:
        """Right addition operator (other + self)."""
        from .tr_ops import tr_add

        try:
            other_val = real(float(other))
        except Exception:
            other_val = other  # type: ignore
        return tr_add(other_val, self)

    def __sub__(self, other: Union[TRScalar, float, int]) -> TRScalar:
        """Subtraction operator (-)."""
        from .tr_ops import tr_sub

        if not isinstance(other, TRScalar):
            try:
                other = real(float(other))
            except Exception:
                pass
        return tr_sub(self, other)

    def __rsub__(self, other: Union[float, int]) -> TRScalar:
        """Right subtraction operator (other - self)."""
        from .tr_ops import tr_sub

        try:
            other_val = real(float(other))
        except Exception:
            other_val = other  # type: ignore
        return tr_sub(other_val, self)

    def __mul__(self, other: Union[TRScalar, float, int]) -> TRScalar:
        """Multiplication operator (*)."""
        from .tr_ops import tr_mul

        if not isinstance(other, TRScalar):
            try:
                other = real(float(other))
            except Exception:
                pass
        return tr_mul(self, other)

    def __rmul__(self, other: Union[float, int]) -> TRScalar:
        """Right multiplication operator (other * self)."""
        from .tr_ops import tr_mul

        try:
            other_val = real(float(other))
        except Exception:
            other_val = other  # type: ignore
        return tr_mul(other_val, self)

    def __truediv__(self, other: Union[TRScalar, float, int]) -> TRScalar:
        """Division operator (/)."""
        from .tr_ops import tr_div

        if not isinstance(other, TRScalar):
            try:
                other = real(float(other))
            except Exception:
                pass
        return tr_div(self, other)

    def __rtruediv__(self, other: Union[float, int]) -> TRScalar:
        """Right division operator (other / self)."""
        from .tr_ops import tr_div

        try:
            other_val = real(float(other))
        except Exception:
            other_val = other  # type: ignore
        return tr_div(other_val, self)

    def __neg__(self) -> TRScalar:
        """Negation operator (-)."""
        from .tr_ops import tr_neg

        return tr_neg(self)

    def __abs__(self) -> TRScalar:
        """Absolute value operator (abs())."""
        from .tr_ops import tr_abs

        return tr_abs(self)

    def __pow__(self, power: int) -> TRScalar:
        """Power operator (**) for integer exponents."""
        from .tr_ops import tr_pow_int

        if not isinstance(power, int):
            raise TypeError(f"Transreal power requires integer exponent, got {type(power)}")
        return tr_pow_int(self, power)


# Factory functions for creating transreal scalars


def real(value: Union[float, int]) -> TRScalar:
    """
    Create a transreal scalar with REAL tag.

    Args:
        value: A finite numeric value

    Returns:
        TRScalar with REAL tag, value enforced to default precision

    Raises:
        ValueError: If value is not finite (NaN or infinity)
    """
    # Enforce precision
    value = PrecisionConfig.enforce_precision(value)

    if math.isnan(value) or math.isinf(value):
        raise ValueError(
            f"real() requires finite value, got {value}. "
            "Use from_ieee() for IEEE special values."
        )

    # Check for overflow in current precision
    if PrecisionConfig.check_overflow(float(value)):
        # Value would overflow, return appropriate infinity
        if value > 0:
            return pinf()
        else:
            return ninf()

    return TRScalar(value, TRTag.REAL)


def pinf() -> TRScalar:
    """Create positive infinity."""
    return TRScalar(float("nan"), TRTag.PINF)


def ninf() -> TRScalar:
    """Create negative infinity."""
    return TRScalar(float("nan"), TRTag.NINF)


def phi() -> TRScalar:
    """Create nullity (undefined/indeterminate form)."""
    return TRScalar(float("nan"), TRTag.PHI)


def bottom() -> TRScalar:
    """
    Create a BOTTOM (⊥) value.

    This is only meaningful in wheel mode and represents
    the bottom element of the algebraic structure.
    """
    return TRScalar(float("nan"), TRTag.BOTTOM)


# Type checking functions


def is_real(x: TRScalar) -> bool:
    """Check if a transreal value has REAL tag."""
    return x.tag == TRTag.REAL


def is_pinf(x: TRScalar) -> bool:
    """Check if a transreal value is positive infinity."""
    return x.tag == TRTag.PINF


def is_ninf(x: TRScalar) -> bool:
    """Check if a transreal value is negative infinity."""
    return x.tag == TRTag.NINF


def is_phi(x: TRScalar) -> bool:
    """Check if a transreal value is nullity."""
    return x.tag == TRTag.PHI


def is_bottom(x: TRScalar) -> bool:
    """Check if a transreal value is bottom (⊥)."""
    return x.tag == TRTag.BOTTOM


def is_finite(x: TRScalar) -> bool:
    """Check if a transreal value is finite (REAL)."""
    return x.tag == TRTag.REAL


def is_infinite(x: TRScalar) -> bool:
    """Check if a transreal value is infinite (PINF or NINF)."""
    return x.tag in (TRTag.PINF, TRTag.NINF)
