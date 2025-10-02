"""
Separated implementations of TR and Wheel operations.

This module demonstrates strict separation of TR and Wheel semantics,
ensuring they never mix within a single operation.
"""

from typing import Callable

from .mode_isolation import (
    check_value_mode_compatibility,
    compile_time_switch,
    isolated_operation,
    tr_only,
    wheel_only,
)
from .tr_scalar import TRScalar, TRTag, bottom, ninf, phi, pinf, real

# =============================================================================
# Pure TR implementations (no BOTTOM handling)
# =============================================================================


@tr_only
def tr_add_pure(a: TRScalar, b: TRScalar) -> TRScalar:
    """
    Pure transreal addition - no Wheel semantics.

    This implementation never produces or handles BOTTOM.
    """
    # PHI propagates
    if a.tag == TRTag.PHI or b.tag == TRTag.PHI:
        return phi()

    # REAL + REAL
    if a.tag == TRTag.REAL and b.tag == TRTag.REAL:
        result = a.value + b.value
        # Check overflow
        if abs(result) > 1e308:  # Near float max
            return pinf() if result > 0 else ninf()
        return real(result)

    # Infinity cases - TR semantics only
    if a.tag == TRTag.PINF:
        if b.tag == TRTag.NINF:
            return phi()  # +∞ + -∞ = Φ in TR
        else:
            return pinf()  # +∞ + anything else = +∞

    if a.tag == TRTag.NINF:
        if b.tag == TRTag.PINF:
            return phi()  # -∞ + +∞ = Φ in TR
        else:
            return ninf()  # -∞ + anything else = -∞

    # REAL + infinity
    if a.tag == TRTag.REAL:
        return b  # REAL + ±∞ = ±∞

    if b.tag == TRTag.REAL:
        return a  # ±∞ + REAL = ±∞

    # Should not reach here
    raise RuntimeError(f"Unhandled TR case: {a.tag} + {b.tag}")


@tr_only
def tr_mul_pure(a: TRScalar, b: TRScalar) -> TRScalar:
    """
    Pure transreal multiplication - no Wheel semantics.

    This implementation never produces or handles BOTTOM.
    """
    # PHI propagates
    if a.tag == TRTag.PHI or b.tag == TRTag.PHI:
        return phi()

    # Check for zeros
    a_is_zero = a.tag == TRTag.REAL and a.value == 0.0
    b_is_zero = b.tag == TRTag.REAL and b.value == 0.0

    # 0 × ∞ = Φ in TR
    if a_is_zero and b.tag in (TRTag.PINF, TRTag.NINF):
        return phi()
    if b_is_zero and a.tag in (TRTag.PINF, TRTag.NINF):
        return phi()

    # 0 × finite = 0
    if a_is_zero or b_is_zero:
        return real(0.0)

    # REAL × REAL
    if a.tag == TRTag.REAL and b.tag == TRTag.REAL:
        result = a.value * b.value
        # Check overflow
        if abs(result) > 1e308:
            sign = 1 if (a.value > 0) == (b.value > 0) else -1
            return pinf() if sign > 0 else ninf()
        return real(result)

    # Determine sign for infinity results
    def sign(x: TRScalar) -> int:
        if x.tag == TRTag.REAL:
            return 1 if x.value > 0 else -1 if x.value < 0 else 0
        elif x.tag == TRTag.PINF:
            return 1
        elif x.tag == TRTag.NINF:
            return -1
        else:
            return 0

    result_sign = sign(a) * sign(b)

    # Any infinity in multiplication gives infinity
    if a.tag in (TRTag.PINF, TRTag.NINF) or b.tag in (TRTag.PINF, TRTag.NINF):
        return pinf() if result_sign > 0 else ninf()

    raise RuntimeError(f"Unhandled TR case: {a.tag} × {b.tag}")


# =============================================================================
# Pure Wheel implementations (with BOTTOM handling)
# =============================================================================


@wheel_only
def wheel_add_pure(a: TRScalar, b: TRScalar) -> TRScalar:
    """
    Pure wheel addition - strict algebraic control.

    Implements wheel axioms including ∞ + ∞ = ⊥.
    """
    # BOTTOM propagates
    if a.tag == TRTag.BOTTOM or b.tag == TRTag.BOTTOM:
        return bottom()

    # PHI maps to BOTTOM in wheel mode
    if a.tag == TRTag.PHI or b.tag == TRTag.PHI:
        return bottom()

    # REAL + REAL
    if a.tag == TRTag.REAL and b.tag == TRTag.REAL:
        result = a.value + b.value
        if abs(result) > 1e308:
            return pinf() if result > 0 else ninf()
        return real(result)

    # Wheel axioms for infinities
    if a.tag == TRTag.PINF:
        if b.tag == TRTag.NINF:
            return bottom()  # +∞ + -∞ = ⊥
        elif b.tag == TRTag.PINF:
            return bottom()  # +∞ + +∞ = ⊥ (wheel axiom)
        else:
            return pinf()  # +∞ + REAL = +∞

    if a.tag == TRTag.NINF:
        if b.tag == TRTag.PINF:
            return bottom()  # -∞ + +∞ = ⊥
        elif b.tag == TRTag.NINF:
            return bottom()  # -∞ + -∞ = ⊥ (wheel axiom)
        else:
            return ninf()  # -∞ + REAL = -∞

    # REAL + infinity
    if a.tag == TRTag.REAL:
        return b
    if b.tag == TRTag.REAL:
        return a

    raise RuntimeError(f"Unhandled Wheel case: {a.tag} + {b.tag}")


@wheel_only
def wheel_mul_pure(a: TRScalar, b: TRScalar) -> TRScalar:
    """
    Pure wheel multiplication - strict algebraic control.

    Implements wheel axioms including 0 × ∞ = ⊥.
    """
    # BOTTOM propagates
    if a.tag == TRTag.BOTTOM or b.tag == TRTag.BOTTOM:
        return bottom()

    # PHI maps to BOTTOM
    if a.tag == TRTag.PHI or b.tag == TRTag.PHI:
        return bottom()

    # Check for zeros
    a_is_zero = a.tag == TRTag.REAL and a.value == 0.0
    b_is_zero = b.tag == TRTag.REAL and b.value == 0.0

    # 0 × ∞ = ⊥ (wheel axiom)
    if a_is_zero and b.tag in (TRTag.PINF, TRTag.NINF):
        return bottom()
    if b_is_zero and a.tag in (TRTag.PINF, TRTag.NINF):
        return bottom()

    # 0 × finite = 0
    if a_is_zero or b_is_zero:
        return real(0.0)

    # REAL × REAL
    if a.tag == TRTag.REAL and b.tag == TRTag.REAL:
        result = a.value * b.value
        if abs(result) > 1e308:
            sign = 1 if (a.value > 0) == (b.value > 0) else -1
            return pinf() if sign > 0 else ninf()
        return real(result)

    # Sign determination
    def sign(x: TRScalar) -> int:
        if x.tag == TRTag.REAL:
            return 1 if x.value > 0 else -1 if x.value < 0 else 0
        elif x.tag == TRTag.PINF:
            return 1
        elif x.tag == TRTag.NINF:
            return -1
        else:
            return 0

    result_sign = sign(a) * sign(b)

    # Infinity multiplication
    if a.tag in (TRTag.PINF, TRTag.NINF) or b.tag in (TRTag.PINF, TRTag.NINF):
        return pinf() if result_sign > 0 else ninf()

    raise RuntimeError(f"Unhandled Wheel case: {a.tag} × {b.tag}")


# =============================================================================
# Compile-time switched operations
# =============================================================================

# Create mode-switched operations
add_switched = compile_time_switch(tr_add_pure, wheel_add_pure)
mul_switched = compile_time_switch(tr_mul_pure, wheel_mul_pure)


# =============================================================================
# Mode-aware operations with isolation
# =============================================================================


@isolated_operation
def safe_add(a: TRScalar, b: TRScalar) -> TRScalar:
    """
    Mode-aware addition with strict isolation.

    Automatically uses correct implementation based on mode,
    ensuring no mixing of semantics within the operation.
    """
    # Validate inputs are appropriate for current mode
    check_value_mode_compatibility(a)
    check_value_mode_compatibility(b)

    # Use switched implementation
    result = add_switched(a, b)

    # Validate output
    check_value_mode_compatibility(result)

    return result


@isolated_operation
def safe_mul(a: TRScalar, b: TRScalar) -> TRScalar:
    """
    Mode-aware multiplication with strict isolation.

    Automatically uses correct implementation based on mode,
    ensuring no mixing of semantics within the operation.
    """
    check_value_mode_compatibility(a)
    check_value_mode_compatibility(b)

    result = mul_switched(a, b)

    check_value_mode_compatibility(result)

    return result


# =============================================================================
# Demonstration of strict separation
# =============================================================================


def demonstrate_separation():
    """
    Demonstrate that TR and Wheel operations are strictly separated.
    """
    from .wheel_mode import ArithmeticMode, WheelModeConfig

    print("Demonstrating Strict Mode Separation")
    print("=" * 50)

    # Test TR mode
    print("\n1. Transreal Mode:")
    WheelModeConfig.set_mode(ArithmeticMode.TRANSREAL)

    # 0 × ∞ = Φ in TR
    result = safe_mul(real(0.0), pinf())
    print(f"   0 × ∞ = {result} (should be Φ)")
    assert result.tag == TRTag.PHI

    # +∞ + -∞ = Φ in TR
    result = safe_add(pinf(), ninf())
    print(f"   +∞ + -∞ = {result} (should be Φ)")
    assert result.tag == TRTag.PHI

    # Test Wheel mode
    print("\n2. Wheel Mode:")
    WheelModeConfig.set_mode(ArithmeticMode.WHEEL)

    # 0 × ∞ = ⊥ in Wheel
    result = safe_mul(real(0.0), pinf())
    print(f"   0 × ∞ = {result} (should be ⊥)")
    assert result.tag == TRTag.BOTTOM

    # +∞ + +∞ = ⊥ in Wheel (different from TR!)
    result = safe_add(pinf(), pinf())
    print(f"   +∞ + +∞ = {result} (should be ⊥)")
    assert result.tag == TRTag.BOTTOM

    # Test isolation
    print("\n3. Mode Isolation Test:")
    WheelModeConfig.set_mode(ArithmeticMode.TRANSREAL)

    # Try to use BOTTOM in TR mode (should fail)
    try:
        check_value_mode_compatibility(bottom())
        print("   ERROR: BOTTOM allowed in TR mode!")
    except Exception as e:
        print(f"   ✓ BOTTOM rejected in TR mode: {type(e).__name__}")

    print("\n" + "=" * 50)
    print("Separation demonstration complete!")


# =============================================================================
# Module-level enforcement
# =============================================================================


class TROnlyModule:
    """
    Example module that only works in TR mode.

    All methods automatically enforce TR mode.
    """

    def __init__(self):
        self.mode = "TR"

    @tr_only
    def compute(self, x: TRScalar, y: TRScalar) -> TRScalar:
        """Computation that requires TR semantics."""
        # This will always run in TR mode
        return safe_add(x, y)

    @tr_only
    def process(self, values: list[TRScalar]) -> TRScalar:
        """Process values using TR semantics."""
        result = real(0.0)
        for val in values:
            result = safe_add(result, val)
        return result


class WheelOnlyModule:
    """
    Example module that only works in Wheel mode.

    All methods automatically enforce Wheel mode.
    """

    def __init__(self):
        self.mode = "Wheel"

    @wheel_only
    def compute(self, x: TRScalar, y: TRScalar) -> TRScalar:
        """Computation that requires Wheel semantics."""
        # This will always run in Wheel mode
        return safe_mul(x, y)

    @wheel_only
    def verify_axiom(self) -> bool:
        """Verify a wheel axiom."""
        # 0 × ∞ should give ⊥
        result = safe_mul(real(0.0), pinf())
        return result.tag == TRTag.BOTTOM
