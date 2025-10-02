"""
Demonstration of wheel mode in ZeroProof.

This example shows the differences between standard transreal arithmetic
and wheel mode, which provides stricter algebraic control.
"""

import zeroproof as zp
from zeroproof import bottom, ninf, phi, pinf, real


def compare_modes():
    """Compare operations between transreal and wheel modes."""
    print("Wheel Mode vs Transreal Mode Comparison")
    print("=" * 50)

    # Test cases that differ between modes
    test_cases = [
        ("0 × ∞", lambda: zp.tr_mul(real(0.0), pinf())),
        ("0 × (-∞)", lambda: zp.tr_mul(real(0.0), ninf())),
        ("∞ + ∞", lambda: zp.tr_add(pinf(), pinf())),
        ("(-∞) + (-∞)", lambda: zp.tr_add(ninf(), ninf())),
        ("∞ + (-∞)", lambda: zp.tr_add(pinf(), ninf())),
        ("∞ / ∞", lambda: zp.tr_div(pinf(), pinf())),
        ("(-∞) / ∞", lambda: zp.tr_div(ninf(), pinf())),
    ]

    # Header
    print(f"{'Operation':<15} {'Transreal':<15} {'Wheel':<15}")
    print("-" * 45)

    for name, operation in test_cases:
        # Transreal mode result
        zp.use_transreal()
        tr_result = operation()
        tr_str = str(tr_result)

        # Wheel mode result
        zp.use_wheel()
        wheel_result = operation()
        wheel_str = str(wheel_result)

        print(f"{name:<15} {tr_str:<15} {wheel_str:<15}")

    # Reset to transreal
    zp.use_transreal()


def demonstrate_propagation():
    """Show how BOTTOM propagates in wheel mode."""
    print("\n\nBOTTOM Propagation in Wheel Mode")
    print("=" * 50)

    with zp.wheel_mode():
        # Create a bottom value from 0 × ∞
        print("Creating BOTTOM: 0 × ∞ =", zp.tr_mul(real(0.0), pinf()))

        # Show propagation
        b = zp.tr_mul(real(0.0), pinf())
        print(f"\n⊥ + 5 = {zp.tr_add(b, real(5.0))}")
        print(f"⊥ × 10 = {zp.tr_mul(b, real(10.0))}")
        print(f"1 / ⊥ = {zp.tr_div(real(1.0), b)}")
        print(f"√⊥ = {zp.tr_sqrt(b)}")
        print(f"log(⊥) = {zp.tr_log(b)}")
        print(f"|⊥| = {zp.tr_abs(b)}")


def algebraic_example():
    """Show algebraic implications of wheel mode."""
    print("\n\nAlgebraic Properties")
    print("=" * 50)

    print("In transreal mode, some 'uncomfortable simplifications' are allowed:")
    print("Example: (x + ∞) - ∞ = x when x is finite")

    x = real(5.0)
    result_tr = zp.tr_add(zp.tr_add(x, pinf()), ninf())
    print(f"Transreal: (5 + ∞) + (-∞) = {result_tr}")

    print("\nIn wheel mode, this becomes bottom:")
    with zp.wheel_mode():
        # First, ∞ + ∞ type operations give ⊥
        inf_sum = zp.tr_add(pinf(), pinf())
        print(f"Wheel: ∞ + ∞ = {inf_sum}")

        # So the expression becomes undefined
        result_wheel = zp.tr_add(zp.tr_add(x, pinf()), ninf())
        print(f"Wheel: (5 + ∞) + (-∞) = {result_wheel}")


def practical_example():
    """Show a practical scenario where wheel mode might be useful."""
    print("\n\nPractical Example: Detecting Algebraic Issues")
    print("=" * 50)

    def unstable_function(x):
        """A function that has algebraic issues at x=0."""
        # f(x) = x / x  (should be 1, but what about x=0?)
        return zp.tr_div(x, x)

    def potentially_undefined(x):
        """A function with potential 0×∞ form."""
        # g(x) = x × (1/x)
        return zp.tr_mul(x, zp.tr_div(real(1.0), x))

    test_values = [real(2.0), real(0.0), pinf(), ninf()]

    print("Function: f(x) = x / x")
    print(f"{'x':<10} {'Transreal':<15} {'Wheel':<15}")
    print("-" * 40)

    for x_val in test_values:
        # Transreal result
        zp.use_transreal()
        tr_result = unstable_function(x_val)

        # Wheel result
        zp.use_wheel()
        wheel_result = unstable_function(x_val)

        print(f"{str(x_val):<10} {str(tr_result):<15} {str(wheel_result):<15}")

    print("\n\nFunction: g(x) = x × (1/x)")
    print(f"{'x':<10} {'Transreal':<15} {'Wheel':<15}")
    print("-" * 40)

    for x_val in test_values:
        # Transreal result
        zp.use_transreal()
        tr_result = potentially_undefined(x_val)

        # Wheel result
        zp.use_wheel()
        wheel_result = potentially_undefined(x_val)

        print(f"{str(x_val):<10} {str(tr_result):<15} {str(wheel_result):<15}")

    # Reset to transreal
    zp.use_transreal()


def mixed_computation():
    """Show how to work with mixed computations."""
    print("\n\nMixed Computations")
    print("=" * 50)

    print("Computing in transreal, then checking in wheel mode:")

    # Do main computation in transreal
    zp.use_transreal()
    a = zp.tr_mul(real(0.0), pinf())  # Gets PHI
    b = zp.tr_add(pinf(), ninf())  # Gets PHI
    result_tr = zp.tr_add(a, b)  # PHI + PHI = PHI

    print(f"Transreal: (0×∞) + (∞-∞) = {a} + {b} = {result_tr}")

    # Check what would happen in wheel mode
    with zp.wheel_mode():
        a_wheel = zp.tr_mul(real(0.0), pinf())  # Gets BOTTOM
        b_wheel = zp.tr_add(pinf(), ninf())  # Gets BOTTOM
        result_wheel = zp.tr_add(a_wheel, b_wheel)  # BOTTOM + BOTTOM = BOTTOM

        print(f"Wheel: (0×∞) + (∞-∞) = {a_wheel} + {b_wheel} = {result_wheel}")

    print("\nThis shows wheel mode is stricter about algebraic validity.")


def main():
    """Run all demonstrations."""
    print("ZeroProof Wheel Mode Demonstration")
    print("=" * 70)
    print()

    # Basic comparison
    compare_modes()

    # Show propagation
    demonstrate_propagation()

    # Algebraic properties
    algebraic_example()

    # Practical examples
    practical_example()

    # Mixed computations
    mixed_computation()

    print("\n" + "=" * 70)
    print("Summary:")
    print("- Transreal mode: More permissive, keeps computations flowing")
    print("- Wheel mode: Stricter algebra, catches potential issues")
    print("- Use transreal for general computation")
    print("- Use wheel for algebraic verification and stricter control")


if __name__ == "__main__":
    main()
