"""Basic usage example of ZeroProof library.

This example demonstrates the core concepts of transreal arithmetic
and how it handles singularities gracefully.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import zeroproof as zp


def demonstrate_transreal_arithmetic():
    """Show basic transreal arithmetic operations."""
    print("=== Basic Transreal Arithmetic ===\n")
    
    # Create transreal scalars
    x = zp.real(3.0)
    y = zp.real(0.0)
    z = zp.real(-2.0)
    
    # Division by zero - returns infinity instead of error
    result1 = x / y  # Returns PINF
    print(f"3.0 / 0.0 = {result1}")
    
    result2 = z / y  # Returns NINF
    print(f"-2.0 / 0.0 = {result2}")
    
    # Undefined forms return PHI
    result3 = y / y  # 0/0 returns PHI
    print(f"0.0 / 0.0 = {result3}")
    
    # Infinity arithmetic
    inf = zp.pinf()
    result4 = inf - inf  # ∞ - ∞ returns PHI
    print(f"∞ - ∞ = {result4}")
    
    result5 = y * inf  # 0 × ∞ returns PHI
    print(f"0 × ∞ = {result5}")
    
    # Regular arithmetic works normally
    result6 = x + z  # 3.0 + (-2.0) = 1.0
    print(f"3.0 + (-2.0) = {result6}")


def demonstrate_special_operations():
    """Show special operations and edge cases."""
    print("\n=== Special Operations ===\n")
    
    # Logarithm with domain checking
    pos = zp.real(2.0)
    neg = zp.real(-1.0)
    zero = zp.real(0.0)
    
    print(f"log(2.0) = {zp.tr_log(pos)}")
    print(f"log(-1.0) = {zp.tr_log(neg)}")  # Returns PHI
    print(f"log(0.0) = {zp.tr_log(zero)}")  # Returns PHI
    print(f"log(+∞) = {zp.tr_log(zp.pinf())}")  # Returns PINF
    
    # Square root with domain checking
    print(f"\n√4.0 = {zp.tr_sqrt(zp.real(4.0))}")
    print(f"√(-1.0) = {zp.tr_sqrt(neg)}")  # Returns PHI
    print(f"√(+∞) = {zp.tr_sqrt(zp.pinf())}")  # Returns PINF
    
    # Powers with special cases
    print(f"\n0^0 = {zp.tr_pow_int(zero, 0)}")  # Returns PHI
    print(f"(+∞)^0 = {zp.tr_pow_int(zp.pinf(), 0)}")  # Returns PHI
    print(f"2^3 = {zp.tr_pow_int(zp.real(2.0), 3)}")
    print(f"2^(-2) = {zp.tr_pow_int(zp.real(2.0), -2)}")


def demonstrate_operator_overloading():
    """Show natural Python operator usage."""
    print("\n=== Operator Overloading ===\n")
    
    try:
        x = zp.real(5.0)
        y = zp.real(2.0)
        
        # Natural operators work as expected
        print(f"x + y = {x + y}")
        print(f"x - y = {x - y}")
        print(f"x * y = {x * y}")
        print(f"x / y = {x / y}")
        print(f"-x = {-x}")
        print(f"|x| = {abs(x)}")
        print(f"x^2 = {x**2}")
        
        # Mixed operations with Python numbers
        print(f"\nx + 3 = {x + 3}")
        print(f"10 / x = {10 / x}")
        
        # Comparisons and type checking
        print(f"\nx is REAL: {x.is_real()}")
        print(f"x is finite: {x.is_finite()}")
        print(f"(+∞) is infinite: {zp.pinf().is_infinite()}")
    except Exception as e:
        print(f"Error in operator overloading: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_ieee_bridge():
    """Show IEEE-754 bridge functionality."""
    print("\n=== IEEE ↔ TR Bridge ===\n")
    
    try:
        # Convert from IEEE floats
        tr_inf = zp.from_ieee(float('inf'))
        tr_nan = zp.from_ieee(float('nan'))
        tr_finite = zp.from_ieee(3.14)
        
        print(f"from_ieee(inf) = {tr_inf} (tag: {tr_inf.tag})")
        print(f"from_ieee(nan) = {tr_nan} (tag: {tr_nan.tag})")
        print(f"from_ieee(3.14) = {tr_finite} (tag: {tr_finite.tag})")
        
        # Convert to IEEE floats
        ieee_val1 = zp.to_ieee(zp.phi())
        ieee_val2 = zp.to_ieee(zp.pinf())
        ieee_val3 = zp.to_ieee(zp.real(2.718))
        
        print(f"\nto_ieee(Φ) = {ieee_val1}")
        print(f"to_ieee(+∞) = {ieee_val2}")
        print(f"to_ieee(2.718) = {ieee_val3}")
        
        # Round-trip preservation
        import math
        original = 42.0
        there_and_back = zp.to_ieee(zp.from_ieee(original))
        print(f"\nRound-trip: {original} → TR → {there_and_back}")
    except Exception as e:
        print(f"Error in IEEE bridge: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("ZeroProof: Transreal Arithmetic Demo")
    print("=====================================\n")
    
    demonstrate_transreal_arithmetic()
    demonstrate_special_operations()
    demonstrate_operator_overloading()
    demonstrate_ieee_bridge()
    
    print("\n=====================================")
    print("With ZeroProof, singularities are no longer errors!")
    print("All operations are total and deterministic.")
