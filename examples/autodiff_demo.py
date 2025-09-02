"""
Demonstration of transreal autodifferentiation with Mask-REAL rule.

This example shows how gradients flow through transreal computations
and how the Mask-REAL rule prevents gradient explosions at singularities.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import zeroproof as zp
from zeroproof.autodiff import (
    TRNode, gradient_tape, tr_grad, tr_value_and_grad,
    tr_log, tr_sqrt, tr_pow_int
)


def demonstrate_basic_autodiff():
    """Show basic automatic differentiation."""
    print("=== Basic Autodiff ===\n")
    
    # Create a parameter (variable we want gradients for)
    x = TRNode.parameter(zp.real(3.0), name="x")
    print(f"x = {x}")
    
    # Compute a function with gradient tape
    with gradient_tape() as tape:
        tape.watch(x)
        
        # y = x^2 + 2x + 1
        y = x * x + 2 * x + 1
        print(f"y = x² + 2x + 1 = {y}")
    
    # Compute gradient dy/dx
    grads = tape.gradient(y, [x])
    grad_x = grads[0]
    
    print(f"dy/dx = 2x + 2 = {grad_x}")
    print(f"At x=3: dy/dx = {grad_x.value.value}")


def demonstrate_mask_real_rule():
    """Show how Mask-REAL prevents gradient explosion."""
    print("\n=== Mask-REAL Rule ===\n")
    
    # Example 1: Division by zero
    print("Example 1: Division by zero")
    with gradient_tape() as tape:
        x = TRNode.parameter(zp.real(0.0), name="x")
        tape.watch(x)
        
        # y = 1/x → +∞ at x=0
        y = TRNode.constant(zp.real(1.0)) / x
        print(f"x = {x}")
        print(f"y = 1/x = {y} (tag: {y.tag})")
    
    grads = tape.gradient(y, [x])
    print(f"dy/dx = {grads[0]} (Mask-REAL sets to 0)")
    
    # Example 2: Indeterminate form
    print("\nExample 2: Indeterminate form 0/0")
    with gradient_tape() as tape:
        x = TRNode.parameter(zp.real(0.0), name="x")
        tape.watch(x)
        
        # y = x/x → Φ at x=0
        y = x / x
        print(f"y = x/x = {y} (tag: {y.tag})")
    
    grads = tape.gradient(y, [x])
    print(f"dy/dx = {grads[0]} (Mask-REAL sets to 0)")


def demonstrate_gradient_functions():
    """Show high-level gradient computation functions."""
    print("\n=== Gradient Functions ===\n")
    
    # Define a function
    def f(x):
        return x * x * x - 2 * x * x + x + 5
    
    print("f(x) = x³ - 2x² + x + 5")
    print("f'(x) = 3x² - 4x + 1")
    
    # Create gradient function
    grad_f = tr_grad(f)
    
    # Evaluate at x = 2
    x = TRNode.parameter(zp.real(2.0))
    df_dx = grad_f(x)
    
    print(f"\nAt x=2:")
    if df_dx is not None:
        print(f"f'(2) = 3(4) - 4(2) + 1 = {df_dx.value.value}")
    else:
        print("f'(2) = None (gradient not computed)")
    
    # Value and gradient together
    value_and_grad_f = tr_value_and_grad(f)
    val, grad = value_and_grad_f(x)
    
    print(f"f(2) = {val.value.value}")
    if grad is not None:
        print(f"f'(2) = {grad.value.value}")
    else:
        print("f'(2) = None")


def demonstrate_domain_aware_gradients():
    """Show gradients of domain-aware operations."""
    print("\n=== Domain-Aware Gradients ===\n")
    
    # Example 1: Valid domain
    print("Example 1: log(x) at x=2")
    with gradient_tape() as tape:
        x = TRNode.parameter(zp.real(2.0))
        tape.watch(x)
        y = tr_log(x)
    
    grads = tape.gradient(y, [x])
    print(f"d/dx[log(x)] = 1/x = {grads[0].value.value}")
    
    # Example 2: Invalid domain
    print("\nExample 2: log(x) at x=-1")
    with gradient_tape() as tape:
        x = TRNode.parameter(zp.real(-1.0))
        tape.watch(x)
        y = tr_log(x)  # log(-1) = Φ
        print(f"log(-1) = {y} (tag: {y.tag})")
    
    grads = tape.gradient(y, [x])
    print(f"d/dx[log(x)] at x=-1 = {grads[0]} (Mask-REAL sets to 0)")
    
    # Example 3: Square root
    print("\nExample 3: sqrt(x) at x=4")
    with gradient_tape() as tape:
        x = TRNode.parameter(zp.real(4.0))
        tape.watch(x)
        y = tr_sqrt(x)
    
    grads = tape.gradient(y, [x])
    print(f"d/dx[sqrt(x)] = 1/(2*sqrt(x)) = {grads[0].value.value}")


def demonstrate_rational_function():
    """Show gradient of a rational function near poles."""
    print("\n=== Rational Function Gradients ===\n")
    
    def rational(x):
        # f(x) = (x + 1) / (x - 2)
        numerator = x + 1
        denominator = x - 2
        return numerator / denominator
    
    print("f(x) = (x + 1) / (x - 2)")
    print("This has a pole at x = 2\n")
    
    # Test at various points
    test_points = [0.0, 1.0, 1.9, 2.0, 2.1, 3.0]
    
    for x_val in test_points:
        x = TRNode.parameter(zp.real(x_val))
        
        # Compute value and gradient
        val_grad_f = tr_value_and_grad(rational)
        try:
            val, grad = val_grad_f(x)
            print(f"x = {x_val}:")
            print(f"  f(x) = {val} (tag: {val.tag})")
            print(f"  f'(x) = {grad}")
        except Exception as e:
            print(f"x = {x_val}: Error - {e}")


def demonstrate_complex_gradient_flow():
    """Show gradient flow through complex expressions."""
    print("\n=== Complex Gradient Flow ===\n")
    
    print("Computing gradient of: f(x,y) = log(x² + y²) / sqrt(x + y)")
    
    def complex_function(x, y):
        # f(x,y) = log(x² + y²) / sqrt(x + y)
        x_squared = x * x
        y_squared = y * y
        sum_squares = x_squared + y_squared
        log_part = tr_log(sum_squares)
        
        sum_xy = x + y
        sqrt_part = tr_sqrt(sum_xy)
        
        return log_part / sqrt_part
    
    # Compute gradients with respect to both inputs
    grad_f = tr_grad(complex_function, argnums=[0, 1])
    
    # Test at x=3, y=1
    x = TRNode.parameter(zp.real(3.0))
    y = TRNode.parameter(zp.real(1.0))
    
    grads = grad_f(x, y)
    
    print(f"\nAt x=3, y=1:")
    print(f"∂f/∂x = {grads[0].value.value:.6f}")
    print(f"∂f/∂y = {grads[1].value.value:.6f}")
    
    # Test near problematic point (x + y = 0)
    print("\nNear problematic point where x + y ≈ 0:")
    x2 = TRNode.parameter(zp.real(1.0))
    y2 = TRNode.parameter(zp.real(-0.999))
    
    grads2 = grad_f(x2, y2)
    print(f"At x=1, y=-0.999:")
    print(f"∂f/∂x = {grads2[0].value.value:.2f}")
    print(f"∂f/∂y = {grads2[1].value.value:.2f}")
    
    # At the singularity
    print("\nAt singularity where x + y = 0:")
    x3 = TRNode.parameter(zp.real(1.0))
    y3 = TRNode.parameter(zp.real(-1.0))
    
    with gradient_tape() as tape:
        tape.watch(x3)
        tape.watch(y3)
        f_val = complex_function(x3, y3)
    
    print(f"f(1, -1) = {f_val} (tag: {f_val.tag})")
    grads3 = tape.gradient(f_val, [x3, y3])
    print(f"∂f/∂x = {grads3[0]} (Mask-REAL)")
    print(f"∂f/∂y = {grads3[1]} (Mask-REAL)")


if __name__ == "__main__":
    print("ZeroProof: Transreal Autodifferentiation Demo")
    print("=============================================\n")
    
    demonstrate_basic_autodiff()
    demonstrate_mask_real_rule()
    demonstrate_gradient_functions()
    demonstrate_domain_aware_gradients()
    demonstrate_rational_function()
    demonstrate_complex_gradient_flow()
    
    print("\n=============================================")
    print("Mask-REAL rule ensures stable gradients even at singularities!")
    print("No gradient explosions, no NaN propagation.")
