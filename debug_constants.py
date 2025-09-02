#!/usr/bin/env python3
"""Debug constant handling in gradients."""

from zeroproof.core import real
from zeroproof.autodiff import TRNode, gradient_tape

print("=== Testing constant handling ===")

# Test 1: x * constant
print("Test 1: x * 3")
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    y = x * 3  # This should work

grads = tape.gradient(y, [x])
print(f"Result: {grads[0].value.value}, expected: 3.0")

# Test 2: constant * x  
print("\nTest 2: 3 * x")
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    y = 3 * x  # This should work

grads = tape.gradient(y, [x])
print(f"Result: {grads[0].value.value}, expected: 3.0")

# Test 3: x^2 + constant*x
print("\nTest 3: x^2 + 3*x (step by step)")
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    
    term1 = x * x
    print(f"  term1 = x^2 = {term1}")
    
    term2 = 3 * x  
    print(f"  term2 = 3*x = {term2}")
    
    y = term1 + term2
    print(f"  y = term1 + term2 = {y}")

grads = tape.gradient(y, [x])
print(f"Result: {grads[0].value.value}, expected: 7.0")

# Test 4: Using explicit TRNode.constant
print("\nTest 4: x^2 + TRNode.constant(3)*x")
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    
    term1 = x * x
    three = TRNode.constant(real(3.0))
    term2 = three * x
    y = term1 + term2

grads = tape.gradient(y, [x])
print(f"Result: {grads[0].value.value}, expected: 7.0")

# Check x._gradient directly
print(f"x._gradient = {x._gradient}")

# Test 5: Different order
print("\nTest 5: 3*x + x^2 (different order)")
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    y = 3 * x + x * x

grads = tape.gradient(y, [x])
print(f"Result: {grads[0].value.value}, expected: 7.0")
