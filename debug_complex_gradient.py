#!/usr/bin/env python3
"""Debug complex gradient computation."""

from zeroproof.core import real
from zeroproof.autodiff import TRNode, gradient_tape

print("=== Debugging complex gradient ===")

with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    
    # Build expression step by step
    print(f"x = {x}")
    
    # x^2
    x_squared = x * x
    print(f"x^2 = {x_squared}")
    
    # 3*x
    three_x = 3 * x
    print(f"3*x = {three_x}")
    
    # x^2 + 3*x
    sum1 = x_squared + three_x
    print(f"x^2 + 3*x = {sum1}")
    
    # x^2 + 3*x + 1
    y = sum1 + 1
    print(f"y = x^2 + 3*x + 1 = {y}")

print(f"\nComputing gradient...")
grads = tape.gradient(y, [x])
print(f"Gradient: {grads[0].value.value}")
print(f"Expected: 7.0")

# Check if x has the right gradient
print(f"x._gradient = {x._gradient}")

# Let's also test just x^2 + 3*x (without the +1)
print(f"\n=== Testing without constant term ===")
with gradient_tape() as tape2:
    x2 = TRNode.parameter(real(2.0))
    tape2.watch(x2)
    y2 = x2 * x2 + 3 * x2  # No +1

grads2 = tape2.gradient(y2, [x2])
print(f"Gradient of x^2 + 3*x: {grads2[0].value.value}")
print(f"Expected: 7.0")

# Test each term separately
print(f"\n=== Testing terms separately ===")

# Just x^2
with gradient_tape() as tape3:
    x3 = TRNode.parameter(real(2.0))
    tape3.watch(x3)
    y3 = x3 * x3

grads3 = tape3.gradient(y3, [x3])
print(f"Gradient of x^2: {grads3[0].value.value}")
print(f"Expected: 4.0")

# Just 3*x  
with gradient_tape() as tape4:
    x4 = TRNode.parameter(real(2.0))
    tape4.watch(x4)
    y4 = 3 * x4

grads4 = tape4.gradient(y4, [x4])
print(f"Gradient of 3*x: {grads4[0].value.value}")
print(f"Expected: 3.0")
