#!/usr/bin/env python3
"""Debug gradient accumulation issue."""

from zeroproof.core import real
from zeroproof.autodiff import TRNode, gradient_tape

print("=== Testing gradient accumulation patterns ===")

# Test 1: x^2 + 3*x (fails)
print("Test 1: x^2 + 3*x")
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    y = x * x + 3 * x

grads = tape.gradient(y, [x])
print(f"Result: {grads[0].value.value}, expected: 7.0")

# Test 2: x^2 + 3*x + 1 (works)
print("\nTest 2: x^2 + 3*x + 1")
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    y = x * x + 3 * x + 1

grads = tape.gradient(y, [x])
print(f"Result: {grads[0].value.value}, expected: 7.0")

# Test 3: x^2 + 3*x + 0 (should work like +1)
print("\nTest 3: x^2 + 3*x + 0")
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    y = x * x + 3 * x + 0

grads = tape.gradient(y, [x])
print(f"Result: {grads[0].value.value}, expected: 7.0")

# Test 4: (x^2 + 3*x) + 0 (explicit parentheses)
print("\nTest 4: (x^2 + 3*x) + 0")
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    temp = x * x + 3 * x
    y = temp + 0

grads = tape.gradient(y, [x])
print(f"Result: {grads[0].value.value}, expected: 7.0")

# Test 5: x + x (simple accumulation)
print("\nTest 5: x + x")
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    y = x + x

grads = tape.gradient(y, [x])
print(f"Result: {grads[0].value.value}, expected: 2.0")

# Test 6: 2*x (equivalent to x + x)
print("\nTest 6: 2*x")
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    y = 2 * x

grads = tape.gradient(y, [x])
print(f"Result: {grads[0].value.value}, expected: 2.0")
