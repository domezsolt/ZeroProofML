#!/usr/bin/env python3
"""Verify gradient fix is working."""

from zeroproof.core import real
from zeroproof.autodiff import TRNode, gradient_tape

print("=== Verifying gradient fix ===")

# Test 1: Simple gradient tape (this was working)
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    y = x * x
    
grads = tape.gradient(y, [x])
print(f"Test 1 - Simple: {grads[0].value.value}, expected: 4.0")

# Test 2: Complex expression (from failing test)
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    # y = x^2 + 3x + 1, dy/dx = 2x + 3 = 7 at x=2
    y = x * x + 3 * x + 1
    
grads = tape.gradient(y, [x])
print(f"Test 2 - Complex: {grads[0].value.value}, expected: 7.0")

# Test 3: tr_grad (this was failing)
from zeroproof.autodiff import tr_grad

def f(x):
    return x * x + 2 * x + 1

grad_f = tr_grad(f)
x = TRNode.parameter(real(3.0))
df_dx = grad_f(x)
print(f"Test 3 - tr_grad: {df_dx.value.value}, expected: 8.0")

print(f"\nAll tests working: {grads[0].value.value == 7.0 and df_dx.value.value == 8.0}")
