#!/usr/bin/env python3
"""Debug tape context interference."""

from zeroproof.core import real
from zeroproof.autodiff import TRNode
from zeroproof.autodiff.backward import backward_pass

print("=== Testing backward_pass outside tape context ===")

# Create computation outside tape
x = TRNode.parameter(real(2.0))
y = x * x + 3 * x

print(f"Before backward_pass: x._gradient = {x._gradient}")

# Clear and run backward pass
x._gradient = None
backward_pass(y)

print(f"After backward_pass (outside tape): x._gradient = {x._gradient}")

print(f"\n=== Testing backward_pass inside tape context ===")

from zeroproof.autodiff import gradient_tape

# Create computation inside tape
with gradient_tape() as tape:
    x2 = TRNode.parameter(real(2.0))
    tape.watch(x2)
    y2 = x2 * x2 + 3 * x2

# Clear and run backward pass (outside tape context now)
x2._gradient = None
backward_pass(y2)

print(f"After backward_pass (computation from tape): x2._gradient = {x2._gradient}")

print(f"\n=== Testing if tape context affects backward_pass ===")

# Create computation inside tape, but call backward_pass while still inside
with gradient_tape() as tape:
    x3 = TRNode.parameter(real(2.0))
    tape.watch(x3)
    y3 = x3 * x3 + 3 * x3
    
    # Call backward_pass while inside tape context
    x3._gradient = None
    print(f"Calling backward_pass while inside tape context...")
    backward_pass(y3)
    print(f"x3._gradient = {x3._gradient}")

# Test if there's something special about the gradient tape's backward_pass call
print(f"\n=== Testing gradient tape's exact call ===")

with gradient_tape() as tape:
    x4 = TRNode.parameter(real(2.0))
    tape.watch(x4)
    y4 = x4 * x4 + 3 * x4

# Manually do what gradient tape does
from zeroproof.autodiff.backward import topological_sort
nodes = topological_sort(y4)
for node in nodes:
    if node.requires_grad:
        node._gradient = None

print(f"After clearing (like tape): x4._gradient = {x4._gradient}")

# Import and call exactly like tape does
from zeroproof.autodiff.backward import backward_pass as tape_backward_pass
tape_backward_pass(y4)

print(f"After tape-style backward_pass: x4._gradient = {x4._gradient}")
