#!/usr/bin/env python3
"""Debug gradient tape clearing issue."""

from zeroproof.core import real
from zeroproof.autodiff import TRNode, gradient_tape
from zeroproof.autodiff.backward import backward_pass, topological_sort

print("=== Testing gradient tape clearing ===")

# Create computation
x = TRNode.parameter(real(2.0))

with gradient_tape() as tape:
    tape.watch(x)
    y = x * x + 3 * x

print(f"After computation: x._gradient = {x._gradient}")

# Simulate what gradient tape does
print(f"\n=== Simulating gradient tape ===")

# 1. Clear gradients
nodes = topological_sort(y)
for node in nodes:
    if node.requires_grad:
        node._gradient = None

print(f"After clearing: x._gradient = {x._gradient}")

# 2. Run backward pass
backward_pass(y)

print(f"After backward_pass: x._gradient = {x._gradient}")

# 3. Collect gradients
sources_list = [x]
gradients = []
for source in sources_list:
    print(f"Collecting from {source}:")
    print(f"  source.gradient = {source.gradient}")
    print(f"  source._gradient = {source._gradient}")
    
    if source.gradient is not None:
        grad_node = TRNode.constant(source.gradient)
        gradients.append(grad_node)
        print(f"  Added gradient node: {grad_node}")
    else:
        if source.requires_grad:
            grad_node = TRNode.constant(real(0.0))
            gradients.append(grad_node)
            print(f"  Added zero gradient node: {grad_node}")
        else:
            gradients.append(None)

print(f"Final gradients: {gradients}")
if gradients:
    print(f"gradients[0].value.value = {gradients[0].value.value}")

# Now test the actual gradient tape
print(f"\n=== Actual gradient tape ===")
grads_actual = tape.gradient(y, [x])
print(f"Actual result: {grads_actual[0].value.value}")
print(f"x._gradient after tape: {x._gradient}")
