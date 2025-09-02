#!/usr/bin/env python3
"""Debug gradient computation step by step."""

from zeroproof.core import real
from zeroproof.autodiff import TRNode, gradient_tape

print("=== Step-by-step gradient debug ===")

with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    print(f"1. x = {x}, requires_grad = {x.requires_grad}")
    tape.watch(x)
    
    # Break down y = x^2 + 3x + 1
    x_squared = x * x
    print(f"2. x^2 = {x_squared}, requires_grad = {x_squared.requires_grad}")
    print(f"   x_squared._grad_info = {x_squared._grad_info}")
    
    three_x = 3 * x
    print(f"3. 3*x = {three_x}, requires_grad = {three_x.requires_grad}")
    print(f"   three_x._grad_info = {three_x._grad_info}")
    
    term1 = x_squared + three_x
    print(f"4. x^2 + 3x = {term1}, requires_grad = {term1.requires_grad}")
    print(f"   term1._grad_info = {term1._grad_info}")
    
    y = term1 + 1
    print(f"5. y = x^2 + 3x + 1 = {y}, requires_grad = {y.requires_grad}")
    print(f"   y._grad_info = {y._grad_info}")

print(f"\n6. Computing gradients...")
grads = tape.gradient(y, [x])
print(f"7. grads = {grads}")
print(f"8. grads[0] = {grads[0]}")
print(f"9. grads[0].value.value = {grads[0].value.value}")

# Check if x has gradient
print(f"\n10. x._gradient = {x._gradient}")

# Let's also check the topological sort
from zeroproof.autodiff.backward import topological_sort
nodes = topological_sort(y)
print(f"\n11. Topological order:")
for i, node in enumerate(nodes):
    print(f"    {i}: {node} (requires_grad={node.requires_grad})")
    if hasattr(node, '_grad_info') and node._grad_info:
        print(f"        _grad_info.op_type = {node._grad_info.op_type}")
        inputs = [inp() for inp in node._grad_info.inputs if inp()]
        print(f"        inputs = {inputs}")
