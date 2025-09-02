#!/usr/bin/env python3
"""Debug backward pass implementation."""

from zeroproof.core import real
from zeroproof.autodiff import TRNode
from zeroproof.autodiff.backward import backward_pass, topological_sort

print("=== Debugging backward pass ===")

# Create computation
x = TRNode.parameter(real(2.0))
y = x * x + 3 * x + 1

print(f"x = {x}, id = {id(x)}")
print(f"y = {y}, id = {id(y)}")

# Get topological order
nodes = topological_sort(y)
print(f"\nTopological order:")
for i, node in enumerate(nodes):
    print(f"  {i}: {node} (id={id(node)}, requires_grad={node.requires_grad})")

# Clear gradients
for node in nodes:
    if node.requires_grad:
        node._gradient = None

print(f"\nBefore backward pass:")
print(f"x._gradient = {x._gradient}")

# Run backward pass with debug
print(f"\nRunning backward_pass(y)...")

# Let's manually trace through backward_pass
grad_output = real(1.0)
print(f"grad_output = {grad_output}")

# Initialize gradients
grad_table = {}
grad_table[id(y)] = grad_output
print(f"grad_table[{id(y)}] = {grad_output}")

# Process nodes in topological order
for node in nodes:
    node_id = id(node)
    print(f"\nProcessing node {node} (id={node_id})")
    
    # Skip if no gradient for this node
    if node_id not in grad_table:
        print(f"  No gradient in table, skipping")
        continue
    
    node_grad = grad_table[node_id]
    print(f"  node_grad = {node_grad}")
    
    # Accumulate gradient for this node if it requires grad
    if node.requires_grad:
        print(f"  Node requires grad, accumulating...")
        if node._gradient is None:
            node._gradient = node_grad
            print(f"    Set node._gradient = {node._gradient}")
        else:
            from zeroproof.core import tr_add
            node._gradient = tr_add(node._gradient, node_grad)
            print(f"    Added to node._gradient = {node._gradient}")
    
    # If this is a leaf node or has no gradient info, continue
    if node._grad_info is None:
        print(f"  No grad_info, continuing")
        continue
    
    print(f"  Has grad_info: {node._grad_info.op_type}")
    
    # Check gradient mode
    from zeroproof.autodiff.grad_mode import GradientModeConfig, GradientMode
    gradient_mode = GradientModeConfig.get_mode()
    print(f"  Gradient mode: {gradient_mode}")
    
    # MASK-REAL RULE: If forward value is non-REAL, all input gradients are zero
    if gradient_mode == GradientMode.MASK_REAL and node.tag != real(0.0).tag:
        print(f"  Node tag {node.tag} is non-REAL, applying MASK-REAL")
        continue
    
    print(f"  Computing input gradients...")
    
print(f"\nFinal x._gradient = {x._gradient}")
