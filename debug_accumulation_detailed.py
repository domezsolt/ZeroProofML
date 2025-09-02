#!/usr/bin/env python3
"""Debug gradient accumulation in detail."""

from zeroproof.core import real, tr_add
from zeroproof.autodiff import TRNode
from zeroproof.autodiff.backward import topological_sort, compute_input_gradients

print("=== Detailed accumulation debug ===")

# Create the failing computation: x^2 + 3*x
x = TRNode.parameter(real(2.0))
x_squared = x * x  # Should contribute gradient 4.0
three_x = 3 * x    # Should contribute gradient 3.0  
y = x_squared + three_x  # Total gradient should be 7.0

print(f"x = {x} (id={id(x)})")
print(f"x_squared = {x_squared}")
print(f"three_x = {three_x}")
print(f"y = {y}")

# Manual backward pass with detailed logging
grad_table = {}
grad_table[id(y)] = real(1.0)

nodes = topological_sort(y)
print(f"\nTopological order: {len(nodes)} nodes")
for i, node in enumerate(nodes):
    print(f"  {i}: {node} (id={id(node)})")

print(f"\n=== Manual backward pass ===")
for node in nodes:
    node_id = id(node)
    print(f"\n--- Processing {node} (id={node_id}) ---")
    
    if node_id not in grad_table:
        print(f"  No gradient, skipping")
        continue
    
    node_grad = grad_table[node_id]
    print(f"  node_grad = {node_grad}")
    
    if node._grad_info is None:
        print(f"  No grad_info")
        continue
    
    print(f"  op_type = {node._grad_info.op_type}")
    
    # Compute input gradients
    input_grads = compute_input_gradients(node, node_grad)
    print(f"  input_grads = {input_grads}")
    
    # Get inputs
    inputs = [ref() for ref in node._grad_info.inputs if ref() is not None]
    print(f"  inputs = {[f'{inp} (id={id(inp)})' for inp in inputs]}")
    
    # Accumulate gradients
    for i, (inp, inp_grad) in enumerate(zip(inputs, input_grads)):
        if inp_grad is not None:
            inp_id = id(inp)
            print(f"    Processing input[{i}]: {inp} (id={inp_id})")
            print(f"    inp_grad = {inp_grad}")
            
            if inp_id not in grad_table:
                grad_table[inp_id] = inp_grad
                print(f"    NEW: grad_table[{inp_id}] = {inp_grad}")
            else:
                old_grad = grad_table[inp_id]
                new_grad = tr_add(old_grad, inp_grad)
                grad_table[inp_id] = new_grad
                print(f"    ACCUMULATE: {old_grad} + {inp_grad} = {new_grad}")

print(f"\nFinal grad_table:")
for node_id, grad in grad_table.items():
    print(f"  {node_id}: {grad}")

print(f"\nFinal x gradient:")
x_id = id(x)
print(f"x_id = {x_id}")
print(f"grad_table[x_id] = {grad_table.get(x_id, 'NOT_FOUND')}")

# Now set the gradient
if x_id in grad_table:
    x._gradient = grad_table[x_id]
    print(f"Set x._gradient = {x._gradient}")
else:
    print(f"ERROR: x not found in grad_table!")
