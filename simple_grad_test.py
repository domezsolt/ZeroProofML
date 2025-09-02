#!/usr/bin/env python3
"""Simple gradient test to debug the issue."""

from zeroproof.core import real
from zeroproof.autodiff import TRNode, gradient_tape

print("Testing basic gradient computation...")

try:
    with gradient_tape() as tape:
        x = TRNode.parameter(real(2.0))
        tape.watch(x)
        
        # y = x^2 + 3x + 1
        y = x * x + 3 * x + 1
        print(f"Forward: y = {y.value.value}")
    
    grads = tape.gradient(y, [x])
    print(f"Gradient: {grads[0].value.value}")
    print(f"Expected: 7.0")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
