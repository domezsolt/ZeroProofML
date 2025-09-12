# Transreal Autodifferentiation with Mask-REAL Rule

> See also: Topic 3 “Autodiff Modes” (`docs/topics/03_autodiff_modes.md`) for Mask‑REAL vs Saturating vs Hybrid and practical setup.

## Overview

The ZeroProof library implements automatic differentiation (autodiff) for transreal arithmetic with the **Mask-REAL rule**. This rule ensures that gradients remain well-defined and stable even when computations produce non-REAL values (infinities or indeterminate forms).

## The Mask-REAL Rule

### Definition

The Mask-REAL rule states:

> **If the forward pass of an operation produces a non-REAL tag (PINF, NINF, or PHI), then all gradients with respect to that operation's inputs and parameters are set to zero.**

This rule prevents gradient explosions and undefined behavior when dealing with singularities, while allowing the rest of the computation graph to continue normally.

### Mathematical Justification

1. **Singularities indicate non-differentiable points**: When a function produces ±∞ or Φ, it typically means we've hit a point where the function is not differentiable in the classical sense.

2. **Zero gradients as subgradients**: Setting gradients to zero can be interpreted as choosing the zero element from the subdifferential at non-differentiable points.

3. **Prevents NaN propagation**: Traditional autodiff would propagate NaN through the graph, making all gradients unusable. Mask-REAL contains the issue locally.

## Implementation Details

### Node Structure

Each `TRNode` in the computation graph contains:
- A transreal value (value + tag)
- Gradient tracking information
- References to input nodes for backpropagation

```python
class TRNode:
    def __init__(self, value: TRScalar, requires_grad: bool = False):
        self._value = value  # TRScalar with tag
        self._requires_grad = requires_grad
        self._gradient = None
        # ... gradient info for backprop
```

### Forward Pass

During the forward pass, operations compute both the value and tag:

```python
# Example: Division
x = TRNode.parameter(real(1.0))
y = TRNode.parameter(real(0.0))
z = x / y  # Forward: z.value = PINF, z.tag = TRTag.PINF
```

### Backward Pass

The backward pass uses topological ordering and applies Mask-REAL:

1. **Topological Sort**: Nodes are processed in reverse topological order to ensure all downstream gradients are computed before upstream ones.

2. **Gradient Computation**:
   - If node.tag == REAL: Compute gradients using standard calculus rules
   - If node.tag ∈ {PINF, NINF, PHI}: Set all input gradients to zero

3. **Gradient Accumulation**: Gradients are accumulated for nodes used multiple times

### Example: Division by Zero

```python
with gradient_tape() as tape:
    x = TRNode.parameter(real(0.0))
    tape.watch(x)
    
    # y = 1/x → +∞ at x=0
    y = TRNode.constant(real(1.0)) / x

# Standard calculus would give dy/dx = -1/x² → -∞
# But Mask-REAL sets dy/dx = 0 because y.tag = PINF
grads = tape.gradient(y, [x])
assert grads[0].value.value == 0.0
```

## Gradient Propagation Rules

### Binary Operations

For `z = op(x, y)`:

| Operation | ∂z/∂x (if z is REAL) | ∂z/∂y (if z is REAL) |
|-----------|---------------------|---------------------|
| x + y     | 1                   | 1                   |
| x - y     | 1                   | -1                  |
| x * y     | y                   | x                   |
| x / y     | 1/y                 | -x/y²               |

If z is non-REAL, both gradients are 0.

### Unary Operations

For `y = op(x)`:

| Operation | ∂y/∂x (if y is REAL) |
|-----------|---------------------|
| -x        | -1                  |
| abs(x)    | sign(x)             |
| log(x)    | 1/x                 |
| sqrt(x)   | 1/(2√x)             |
| x^n       | n·x^(n-1)           |

If y is non-REAL, the gradient is 0.

## Chain Rule with Mask-REAL

The key insight is that Mask-REAL propagates through the chain rule:

> **If any intermediate node in a computation path has a non-REAL tag, the entire path contributes zero to the final gradient.**

This is implemented by having non-REAL nodes propagate zero gradients to all their inputs, effectively "breaking" the chain at that point.

### Example: Nested Computation

```python
x → f(x) → g(f(x)) → h(g(f(x)))
    ↓        ↓          ↓
   REAL    PINF       REAL

Even though h returns REAL, ∂h/∂x = 0 because of the non-REAL intermediate g(f(x)).
```

## Practical Benefits

1. **Numerical Stability**: No gradient explosions at singularities
2. **Robust Training**: Models can train even with occasional singularities
3. **Clean Semantics**: Clear, deterministic behavior at edge cases
4. **No ε-hacks**: No need for small epsilon values to avoid division by zero

## Usage Examples

### Basic Gradient Computation

```python
from zeroproof.autodiff import TRNode, gradient_tape, real

# Simple function with gradient
with gradient_tape() as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    y = x * x + 2 * x + 1

grads = tape.gradient(y, [x])
print(f"dy/dx at x=2: {grads[0].value.value}")  # Output: 6.0
```

### Handling Singularities

```python
# Function with pole at x=1
def f(x):
    return TRNode.constant(real(1.0)) / (x - TRNode.constant(real(1.0)))

# At the pole
x = TRNode.parameter(real(1.0))
with gradient_tape() as tape:
    tape.watch(x)
    y = f(x)  # y = 1/0 = +∞

grads = tape.gradient(y, [x])
print(f"Gradient at pole: {grads[0].value.value}")  # Output: 0.0

# Near the pole  
x = TRNode.parameter(real(1.01))
with gradient_tape() as tape:
    tape.watch(x)
    y = f(x)  # y = 1/0.01 = 100

grads = tape.gradient(y, [x])
print(f"Gradient near pole: {grads[0].value.value}")  # Large negative value
```

### Higher-Order Functions

```python
from zeroproof.autodiff import tr_grad

# Define a function
def f(x):
    return x * x * x - 2 * x * x + x

# Get gradient function
grad_f = tr_grad(f)

# Evaluate gradient
x = TRNode.parameter(real(3.0))
df_dx = grad_f(x)
print(f"f'(3) = {df_dx.value.value}")
```

## Testing Gradients

The library provides utilities for testing gradient implementations:

```python
from zeroproof.autodiff import check_gradient

def f(x):
    return x * x + tr_log(x)

x = TRNode.parameter(real(2.0))
analytical, numerical, error = check_gradient(f, x)

print(f"Analytical gradient: {analytical.value.value}")
print(f"Numerical gradient: {numerical}")
print(f"Relative error: {error}")
```

## Advanced Topics

### Persistent Gradient Tapes

For multiple gradient computations:

```python
with gradient_tape(persistent=True) as tape:
    x = TRNode.parameter(real(2.0))
    tape.watch(x)
    y = x * x
    z = x * x * x

# Can compute multiple gradients
dy_dx = tape.gradient(y, [x])
dz_dx = tape.gradient(z, [x])
```

### Custom Operations

To add new operations, implement the forward computation and gradient rule:

```python
# In forward pass
result_value = custom_op(input.value)
result = TRNode(result_value, ...)

# In backward pass gradient computation
if op_type == OpType.CUSTOM:
    grad_input = custom_gradient_rule(inputs, grad_output)
    return [grad_input]
```

## Comparison with Traditional Autodiff

| Aspect | Traditional Autodiff | TR Autodiff with Mask-REAL |
|--------|---------------------|---------------------------|
| Division by zero | NaN/exception | Tagged as PINF/NINF, gradient = 0 |
| Indeterminate forms | NaN propagation | Tagged as PHI, gradient = 0 |
| Gradient at poles | ±∞ or NaN | 0 (controlled) |
| Training stability | Can fail | Continues with zero gradients |
| Need for ε-tricks | Yes | No |

## References

1. James Anderson, "Transreal Arithmetic"
2. The complete specification in `complete_v2.md`
3. Implementation in `zeroproof/autodiff/`
