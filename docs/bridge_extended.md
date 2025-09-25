# Extended IEEE Bridge Documentation

## Overview

The ZeroProof bridge provides comprehensive interoperability between transreal arithmetic and popular numerical computing frameworks. It handles conversions, preserves semantics, and enables seamless integration with existing workflows.

## Core Features

### 1. Framework Integration

#### NumPy Bridge
```python
import numpy as np
from zeroproof.bridge import from_numpy, to_numpy, TRArray

# Convert NumPy array to transreal
arr = np.array([1.0, np.inf, -np.inf, np.nan, 0.0])
tr_arr = from_numpy(arr)

# Access tag information
print(tr_arr.is_real())     # [True, False, False, False, True]
print(tr_arr.is_pinf())     # [False, True, False, False, False]
print(tr_arr.is_phi())      # [False, False, False, True, False]

# Convert back to NumPy
ieee_arr = to_numpy(tr_arr)
```

##### Packed Arrays (bit‑packed tags)

For memory efficiency, ZeroProof offers a packed representation that stores tag masks as bit arrays (1 bit/node) and implies PHI as the remainder:

```python
from zeroproof.bridge import from_numpy_packed, to_numpy, TRArrayPacked

packed = from_numpy_packed(arr)   # TRArrayPacked
restored = to_numpy(packed)       # IEEE round‑trip
```

Use `TRArrayPacked` when tag storage overhead matters (large arrays). Standard `TRArray` remains the most convenient for tag inspection.

#### PyTorch Bridge
```python
import torch
from zeroproof.bridge import from_torch, to_torch, TRTensor

# Convert PyTorch tensor
tensor = torch.tensor([1.0, float('inf'), float('nan')])
tr_tensor = from_torch(tensor, requires_grad=True)

# Automatic Mask-REAL in gradients
output = to_torch(tr_tensor)
loss = (output ** 2).sum()
loss.backward()  # Gradients are zero for non-REAL elements

# GPU support
if torch.cuda.is_available():
    cuda_tr = tr_tensor.cuda()
```

#### JAX Bridge
```python
import jax.numpy as jnp
from zeroproof.bridge import from_jax, to_jax, tr_add_jax

# Convert JAX array
arr = jnp.array([1.0, jnp.inf, -jnp.inf])
tr_arr = from_jax(arr)

# TR operations in JAX
a = from_jax(jnp.array([1.0, jnp.inf]))
b = from_jax(jnp.array([2.0, -jnp.inf]))
result = tr_add_jax(a, b)  # [3.0, PHI]
```

### 2. Precision Support

#### Precision Contexts
```python
from zeroproof.bridge import Precision, PrecisionContext

# Set computation precision
with PrecisionContext(Precision.FLOAT32):
    # All operations use float32 precision
    result = tr_add(a, b)
    
    # Nested contexts
    with PrecisionContext(Precision.FLOAT16):
        # Now using float16
        low_prec_result = tr_mul(x, y)
```

#### Precision-Aware Operations
```python
from zeroproof.bridge import with_precision, check_precision_overflow

# Apply precision to a value
x = real(1e10)
x_f16 = with_precision(x, Precision.FLOAT16)  # Overflows to PINF

# Check for overflow
overflow_tag = check_precision_overflow(1e40, Precision.FLOAT32)
print(overflow_tag)  # TRTag.PINF
```

#### Mixed Precision Strategies
```python
from zeroproof.bridge import MixedPrecisionStrategy

# Define strategy
strategy = MixedPrecisionStrategy(
    compute_precision=Precision.FLOAT16,      # Fast computation
    accumulate_precision=Precision.FLOAT32,   # Accurate accumulation
    output_precision=Precision.FLOAT16        # Compact output
)

# Use strategy
result = strategy.accumulate(values)
final = strategy.finalize(result)
```

### 3. Array Operations

#### Tag Analysis
```python
from zeroproof.bridge import count_tags, where_real, real_values

# Count elements by tag
counts = count_tags(tr_arr)
print(counts)  # {'REAL': 10, 'PINF': 2, 'NINF': 1, 'PHI': 3}

# Find REAL elements
real_indices = where_real(tr_arr)

# Extract only REAL values
reals = real_values(tr_arr)
```

#### Utility Functions
```python
from zeroproof.bridge import validate_array, check_finite, clip_infinities

# Validate input
validate_array(numpy_array)  # Raises if invalid

# Check for all finite
check_finite(tr_arr)  # Raises if non-REAL values present

# Clip infinities for compatibility
clipped = clip_infinities(tr_arr, max_value=1e308)
```

## Conversion Semantics

### IEEE to TR Mapping

| IEEE Value | TR Representation |
|------------|-------------------|
| Finite float | (value, REAL) |
| +∞ | (—, PINF) |
| -∞ | (—, NINF) |
| NaN | (—, PHI) |

### Special Value Handling

- **Signed zeros**: Preserved in REAL values
- **Subnormals**: Treated as normal REAL values
- **NaN payloads**: Not preserved (all NaN → PHI)

### Partial Homomorphism (Non‑NaN Regime)

Let `Φ` be the IEEE→TR mapping and `Ψ` the TR→IEEE mapping on REAL/INF tags. For IEEE scalars `x, y` and an operation `∘ ∈ {+, −, ×, ÷}`:

> If IEEE evaluates `x ∘ y` without producing NaN, then
> `Φ(x) ∘_TR Φ(y) = Φ(x ∘ y)`.

This means the bridge is a homomorphism on the non‑NaN subset; NaNs map to `PHI`, for which TR algebra has well‑defined (total) behavior.

### Signed Zero Retention

IEEE “−0.0” is retained on REAL zero values via a latent sign flag (e.g., policy logic uses `copysign`). This matters for directional limits like `1/±0`:

```python
from zeroproof.policy import TRPolicy, TRPolicyConfig

pol = TRPolicy(keep_signed_zero=True)
TRPolicyConfig.set_policy(pol)

# Classifier can distinguish approach direction when computing tag signs
```

### Export Policy (Φ → IEEE)

`PHI` has no IEEE numeric, so `Ψ(PHI)` returns NaN by default. Consumers may choose a stricter policy (raise) or a tolerant one (return NaN) depending on context.

### Determinism and ULP Bands

Floating‑point perturbations near guard bands can flip tag classification. Define ULP‑scaled thresholds (`τ_Q, τ_P = Θ(ULP)`) and use hysteresis for deterministic tag decisions across devices. Prefer deterministic reductions (pairwise/Kahan) to avoid order‑sensitivity.

### Round-Trip Guarantees

```python
# Scalar round-trip
x = 3.14
assert to_ieee(from_ieee(x)) == x

# Array round-trip (NumPy)
arr = np.array([1.0, np.inf, np.nan])
assert np.array_equal(
    to_numpy(from_numpy(arr)), 
    arr, 
    equal_nan=True
)
```

## Gradient Integration

### Automatic Mask-REAL

When using PyTorch or JAX bridges, the Mask-REAL rule is automatically applied:

```python
# PyTorch example
x = torch.tensor([1.0, float('inf'), 2.0], requires_grad=True)
tr_x = from_torch(x)

# Perform operations...
y = some_tr_operation(tr_x)

# Convert back and compute gradients
y_torch = to_torch(y)
loss = y_torch.sum()
loss.backward()

# x.grad will be [some_value, 0.0, some_value]
# The gradient is zero at the infinity due to Mask-REAL
```

### Custom Operations

```python
from zeroproof.bridge import TRFunction, mask_real_backward

class MyTROperation(TRFunction):
    @staticmethod
    def forward(ctx, values, tags):
        # Implement forward pass
        output_values = custom_forward(values)
        output_tags = custom_tag_logic(values, tags)
        ctx.save_for_backward(tags, output_tags)
        return output_values, output_tags
    
    @staticmethod
    def backward(ctx, grad_values, grad_tags):
        tags, output_tags = ctx.saved_tensors
        # Mask-REAL automatically applied
        masked_grad = mask_real_backward(grad_values, output_tags)
        return masked_grad, None
```

## Precision Analysis

### Analyzing Data Requirements

```python
from zeroproof.bridge import analyze_precision_requirements

values = [1e-5, 1.0, 100.0, 1e6, 1e20]
analysis = analyze_precision_requirements(values)

print(analysis)
# {
#     'min_precision': Precision.FLOAT32,
#     'recommended_precision': Precision.FLOAT64,
#     'range': (1e-5, 1e20),
#     'needs_float64': False,
#     'fits_float16': False
# }
```

### Precision-Safe Operations

```python
from zeroproof.bridge import precision_safe_operation

# Automatically handles overflow based on precision
result = precision_safe_operation(
    'mul', 
    real(1e20), 
    real(1e20),
    precision=Precision.FLOAT32
)
# Result will be PINF due to float32 overflow
```

## Best Practices

### 1. Choose Appropriate Precision

- **FLOAT16**: Neural network inference, memory-constrained applications
- **FLOAT32**: General computation, neural network training
- **FLOAT64**: Scientific computing, accumulations
- **BFLOAT16**: Deep learning with wide dynamic range

### 2. Handle Framework Availability

```python
from zeroproof.bridge import NUMPY_AVAILABLE, TORCH_AVAILABLE

if NUMPY_AVAILABLE:
    # Use NumPy bridge
    from zeroproof.bridge import TRArray
else:
    # Fallback to basic arrays
    pass
```

### 3. Preserve Semantics

- Always use bridge functions for conversion
- Don't mix TR and IEEE operations without conversion
- Be aware of precision limits

### 4. Optimize for Performance

```python
# Batch conversions
values = [from_ieee(x) for x in ieee_values]  # Slow
tr_arr = from_numpy(np.array(ieee_values))    # Fast

# Reuse arrays
tr_arr = from_numpy(data)
# ... many operations on tr_arr ...
result = to_numpy(tr_arr)  # Single conversion at end
```

## Advanced Topics

### Cross-Framework Workflows

```python
# Start with NumPy
np_data = np.load('data.npy')
tr_arr = from_numpy(np_data)

# Process with custom TR operations
processed = custom_tr_algorithm(tr_arr)

# Convert to PyTorch for neural network
if TORCH_AVAILABLE:
    torch_data = torch.tensor(to_numpy(processed))
    model_output = neural_network(torch_data)
    
# Back to TR for stable post-processing
tr_output = from_torch(model_output)
final_result = tr_post_process(tr_output)
```

### Debugging Tools

```python
# Print tag distribution
def debug_tags(tr_array):
    counts = count_tags(tr_array)
    total = sum(counts.values())
    print("Tag distribution:")
    for tag, count in counts.items():
        percent = 100 * count / total
        print(f"  {tag}: {count} ({percent:.1f}%)")

# Check for unexpected non-REAL values
def check_computation_health(tr_array, name="array"):
    counts = count_tags(tr_array)
    non_real = counts['PINF'] + counts['NINF'] + counts['PHI']
    if non_real > 0:
        print(f"Warning: {name} has {non_real} non-REAL values")
        debug_tags(tr_array)
```

## Performance Considerations

1. **Conversion Overhead**: Minimize conversions between TR and IEEE representations
2. **Memory Usage**: TR arrays store both values and tags (2x memory)
3. **GPU Operations**: Use framework-specific bridges for GPU computation
4. **Precision**: Lower precision reduces memory and may improve performance

## Future Extensions

- Sparse tensor support
- Distributed array operations
- Custom CUDA kernels for TR operations
- Integration with more frameworks (TensorFlow, MXNet, etc.)
- Automatic mixed precision optimization
