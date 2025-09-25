# ZeroProof Quick Reference

## Core Operations

### Creating TR Values
```python
import zeroproof as zp

# Basic values
x = zp.real(3.14)      # Finite real
inf = zp.pinf()        # Positive infinity
ninf = zp.ninf()       # Negative infinity
null = zp.phi()        # Nullity (indeterminate)

# From IEEE
tr_val = zp.from_ieee(float('nan'))  # nan → PHI
ieee_val = zp.to_ieee(tr_val)        # PHI → nan
```

### Arithmetic
```python
# Basic operations (never throw exceptions)
result = zp.tr_add(x, y)    # Addition
result = zp.tr_sub(x, y)    # Subtraction
result = zp.tr_mul(x, y)    # Multiplication
result = zp.tr_div(x, y)    # Division

# Special cases handled automatically
zp.tr_div(real(1), real(0))    # → PINF
zp.tr_mul(real(0), pinf())     # → PHI
zp.tr_add(pinf(), ninf())      # → PHI
```

## Autodiff with Mask-REAL

### Basic Gradients
```python
from zeroproof.autodiff import TRNode, gradient_tape

# Create parameters
x = TRNode.parameter(zp.real(2.0))

# Compute with gradient tape
with gradient_tape() as tape:
    tape.watch(x)
    y = zp.tr_mul(x, x)  # x²

# Get gradient
grad = tape.gradient(y, x)  # 2x = 4.0
```

### Mask-REAL Rule
```python
# Non-REAL outputs → zero gradients
x = TRNode.parameter(zp.real(0.0))
with gradient_tape() as tape:
    tape.watch(x)
    y = zp.tr_div(zp.real(1.0), x)  # 1/0 → PINF

grad = tape.gradient(y, x)  # 0.0 (masked)
```

## Neural Network Layers

### TR-Rational Layer
```python
from zeroproof.layers import TRRational

# P(x)/Q(x) with degrees
layer = TRRational(d_p=3, d_q=2)

# Forward pass
x = TRNode.constant(zp.real(1.5))
y, tag = layer.forward(x)
```

### TR-Norm (Epsilon-Free)
```python
from zeroproof.layers import TRNorm

# Batch normalization without epsilon
norm = TRNorm(num_features=10)

# Forward pass (handles zero variance)
output = norm.forward(batch)
```

### TR-Softmax (Rational Surrogate)
```python
from zeroproof.layers import tr_softmax
from zeroproof.autodiff import TRNode

logits = [TRNode.constant(zp.real(0.0)), TRNode.constant(zp.real(1.5)), TRNode.constant(zp.real(-0.5))]
probs = tr_softmax(logits)  # List[TRNode]
```
- Sums to 1.0 in REAL regions; stable for extreme logits.
- Policy toggle (optional): force one‑hot when any `+∞` is present:
```python
from zeroproof.policy import TRPolicy, TRPolicyConfig
TRPolicyConfig.set_policy(TRPolicy(softmax_one_hot_infinity=True))
```

## Optimization Tools

### Profiling
```python
from zeroproof.utils import TRProfiler, profile_tr_operation

# Context manager
with TRProfiler() as profiler:
    result = expensive_computation()
print(profiler.generate_report())

# Decorator
@profile_tr_operation("my_op")
def my_operation(x):
    return zp.tr_mul(x, x)
```

### Caching
```python
from zeroproof.utils import memoize_tr, TRCache

# Simple memoization
@memoize_tr()
def fibonacci(n):
    if n <= 1:
        return zp.real(n)
    return zp.tr_add(fibonacci(n-1), fibonacci(n-2))

# Custom cache
cache = TRCache(max_size=1000, eviction_policy='lru')

@memoize_tr(cache=cache)
def cached_function(x):
    return expensive_operation(x)
```

### Parallel Processing
```python
from zeroproof.utils import parallel_map, ParallelConfig

# Parallel computation
config = ParallelConfig(num_workers=4, backend='thread')
results = parallel_map(process_item, items, config)

# Vectorized operations
from zeroproof.utils import vectorize_operation

@vectorize_operation
def square(x):
    return zp.tr_mul(x, x)

# Works on single values or collections
result = square(zp.real(3))           # Single
results = square([zp.real(i) for i in range(10)])  # List
```

### Graph Optimization
```python
from zeroproof.utils import optimize_tr_graph

# Build graph
with gradient_tape() as tape:
    x = TRNode.parameter(zp.real(2.0))
    y = zp.tr_add(x, zp.real(0))  # x + 0
    z = zp.tr_mul(y, zp.real(1))  # x * 1

# Optimize (removes redundant ops)
optimized = optimize_tr_graph(z)
```

## Bridge Functions

### NumPy Integration
```python
from zeroproof.bridge import from_numpy, to_numpy

# Convert arrays
arr = np.array([1.0, np.inf, np.nan])
tr_arr = from_numpy(arr)
back = to_numpy(tr_arr)

# Tag analysis
print(tr_arr.is_real())   # [True, False, False]
print(tr_arr.is_pinf())   # [False, True, False]
print(tr_arr.is_phi())    # [False, False, True]
```

### PyTorch Integration
```python
from zeroproof.bridge import from_torch, to_torch

# Convert tensors
tensor = torch.tensor([1.0, float('inf')])
tr_tensor = from_torch(tensor, requires_grad=True)

# Automatic Mask-REAL in backward
output = to_torch(tr_tensor)
loss = output.sum()
loss.backward()  # Gradients masked for non-REAL
```

### Precision Control
```python
from zeroproof.bridge import Precision, PrecisionContext

# Set computation precision
with PrecisionContext(Precision.FLOAT32):
    result = expensive_computation()

# Mixed precision
from zeroproof.bridge import MixedPrecisionStrategy

strategy = MixedPrecisionStrategy(
    compute_precision=Precision.FLOAT16,
    accumulate_precision=Precision.FLOAT32
)
```

## Common Patterns

### Safe Division
```python
# Never throws, returns appropriate tag
def safe_divide(a, b):
    result = zp.tr_div(a, b)
    if result.tag == zp.TRTag.REAL:
        return result.value
    elif result.tag == zp.TRTag.PINF:
        return float('inf')
    elif result.tag == zp.TRTag.NINF:
        return float('-inf')
    else:  # PHI
        return float('nan')
```

### Robust Statistics
```python
# Mean that handles infinities
def robust_mean(values):
    total = zp.real(0.0)
    count = 0
    
    for val in values:
        if val.tag == zp.TRTag.REAL:
            total = zp.tr_add(total, val)
            count += 1
    
    if count > 0:
        return zp.tr_div(total, zp.real(count))
  else:
      return zp.phi()  # No valid values
```

## Robotics CLI (RR IK)

### Dataset Generator
```bash
python examples/robotics/rr_ik_dataset.py \
  --n_samples 20000 \
  --singular_ratio 0.35 \
  --displacement_scale 0.1 \
  --singularity_threshold 1e-3 \
  --stratify_by_detj --train_ratio 0.8 \
  --force_exact_singularities \
  --min_detj 1e-6 \
  --bucket-edges 0 1e-5 1e-4 1e-3 1e-2 inf \
  --ensure_buckets_nonzero \
  --seed 123 \
  --output data/rr_ik_dataset.json
```
- JSON metadata: `bucket_edges`, `train_bucket_counts`, `test_bucket_counts`, and when stratified: `stratified_by_detj`, `train_ratio`, `ensured_buckets_nonzero`, optional `singular_ratio_split`, and `seed`.

### Comparator Driver (Parity Runner)
```bash
python experiments/robotics/run_all.py \
  --dataset data/rr_ik_dataset.json \
  --profile quick|full \
  --models tr_basic tr_full rational_eps mlp dls \
  --max_train 2000 --max_test 500 \
  --seed 123 \
  --output_dir results/robotics/run
```
- Quick mode: stratifies the test subset by |det(J)|≈|sin(theta2)| to ensure B0–B3 presence when available, recomputes bucket MSE on the subset, and aligns DLS to evaluate on the same subset.
- Outputs: `comprehensive_comparison.json` with per-bucket MSE and counts, plus per-method JSON summaries.

### Bench Metrics (Per-Epoch)
- Hybrid trainer prints and stores: `avg_step_ms`, `data_time_ms`, `optim_time_ms`, and `batches`.
- Available in training summaries under `bench_history`.

### Optimization Pipeline
```python
# Complete optimization setup
from zeroproof.utils import (
    TRProfiler, memoize_tr, parallel_map,
    optimize_tr_graph
)

# 1. Profile
profiler = TRProfiler()

# 2. Cache
@memoize_tr()
@profiler.profile_operation("compute")
def compute(x):
    return expensive_operation(x)

# 3. Parallelize
with profiler:
    results = parallel_map(compute, data)

# 4. Analyze
print(profiler.generate_report())
```

## Important Constants

### Tag Values
```python
from zeroproof import TRTag

TRTag.REAL  # Finite real numbers
TRTag.PINF  # Positive infinity
TRTag.NINF  # Negative infinity
TRTag.PHI   # Nullity/indeterminate
```

### Special Cases Table

| Operation | Result |
|-----------|--------|
| 1 / 0 | PINF |
| -1 / 0 | NINF |
| 0 / 0 | PHI |
| 0 × ∞ | PHI |
| ∞ + (-∞) | PHI |
| ∞ / ∞ | PHI |
| log(x≤0) | PHI |
| sqrt(x<0) | PHI |

## Performance Tips

1. **Use caching** for expensive repeated computations
2. **Parallelize** independent operations
3. **Optimize graphs** before repeated evaluation
4. **Profile first** to identify bottlenecks
5. **Batch operations** to reduce overhead
6. **Handle special values early** to prevent propagation
7. **Use appropriate precision** for your application

## Error-Free Guarantee

ZeroProof operations **never throw exceptions**. Instead:
- Division by zero → PINF/NINF
- Invalid operations → PHI
- Overflow → PINF/NINF
- Gradients through singularities → 0 (Mask-REAL)

This enables robust numerical computing without defensive programming!
