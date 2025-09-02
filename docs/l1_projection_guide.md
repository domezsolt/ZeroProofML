# L1 Projection Guide for TR-Rational Layers

## Overview

L1 projection is a stability mechanism for TR-Rational layers that constrains the denominator coefficients φ to lie within an L1 ball of radius B. This ensures the denominator Q(x) stays away from zero, preventing numerical instabilities near poles.

## Mathematical Foundation

For a TR-Rational layer computing y = P(x)/Q(x) where:
- P(x) = Σ θₖ ψₖ(x) (numerator)
- Q(x) = 1 + Σ φₖ ψₖ(x) (denominator with leading 1)

The L1 projection ensures:
```
||φ||₁ = Σ |φₖ| ≤ B
```

When ||φ||₁ > B, all coefficients are scaled uniformly:
```
φₖ ← φₖ * (B / ||φ||₁)
```

## Implementation Details

### Automatic Projection

The projection is applied automatically in two places:

1. **During Forward Pass**: Before computing y = P/Q
2. **After Optimization Step**: When using `optimizer.step(model=layer)`

### Code Example

```python
from zeroproof.layers import TRRational
from zeroproof.training import Optimizer

# Create layer with L1 projection bound
layer = TRRational(
    d_p=3,           # Numerator degree
    d_q=3,           # Denominator degree  
    l1_projection=1.0,  # L1 bound for φ
    alpha_phi=0.01   # L2 regularization
)

# Optimizer will apply projection after updates
optimizer = Optimizer(layer.parameters(), learning_rate=0.01)

# Training step
optimizer.zero_grad()
loss.backward()
optimizer.step(model=layer)  # Applies L1 projection
```

## Benefits

### 1. Guaranteed Stability Region

With L1 projection, the denominator satisfies:
```
|Q(x)| ≥ 1 - B * max_k |ψₖ(x)|
```

For monomial basis with bounded inputs |x| ≤ R:
```
|Q(x)| ≥ 1 - B * R^d_q
```

Choosing B < 1/R^d_q guarantees Q(x) never reaches zero.

### 2. Prevents Gradient Explosion

Near poles where Q(x) ≈ 0, gradients can explode. L1 projection keeps Q(x) bounded away from zero, preventing this issue.

### 3. Improved Convergence

By maintaining a stable optimization landscape, L1 projection often leads to faster and more reliable convergence.

## Choosing the L1 Bound

The optimal bound B depends on:

1. **Input Range**: Larger input ranges require smaller B
2. **Denominator Degree**: Higher degrees require smaller B  
3. **Basis Functions**: Different bases have different bounds

### Guidelines

For monomial basis with inputs in [-R, R]:

| d_q | Suggested B for R=1 | Suggested B for R=2 | Suggested B for R=3 |
|-----|---------------------|---------------------|---------------------|
| 1   | 0.8                 | 0.4                 | 0.25                |
| 2   | 0.5                 | 0.2                 | 0.1                 |
| 3   | 0.3                 | 0.1                 | 0.05                |
| 4   | 0.2                 | 0.05                | 0.02                |

### Adaptive Selection

Start with a conservative bound and gradually increase if needed:

```python
# Start conservative
initial_bound = 0.5

# Monitor stability
q_min = layer.compute_q_min(x_batch)

# Adjust if stable
if q_min > 0.5:
    layer.l1_projection *= 1.5
elif q_min < 0.1:
    layer.l1_projection *= 0.7
```

## Integration with Other Features

### With Adaptive Loss

L1 projection works seamlessly with adaptive loss policies:

```python
from zeroproof.training import create_adaptive_loss

layer = TRRational(
    d_p=3, d_q=3,
    l1_projection=1.0,
    adaptive_loss_policy=create_adaptive_loss()
)
```

### With Saturating Gradients

Can be combined with saturating gradient modes:

```python
from zeroproof.layers import SaturatingRational
from zeroproof.autodiff import GradientMode

layer = SaturatingRational(
    d_p=3, d_q=3,
    l1_projection=1.0,
    gradient_mode=GradientMode.SATURATING,
    saturation_bound=10.0
)
```

## Performance Considerations

### Computational Cost

L1 projection has minimal overhead:
- O(d_q) for computing L1 norm
- O(d_q) for scaling coefficients
- Applied once per forward pass

### Memory Usage

No additional memory required beyond the layer parameters.

### Gradient Consistency

When projection occurs, gradients are also scaled to maintain consistency:
```python
if projection_applied:
    gradient *= projection_scale
```

This ensures correct gradient flow through the projected parameters.

## Troubleshooting

### Issue: Projection Not Applied

**Symptom**: ||φ||₁ exceeds bound during training

**Solution**: Ensure using `optimizer.step(model=layer)` instead of `optimizer.step()`

### Issue: Training Instability

**Symptom**: Loss oscillates or diverges

**Solution**: Reduce L1 bound or increase L2 regularization

### Issue: Poor Approximation

**Symptom**: Model can't fit target function

**Solution**: Increase L1 bound (if stable) or increase model capacity

## Complete Example

See `examples/l1_projection_demo.py` for a complete demonstration showing:
- Training with and without projection
- Stability comparison
- Visualization of φ evolution
- Performance metrics

## Summary

L1 projection provides a simple yet effective mechanism for ensuring stability in TR-Rational layers. By constraining denominator coefficients, it:

1. Guarantees numerical stability
2. Prevents gradient explosion
3. Improves training convergence
4. Works seamlessly with other ZeroProof features

The implementation is efficient, fully integrated, and requires minimal configuration to use effectively.
