# Saturating Gradient Mode Guide

## Overview

ZeroProof provides two gradient computation modes for handling singularities and undefined operations:

1. **Mask-REAL** (default): Zero gradients for non-REAL forward values
2. **Saturating**: Bounded gradients that smoothly saturate near singularities

The saturating gradient mode is an alternative approach that maintains gradient flow even at singularities while preventing gradient explosion.

## Theory

### Mask-REAL Rule (Default)
When the forward pass produces a non-REAL tag (PINF, NINF, PHI), all gradients with respect to that operation's inputs are set to zero:

```
if forward_tag ∈ {PINF, NINF, PHI}:
    ∂output/∂input = 0
```

This ensures stability but completely stops gradient flow through singular paths.

### Saturating Gradient Rule
Instead of zeroing gradients, the saturating mode bounds them using transreal arithmetic:

- For reciprocals: `1/x → x/(x² + bound²)`
- For quotients: `1/Q² → 1/(Q² + bound²)`
- Smoothly transitions to bounded values near singularities

This maintains gradient flow while preventing explosion.

## Basic Usage

### Setting Gradient Mode

```python
import zeroproof as zp
from zeroproof.autodiff import GradientMode, gradient_mode, use_saturating

# Global mode change
zp.autodiff.use_saturating(bound=1.0)

# Context manager for temporary change
with gradient_mode(GradientMode.SATURATING, saturation_bound=5.0):
    # Gradients computed with saturation
    loss.backward()

# Back to default
zp.autodiff.use_mask_real()
```

### Using with TR-Rational Layers

```python
from zeroproof.layers import SaturatingTRRational

# Create layer with saturating gradients
model = SaturatingTRRational(
    d_p=4,
    d_q=3,
    gradient_mode=GradientMode.SATURATING,
    saturation_bound=1.0
)

# Forward pass uses specified mode
y, tag = model.forward(x)
```

## Examples

### Comparing Gradient Modes

```python
import zeroproof as zp
from zeroproof.autodiff import TRNode, GradientMode, gradient_mode

# Create a division near zero
x = TRNode.parameter(zp.real(1.0))
epsilon = TRNode.parameter(zp.real(0.01))

# Mask-REAL mode
with gradient_mode(GradientMode.MASK_REAL):
    y = x / epsilon  # Gradient will be large: 1/0.01 = 100
    y.backward()
    mask_grad = epsilon.gradient.value
    print(f"Mask-REAL gradient: {mask_grad}")

# Reset
x.zero_grad()
epsilon.zero_grad()

# Saturating mode
with gradient_mode(GradientMode.SATURATING, saturation_bound=1.0):
    y = x / epsilon  # Gradient will be bounded
    y.backward()
    sat_grad = epsilon.gradient.value
    print(f"Saturating gradient: {sat_grad}")
```

### Training Near Poles

```python
# Model that can learn functions with poles
model = SaturatingTRRational(
    d_p=3, 
    d_q=2,
    gradient_mode=GradientMode.SATURATING,
    saturation_bound=10.0,  # Higher bound allows more gradient flow
    alpha_phi=0.01  # Regularization still important
)

# Training loop
optimizer = zp.training.Optimizer(model.parameters(), lr=0.001)

for x, y_true in data:
    optimizer.zero_grad()
    
    # Forward pass
    y_pred, tag = model.forward(x)
    
    # Loss computation
    if tag == zp.TRTag.REAL:
        loss = 0.5 * (y_pred - y_true) ** 2
    else:
        loss = zp.TRNode.constant(zp.real(1.0))  # Penalty
    
    # Backward with saturating gradients
    loss.backward()
    
    # Update - gradients are bounded even near poles
    optimizer.step()
```

### Ablation Study Helper

```python
# Compare gradient modes on the same model
model = SaturatingTRRational(d_p=4, d_q=3)

# Test batch near singularities
x_batch = [zp.real(0.0), zp.real(0.1), zp.real(0.5)]

# Run comparison
results = model.compare_gradient_modes(x_batch)

# Analyze results
for mode in ['mask_real', 'saturating']:
    print(f"\n{mode} mode:")
    for i, (grads, tag) in enumerate(zip(
        results[mode]['gradients'], 
        results[mode]['tags']
    )):
        print(f"  x={x_batch[i].value}: tag={tag.name}, "
              f"grad_norm={sum(g**2 for g in grads)**0.5:.4f}")
```

## Choosing Saturation Bound

The saturation bound controls how gradients behave near singularities:

- **Small bound (0.1-1.0)**: Strong saturation, very stable but may limit learning
- **Medium bound (1.0-10.0)**: Good balance for most applications
- **Large bound (10.0-100.0)**: Allows more gradient flow, closer to standard gradients

```python
# Conservative - very stable
model_stable = create_saturating_rational(
    d_p=3, d_q=2, 
    mode="saturating",
    saturation_bound=0.5
)

# Balanced - good for most cases
model_balanced = create_saturating_rational(
    d_p=3, d_q=2,
    mode="saturating", 
    saturation_bound=5.0
)

# Aggressive - more gradient flow
model_aggressive = create_saturating_rational(
    d_p=3, d_q=2,
    mode="saturating",
    saturation_bound=50.0
)
```

## When to Use Each Mode

### Use Mask-REAL (default) when:
- You want guaranteed stability
- Singularities should be strictly avoided
- You prefer simpler gradient dynamics
- You're using adaptive loss policies

### Use Saturating when:
- You need to learn functions with poles
- Gradient flow through near-singular regions is important
- You're doing research on gradient behavior
- You want continuous gradient transitions

## Implementation Details

### Saturation Functions

The key insight is using transreal arithmetic to create bounded versions of problematic operations:

```python
# Standard reciprocal: 1/x → ∞ as x → 0
# Saturating reciprocal: x/(x² + bound²) → bounded

# Standard quotient gradient: -P/Q² → ∞ as Q → 0  
# Saturating gradient: -P/(Q² + bound²) → bounded
```

### No Epsilon Hacks

Unlike traditional approaches, saturation uses transreal arithmetic without arbitrary epsilon values:

```python
# Traditional (problematic)
grad = 1 / (x + epsilon)  # Arbitrary epsilon

# Saturating (principled)
grad = x / (x² + bound²)  # Smooth transition, no discontinuity
```

## Performance Considerations

1. **Computation Cost**: Saturating mode requires additional operations (squaring, addition, division)
2. **Memory**: Same memory footprint as Mask-REAL
3. **Convergence**: May converge differently due to maintained gradient flow
4. **Stability**: More stable than unbounded gradients, less "aggressive" than Mask-REAL

## Best Practices

1. **Start with Mask-REAL**: Use the default mode unless you specifically need saturating behavior
2. **Tune the Bound**: Experiment with different saturation bounds for your problem
3. **Monitor Gradients**: Track gradient norms to ensure they remain reasonable
4. **Combine with Regularization**: Still use denominator regularization (alpha_phi)
5. **Test Both Modes**: Compare performance as an ablation study

## Troubleshooting

### Gradients Still Exploding
- Decrease saturation bound
- Add more regularization
- Check for numerical issues in data

### Learning Too Slow
- Increase saturation bound
- Increase learning rate
- Consider switching to Mask-REAL

### Unstable Training
- Ensure proper initialization
- Use gradient clipping in addition
- Verify data preprocessing

## API Reference

### Functions
- `use_mask_real()`: Switch to Mask-REAL mode
- `use_saturating(bound)`: Switch to saturating mode
- `gradient_mode(mode, bound)`: Context manager

### Classes
- `GradientMode`: Enum with MASK_REAL and SATURATING
- `GradientModeConfig`: Global configuration
- `SaturatingTRRational`: Rational layer with mode support

### Parameters
- `gradient_mode`: Which mode to use
- `saturation_bound`: Bound parameter for saturation
- `alpha_phi`: Regularization (works with both modes)
