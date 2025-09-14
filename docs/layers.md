# Transreal Neural Network Layers

> See also: Topic 4 “Layers & Variants” (`docs/topics/04_layers.md`) for a curated overview and when to use each variant.

## Overview

ZeroProof provides neural network layers that leverage transreal arithmetic to handle singularities gracefully. These layers never produce NaN or raise exceptions, instead using the TR tag system to track special values.

## TR-Rational Layer

### Concept

The TR-Rational layer computes:
```
y = P_θ(x) / Q_φ(x)
```

where P and Q are polynomials with learnable coefficients. This can represent a wide variety of functions including those with poles and asymptotes.

### Key Features

1. **Total Operations**: Division by zero produces tagged infinities (PINF/NINF) or PHI, never exceptions
2. **Stable Gradients**: Mask-REAL rule prevents gradient explosions at singularities
3. **Identifiability**: Q has a leading 1 to prevent trivial scaling ambiguity
4. **Regularization**: Optional L2 penalty on denominator coefficients to control pole locations

### Mathematical Details

Given basis functions ψ(x) = [ψ₀(x), ψ₁(x), ..., ψ_d(x)]:

- P(x) = Σ_{k=0}^{d_p} θ_k ψ_k(x)
- Q(x) = 1 + Σ_{k=1}^{d_q} φ_k ψ_k(x)

Forward pass rules:
- If Q(x) ≠ 0: y = P(x)/Q(x) with tag REAL
- If Q(x) = 0 and P(x) ≠ 0: y has tag PINF or NINF (by sign of P)
- If Q(x) = 0 and P(x) = 0: y has tag PHI

### Usage Example

```python
from zeroproof.layers import TRRational, ChebyshevBasis

# Create a rational layer with degree 2 numerator, degree 1 denominator
layer = TRRational(d_p=2, d_q=1, basis=ChebyshevBasis())

# Forward pass
x = TRNode.constant(real(0.5))
y, tag = layer.forward(x)

# The layer can be trained even with inputs at poles
x_pole = TRNode.constant(real(1.0))  # If Q(1) = 0
y_pole, tag_pole = layer.forward(x_pole)  # tag_pole may be PINF/NINF/PHI
```

### Multi‑Input / Multi‑Output Variants

For vector inputs and multi‑output predictions with shared pole structure, see Topic 4 “Layers & Variants” (`docs/topics/04_layers.md`):

- `TRRationalMulti`: multiple outputs with optional shared denominator `Q`.
- `TRMultiInputRational`: lightweight TR‑MLP front end (R^D→K features) feeding TR‑Rational heads; supports shared `Q` and an optional pole head.

These variants expose vector forward helpers and a structured result (`forward_fully_integrated`) including tags and optional `Q` diagnostics. Refer to `zeroproof/layers/multi_input_rational.py` for details and a 4D→2D robotics usage example.

### Basis Functions

The library provides several basis function options:

1. **Monomial Basis**: ψ_k(x) = x^k
   - Simple but can be numerically unstable for high degrees
   - Good for low-degree polynomials

2. **Chebyshev Basis**: Chebyshev polynomials of the first kind
   - Optimal for approximation on bounded intervals
   - Numerically stable for high degrees
   - Bounded by [-1, 1] on the standard domain

3. **Fourier Basis**: Sine and cosine functions (planned)

### Training Considerations

1. **Regularization**: Use `alpha_phi` to penalize large denominator coefficients
2. **Stability Monitoring**: Track `q_min` (minimum |Q(x)|) during training
3. **L1 Projection**: Constrain ||φ||₁ ≤ B to ensure Q stays away from zero
   - Fully implemented with automatic projection during forward pass
   - Integrated with optimizer for post-update projection
   - Preserves gradient consistency by scaling gradients proportionally

4. **TR-Norm Running Stats**: Optional tracking for compatibility
   - `track_running_stats=True` records running mean/var using EMA with `momentum`
   - Running stats are tracked but not used for inference-time normalization yet
   - No exceptions are raised; fields: `running_mean`, `running_var`, `num_batches_tracked`

## TR-Norm (Epsilon-Free Normalization)

### Concept

TR-Norm implements batch/layer normalization without epsilon hacks:

```
If σ² > 0: ŷ = γ(x - μ)/σ + β
If σ² = 0: ŷ = β  (deterministic bypass)
```

This is the mathematical limit of standard normalization as ε→0⁺.

### Key Features

1. **No Epsilon**: Handles zero variance deterministically
2. **Drop-Null Statistics**: Computes mean/variance over REAL values only
3. **Stable Gradients**: No division-by-zero in backward pass
4. **Limit Equivalence**: Matches standard BN(ε) as ε→0⁺ when σ²>0

### Mathematical Details

For each feature j:

1. Collect REAL values: S_j = {i | x_{ij}.tag = REAL}
2. Compute statistics:
   - μ_j = (1/|S_j|) Σ_{i∈S_j} x_{ij}.value
   - σ²_j = (1/|S_j|) Σ_{i∈S_j} (x_{ij}.value - μ_j)²
3. Normalize:
   - If σ²_j > 0: ŷ_{ij} = γ_j(x_{ij} - μ_j)/√σ²_j + β_j
   - If σ²_j = 0: ŷ_{ij} = β_j

### Usage Example

```python
from zeroproof.layers import TRNorm, TRLayerNorm

# Batch normalization
bn = TRNorm(num_features=10)

# Input: [batch_size, num_features]
batch = [
    [real(1.0), real(2.0), ...],  # Sample 1
    [real(3.0), real(4.0), ...],  # Sample 2
    # ...
]

normalized = bn(batch)

# Layer normalization (normalizes across features per sample)
ln = TRLayerNorm(normalized_shape=10)
sample = [real(1.0), real(2.0), ..., real(10.0)]
normalized_sample = ln(sample)
```

### Handling Non-REAL Values

When the input contains non-REAL values:

1. Statistics are computed over REAL values only (drop-null)
2. Non-REAL inputs still produce outputs (may remain non-REAL)
3. If all values are non-REAL, triggers bypass to β

Example:
```python
# Mixed batch
batch = [
    [real(1.0)],
    [TRNode.constant(pinf())],  # Dropped from statistics
    [real(3.0)],
    [TRNode.constant(phi())],    # Dropped from statistics
    [real(5.0)],
]

# Stats computed from: 1.0, 3.0, 5.0 only
normalized = norm(batch)
```

## Gradient Flow

Both layers work seamlessly with the autodiff system:

### TR-Rational Gradients

When the forward pass produces REAL output:
- ∂y/∂θ_k = ψ_k(x) / Q(x)
- ∂y/∂φ_k = -y · ψ_k(x) / Q(x)

When the forward pass produces non-REAL output:
- All parameter gradients are zero (Mask-REAL)

### TR-Norm Gradients

For σ² > 0 (regular branch):
- Standard backpropagation through normalization
- Coupled gradients due to batch statistics

For σ² = 0 (bypass branch):
- ∂ŷ/∂x = 0 (no dependence on input)
- ∂ŷ/∂β = 1 (direct pass-through)
- ∂ŷ/∂γ = 0 (not used in bypass)

## Design Philosophy

These layers embody the ZeroProof philosophy:

1. **Totality**: Every operation is defined for all inputs
2. **Determinism**: No hidden epsilon values or random behavior
3. **Stability**: Gradients remain bounded even at singularities
4. **Transparency**: Clear semantics via the tag system

## Comparison with Traditional Layers

| Aspect | Traditional | Transreal |
|--------|------------|-----------|
| Division by zero | NaN/exception | Tagged infinity |
| Zero variance norm | Add ε hack | Deterministic bypass |
| Gradient at poles | Explosion/NaN | Zero (Mask-REAL) |
| Numerical stability | Requires careful tuning | Built-in via tags |

## Best Practices

1. **Monitor Tags**: Track the frequency of non-REAL outputs during training
2. **Regularize Denominators**: Use alpha_phi to keep poles under control
3. **Choose Appropriate Basis**: Use Chebyshev for high-degree approximations
4. **Start Simple**: Begin with low polynomial degrees and increase as needed
5. **Leverage Bypass**: The zero-variance bypass in TR-Norm can be useful for certain architectures

## Future Extensions

- Multivariate rational functions
- Padé approximants with TR semantics
- Specialized bases (Hermite, Laguerre, etc.)
- GPU-optimized implementations
- Integration with deep learning frameworks
