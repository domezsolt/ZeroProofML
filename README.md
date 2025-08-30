# ZeroProof

[![Build](https://img.shields.io/github/actions/workflow/status/zeroproof/zeroproof/ci.yml?branch=main)](https://github.com/zeroproof/zeroproof/actions)
[![Coverage](https://img.shields.io/badge/coverage-unknown-lightgrey)](#)
[![Property Suite](https://img.shields.io/badge/property%20tests-passing-brightgreen)](#)
[![E2E No‑NaN](https://img.shields.io/badge/e2e%20no‑NaN-✔-brightgreen)](#)

<div align="center">

![Build Status](https://img.shields.io/github/actions/workflow/status/zeroproof/zeroproof/ci.yml?branch=main)
![Coverage](https://img.shields.io/codecov/c/github/zeroproof/zeroproof)
![PyPI](https://img.shields.io/pypi/v/zeroproof)
![Python Version](https://img.shields.io/pypi/pyversions/zeroproof)
![License](https://img.shields.io/github/license/zeroproof/zeroproof)

**Transreal arithmetic for stable machine learning without epsilon hacks**

[Documentation](https://zeroproof.readthedocs.io) | [Paper](https://arxiv.org/abs/xxxx.xxxxx) | [Examples](examples/)

</div>

## What is ZeroProof?

ZeroProof is a Python library that implements **Transreal (TR) arithmetic** - a revolutionary approach to handling mathematical singularities and undefined operations in machine learning. Instead of using epsilon hacks or arbitrary thresholds, ZeroProof extends real arithmetic with special values that make all operations **total** (never throwing exceptions).

### Key Features

- 🚀 **No NaN propagation** - Undefined forms are handled gracefully with the PHI tag
- 🎯 **Stable training near poles** - Gradients remain bounded even at singularities  
- 🔬 **Deterministic behavior** - No epsilon thresholds or arbitrary choices
- 📐 **Clean mathematics** - Operations match classical calculus on regular paths
- 🔧 **Framework integration** - Works with PyTorch, JAX, and NumPy
- ⚡ **High performance** - Optional JIT compilation with Numba
- 🎛️ **Adaptive loss policy** - Automatic λ adjustment for target coverage
- 🌊 **Saturating gradients** - Alternative gradient mode for continuous flow
- ⚙️ **Wheel mode** - Optional stricter algebra for formal verification

## Quick Start

```python
import zeroproof as zp

# Create transreal scalars
x = zp.real(3.0)      # Regular real number
y = zp.real(0.0)      # Zero
inf = zp.pinf()       # Positive infinity
phi = zp.phi()        # Nullity (undefined)

# All operations are total
result = x / y        # Returns zp.pinf() instead of raising error
result2 = y / y       # Returns zp.phi() (0/0 is undefined)
result3 = inf - inf   # Returns zp.phi() (∞ - ∞ is undefined)

# Use in neural networks
layer = zp.layers.TRRational(degree_p=3, degree_q=2)
output = layer(input_tensor)  # Handles poles gracefully
```

## Installation

```bash
# Basic installation
pip install zeroproof

# With PyTorch support
pip install zeroproof[torch]

# With JAX support  
pip install zeroproof[jax]

# All features
pip install zeroproof[all]
```

## Core Concepts

### Transreal Numbers

ZeroProof extends real numbers with three special values:

- **PINF** (+∞): Positive infinity as a first-class value
- **NINF** (-∞): Negative infinity as a first-class value  
- **PHI** (Φ): Nullity representing undefined forms (0/0, ∞-∞, 0×∞, etc.)

### Total Operations

All arithmetic operations are **total** - they always return a valid transreal value:

```python
# Division by zero
zp.real(1.0) / zp.real(0.0)  # → PINF
zp.real(-1.0) / zp.real(0.0) # → NINF
zp.real(0.0) / zp.real(0.0)  # → PHI

# Infinity arithmetic
zp.pinf() + zp.real(5.0)     # → PINF
zp.pinf() - zp.pinf()        # → PHI
zp.real(0.0) * zp.pinf()     # → PHI
```

### Mask-REAL Autodiff

Gradients flow only through REAL-valued paths:

```python
# If forward pass produces PINF/NINF/PHI, gradients are zero
# This prevents gradient explosions at singularities
loss = model(x)  # If any intermediate is non-REAL, that path contributes 0 gradient
```

### Precision Control

ZeroProof uses float64 by default for maximum precision, with optional modes:

```python
# Default: float64 precision
x = zp.real(1.0 / 3.0)

# Temporary float32 for performance
with zp.precision_context('float32'):
    y = zp.real(2.0)
    z = x * y  # Computed in float32

# Global precision change
zp.PrecisionConfig.set_precision('float16')  # For embedded systems
```

### Adaptive Loss Policy

ZeroProof automatically adjusts the rejection penalty to achieve target coverage:

```python
# Create model with adaptive loss
from zeroproof.training import create_adaptive_loss

adaptive_loss = create_adaptive_loss(
    target_coverage=0.95,    # Target 95% REAL outputs
    learning_rate=0.01       # Lambda adjustment rate
)

model = zp.layers.TRRational(
    d_p=4, d_q=3,
    adaptive_loss_policy=adaptive_loss
)

# Train with automatic lambda adjustment
trainer = zp.training.TRTrainer(model)
history = trainer.train(data)

# Monitor coverage and lambda evolution
print(f"Final coverage: {history['coverage'][-1]:.3f}")
print(f"Final λ_rej: {history['lambda_rej'][-1]:.3f}")
```

### Saturating Gradients

Choose between two gradient modes for handling singularities:

```python
# Default: Mask-REAL (zero gradients at singularities)
model = zp.layers.TRRational(d_p=3, d_q=2)

# Alternative: Saturating (bounded gradients)
from zeroproof.layers import SaturatingTRRational

model = SaturatingTRRational(
    d_p=3, d_q=2,
    gradient_mode=zp.autodiff.GradientMode.SATURATING,
    saturation_bound=5.0  # Controls gradient bounding
)

# Temporary mode switch
with zp.autodiff.gradient_mode(zp.autodiff.GradientMode.SATURATING):
    loss.backward()  # Gradients computed with saturation
```

### Wheel Mode

Optional stricter algebra for formal verification:

```python
# Standard transreal mode (default)
result = zp.tr_mul(zp.real(0), zp.pinf())  # Returns PHI

# Wheel mode - stricter algebra
with zp.wheel_mode():
    result = zp.tr_mul(zp.real(0), zp.pinf())  # Returns BOTTOM (⊥)
    
# Key differences in wheel mode:
# - 0 × ∞ = ⊥ (instead of Φ)
# - ∞ + ∞ = ⊥ (instead of ∞)
# - Bottom propagates through all operations
```

## Advanced Features

### TR-Rational Layers

Learn rational functions P(x)/Q(x) that can model poles:

```python
layer = zp.layers.TRRational(
    degree_p=4,           # Numerator degree
    degree_q=3,           # Denominator degree  
    basis='chebyshev',    # Polynomial basis
    lambda_rej=1.0,       # Penalty for non-REAL outputs
)
```

### Epsilon-Free Normalization

Batch normalization without epsilon hacks:

```python
norm = zp.layers.TRNorm(num_features=128)
# Handles zero-variance features deterministically
# No more eps=1e-5 parameters!
```

### IEEE-754 Bridge

Seamless conversion between IEEE floats and transreal values:

```python
# From IEEE to TR
tr_value = zp.from_ieee(float('inf'))   # → PINF
tr_value = zp.from_ieee(float('nan'))   # → PHI

# From TR to IEEE  
ieee_value = zp.to_ieee(zp.phi())       # → NaN
```

## Examples

### Learning Functions with Poles

```python
import zeroproof as zp
import torch

# Define a rational model that can learn poles
model = zp.torch.TRRationalNet(
    input_dim=1,
    hidden_dims=[64, 32],
    output_dim=1,
    rational_degree=(3, 2),
)

# Train on data near poles - no special handling needed!
optimizer = torch.optim.Adam(model.parameters())
for x, y in data_loader:
    pred = model(x)
    loss = zp.losses.tr_mse(pred, y, lambda_rej=1.0)
    loss.backward()
    optimizer.step()
```

### Stable Optimization Near Singularities

```python
# Adaptive learning rate based on denominators
scheduler = zp.optim.SafeLRScheduler(
    optimizer,
    monitor='q_min',  # Minimum |Q(x)| in batch
    patience=10,
)
```

## Documentation

- [Getting Started Guide](https://zeroproof.readthedocs.io/en/latest/getting_started.html)
- [Mathematical Theory](https://zeroproof.readthedocs.io/en/latest/theory.html)
- [API Reference](https://zeroproof.readthedocs.io/en/latest/api.html)
- [Examples Gallery](https://zeroproof.readthedocs.io/en/latest/examples.html)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Clone the repository
git clone https://github.com/zeroproof/zeroproof.git
cd zeroproof

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Run property-based tests
pytest -m property
```

## Citation

If you use ZeroProof in your research, please cite:

```bibtex
@article{zeroproof2024,
  title={ZeroProof: Transreal Arithmetic for Stable Machine Learning},
  author={ZeroProof Team},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## License

ZeroProof is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This implementation is based on the transreal arithmetic theory developed by [James Anderson](https://en.wikipedia.org/wiki/James_A._D._W._Anderson) and colleagues.

---

<div align="center">
Made with ❤️ by the ZeroProof Team
</div>
