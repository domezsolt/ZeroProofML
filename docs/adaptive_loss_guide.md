# Adaptive Loss Policy Guide

## Overview

The adaptive loss policy in ZeroProof automatically adjusts the rejection penalty (λ_rej) during training to achieve a target coverage rate. This implements the Lagrange multiplier approach specified in the original paper, where coverage is the proportion of outputs with REAL tags.

## Key Concepts

### Coverage
Coverage is the proportion of model outputs that have REAL tags (as opposed to PINF, NINF, or PHI). High coverage indicates the model is successfully avoiding singularities.

### Rejection Penalty (λ_rej)
When a model output has a non-REAL tag, instead of computing the normal loss, we apply a fixed penalty λ_rej. This encourages the model to produce REAL outputs.

### Lagrange Multiplier Update
The adaptive policy treats λ_rej as a Lagrange multiplier and updates it according to:
```
λ ← λ + η_λ * (c* - c_actual)
```
where:
- `c*` is the target coverage (e.g., 0.95 for 95% REAL outputs)
- `c_actual` is the observed coverage
- `η_λ` is the learning rate for lambda updates

## Basic Usage

### 1. Create Adaptive Loss Policy

```python
from zeroproof.training import create_adaptive_loss

# Create with default settings
adaptive_loss = create_adaptive_loss(
    target_coverage=0.95,    # Target 95% REAL outputs
    learning_rate=0.01,      # Lambda update rate
    initial_lambda=1.0,      # Starting penalty
    base_loss="mse"          # Base loss function
)
```

### 2. Use with TR-Rational Layer

```python
from zeroproof.layers import TRRational

model = TRRational(
    d_p=4,
    d_q=3,
    adaptive_loss_policy=adaptive_loss  # Pass the policy
)
```

### 3. Train with Adaptive Policy

```python
from zeroproof.training import TRTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    learning_rate=0.001,
    use_adaptive_loss=True,
    target_coverage=0.95
)

# Create trainer
trainer = TRTrainer(model, config=config)

# Train
history = trainer.train(train_data, val_data)
```

## Advanced Configuration

### Custom Adaptive Loss Config

```python
from zeroproof.training import AdaptiveLambda, AdaptiveLossConfig

config = AdaptiveLossConfig(
    initial_lambda=1.0,         # Starting penalty
    target_coverage=0.9,        # Target 90% coverage
    learning_rate=0.05,         # Faster adaptation
    lambda_min=0.0,             # Minimum penalty
    lambda_max=10.0,            # Maximum penalty
    momentum=0.9,               # Momentum for smoother updates
    warmup_steps=100,           # Steps before adapting
    update_frequency=10,        # Update every N steps
    exponential_decay=0.999     # Optional LR decay
)

adaptive_lambda = AdaptiveLambda(config)
```

### Multiple Output Coverage

For models with multiple outputs:

```python
from zeroproof.training import MultiOutputCoverageTracker

tracker = MultiOutputCoverageTracker(
    n_outputs=3,
    target_coverage=0.95,
    aggregate_mode="min"  # Use minimum coverage across outputs
)
```

## Monitoring and Analysis

### Get Statistics During Training

```python
# During or after training
stats = trainer.loss_policy.get_statistics()

print(f"Current coverage: {stats['current_coverage']:.3f}")
print(f"Lambda value: {stats['lambda_rej']:.3f}")
print(f"Coverage gap: {stats['coverage_gap']:.3f}")
print(f"Tag distribution: {stats['tag_distribution']}")
```

### Visualize Adaptation

```python
import matplotlib.pyplot as plt

# Plot coverage evolution
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(history['coverage'])
plt.axhline(y=0.95, color='r', linestyle='--', label='Target')
plt.xlabel('Epoch')
plt.ylabel('Coverage')
plt.title('Coverage Over Time')
plt.legend()

plt.subplot(132)
plt.plot(history['lambda_rej'])
plt.xlabel('Epoch')
plt.ylabel('λ_rej')
plt.title('Adaptive Penalty')

plt.subplot(133)
plt.plot(history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.tight_layout()
plt.show()
```

## Best Practices

### 1. Choose Appropriate Target Coverage
- **0.90-0.95**: Good default for most problems
- **0.99+**: Very conservative, may limit model flexibility
- **<0.85**: Allows more singularities, use with caution

### 2. Set Learning Rates Carefully
- Lambda learning rate should be ~10x smaller than model learning rate
- Too high: Unstable oscillations in coverage
- Too low: Slow adaptation to target

### 3. Use Warmup Period
- Allow model to stabilize before adapting lambda
- Typical warmup: 50-200 steps

### 4. Monitor Tag Distribution
```python
dist = stats['tag_distribution']
if dist['PHI'] > 0.05:  # More than 5% undefined
    print("Warning: High proportion of undefined outputs")
```

### 5. Regularization Interaction
- Denominator regularization (alpha_phi) affects singularity likelihood
- Balance regularization with adaptive loss for best results

## Example: Training Near Singularities

```python
import numpy as np
import zeroproof as zp

# Generate data with known singularity
x = np.linspace(-2, 2, 1000)
x = x[np.abs(x) > 0.1]  # Avoid x=0
y = 1 / x + np.random.normal(0, 0.1, len(x))

# Prepare data
train_data = prepare_data(x, y)

# Model with adaptive loss
model = zp.layers.TRRational(
    d_p=3, 
    d_q=2,
    alpha_phi=0.01  # Regularize denominator
)

# Train with coverage target
config = zp.training.TrainingConfig(
    use_adaptive_loss=True,
    target_coverage=0.92,  # Allow 8% non-REAL
    lambda_learning_rate=0.02
)

trainer = zp.training.TRTrainer(model, config=config)
history = trainer.train(train_data)

# Check final coverage
final_coverage = history['coverage'][-1]
print(f"Final coverage: {final_coverage:.3f}")
```

## Troubleshooting

### Coverage Too Low
- Increase lambda_learning_rate
- Increase initial_lambda
- Add more regularization (alpha_phi)
- Check if singularities are avoidable

### Coverage Oscillating
- Reduce lambda_learning_rate
- Increase momentum
- Increase update_frequency

### Lambda Saturating at Bounds
- Adjust lambda_min/lambda_max
- Check if target_coverage is achievable
- Consider different model architecture

## Theory and Implementation

The adaptive loss policy implements the optimization problem:

```
min_θ E[L(f_θ(x), y)] + λ * (coverage_constraint)
```

Where λ is adjusted to satisfy the coverage constraint as a Lagrange multiplier. This ensures:

1. **Stability**: Gradients remain bounded even at singularities (Mask-REAL)
2. **Flexibility**: Model can learn near poles when beneficial
3. **Control**: User specifies acceptable singularity rate

The implementation is fully compatible with:
- Mask-REAL gradient rule
- TR-Norm layers
- All TR arithmetic operations
- Standard optimizers (SGD, Adam, etc.)
