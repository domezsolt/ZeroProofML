# Wheel Mode Guide

## Overview

ZeroProof provides two arithmetic modes:

1. **Transreal Mode** (default): Standard transreal arithmetic with PHI for indeterminate forms
2. **Wheel Mode**: Stricter algebra with bottom (⊥) for certain operations

Wheel mode implements a projective wheel algebra where certain operations that produce PHI in transreal mode instead produce a bottom element (⊥), providing stricter algebraic control.

## Key Differences

| Operation | Transreal Mode | Wheel Mode |
|-----------|----------------|------------|
| 0 × ∞     | PHI (Φ)        | BOTTOM (⊥) |
| ∞ + ∞     | ∞              | BOTTOM (⊥) |
| -∞ + -∞   | -∞             | BOTTOM (⊥) |
| ∞ - ∞     | PHI (Φ)        | BOTTOM (⊥) |
| ∞ / ∞     | PHI (Φ)        | BOTTOM (⊥) |

## Basic Usage

### Switching Modes

```python
import zeroproof as zp

# Check current mode
from zeroproof.core import WheelModeConfig
print(WheelModeConfig.is_transreal())  # True by default

# Switch to wheel mode globally
zp.use_wheel()

# Operations now use wheel algebra
result = zp.tr_mul(zp.real(0.0), zp.pinf())  # Returns BOTTOM

# Switch back to transreal
zp.use_transreal()
```

### Context Manager

For temporary mode changes:

```python
# Normal transreal mode
result1 = zp.tr_mul(zp.real(0.0), zp.pinf())  # PHI

# Temporary wheel mode
with zp.wheel_mode():
    result2 = zp.tr_mul(zp.real(0.0), zp.pinf())  # BOTTOM

# Back to transreal
result3 = zp.tr_mul(zp.real(0.0), zp.pinf())  # PHI
```

### Creating Bottom Values

```python
# Direct creation
b = zp.bottom()

# From operations in wheel mode
with zp.wheel_mode():
    b1 = zp.tr_mul(zp.real(0.0), zp.pinf())  # 0 × ∞ = ⊥
    b2 = zp.tr_add(zp.pinf(), zp.pinf())     # ∞ + ∞ = ⊥
```

## Bottom Propagation

In wheel mode, BOTTOM propagates through all operations:

```python
with zp.wheel_mode():
    b = zp.bottom()
    
    # Arithmetic operations
    print(b + zp.real(5))    # ⊥
    print(b * zp.real(10))   # ⊥
    print(zp.real(1) / b)    # ⊥
    
    # Unary operations
    print(-b)                # ⊥
    print(abs(b))            # ⊥
    print(zp.tr_sqrt(b))     # ⊥
```

## Design Philosophy

### Transreal Mode (Default)
- **Goal**: Keep computations flowing
- **Approach**: Use PHI for genuine indeterminate forms
- **Example**: ∞ + ∞ = ∞ (models limit-like intuition)

### Wheel Mode
- **Goal**: Strict algebraic control
- **Approach**: Use BOTTOM for operations that violate algebraic laws
- **Example**: ∞ + ∞ = ⊥ (prevents "uncomfortable simplifications")

## When to Use Each Mode

### Use Transreal Mode When:
- Doing general numerical computation
- Training neural networks
- You want maximum computational flow
- Working with limits and approximations

### Use Wheel Mode When:
- Verifying algebraic properties
- Detecting potential mathematical issues
- Implementing formal algebraic systems
- You need stricter error propagation

## Examples

### Detecting Algebraic Issues

```python
def potentially_bad_simplification(x):
    """Function that might hide algebraic issues."""
    # Mathematically: (x + ∞) - ∞ should equal x
    # But this assumes commutativity with infinities
    return zp.tr_add(zp.tr_add(x, zp.pinf()), zp.ninf())

# In transreal mode - allows the simplification
zp.use_transreal()
result_tr = potentially_bad_simplification(zp.real(5))
print(f"Transreal: (5 + ∞) - ∞ = {result_tr}")  # PHI

# In wheel mode - catches the issue
zp.use_wheel()
result_wheel = potentially_bad_simplification(zp.real(5))
print(f"Wheel: (5 + ∞) - ∞ = {result_wheel}")  # BOTTOM
```

### Algebraic Verification

```python
def verify_identity(f, g, test_values):
    """Check if two functions are algebraically equivalent."""
    issues = []
    
    with zp.wheel_mode():
        for x in test_values:
            f_result = f(x)
            g_result = g(x)
            
            # If either gives BOTTOM, there's an algebraic issue
            if f_result.tag == zp.TRTag.BOTTOM:
                issues.append(f"f undefined at {x}")
            if g_result.tag == zp.TRTag.BOTTOM:
                issues.append(f"g undefined at {x}")
    
    return issues

# Example: Are these equivalent?
f = lambda x: zp.tr_div(x, x)  # x/x
g = lambda x: zp.real(1.0)     # 1

issues = verify_identity(f, g, [zp.real(0), zp.pinf()])
print("Algebraic issues:", issues)
```

## Implementation Notes

### No Mode Mixing
Operations should not mix values computed in different modes:

```python
# DON'T DO THIS
zp.use_transreal()
a = zp.tr_mul(zp.real(0), zp.pinf())  # PHI

zp.use_wheel()
b = zp.tr_mul(zp.real(0), zp.pinf())  # BOTTOM

# Mixing modes - undefined behavior
result = zp.tr_add(a, b)  # Don't mix PHI and BOTTOM!
```

### IEEE Bridge
BOTTOM converts to NaN when exporting to IEEE:

```python
with zp.wheel_mode():
    b = zp.bottom()
    ieee_val = zp.to_ieee(b)  # float('nan')
```

### Type Checking

```python
# Check for bottom values
if zp.is_bottom(value):
    print("Hit bottom - algebraic issue detected")
```

## Theory Background

Wheel mode implements a **projective wheel algebra** where:

- The wheel has points: 0, finite numbers, ∞, and ⊥
- Control laws prevent problematic simplifications:
  - 0 × ∞ = ⊥ (not 0 or ∞)
  - ∞ + ∞ = ⊥ (not 2∞)
- ⊥ is absorptive: x ⊕ ⊥ = ⊥ for any operation ⊕

This provides an **initial algebra** for the wheel equations, making it suitable for algebraic verification and formal reasoning.

## Best Practices

1. **Default to Transreal**: Use transreal mode unless you specifically need wheel strictness
2. **Don't Mix Modes**: Complete computations in one mode before switching
3. **Use Context Managers**: For temporary mode changes
4. **Document Mode Usage**: Make it clear when wheel mode is required
5. **Test Both Modes**: For critical computations, verify behavior in both modes

## Comparison with Other Systems

- **IEEE-754**: Has NaN but allows NaN != NaN; wheel has ⊥ = ⊥
- **Interval Arithmetic**: Tracks bounds; wheel tracks algebraic validity
- **Automatic Differentiation**: Orthogonal - both modes work with AD
- **Symbolic Systems**: Wheel mode closer to CAS behavior

## API Reference

### Mode Control
- `use_transreal()`: Switch to transreal mode
- `use_wheel()`: Switch to wheel mode
- `wheel_mode()`: Context manager for wheel mode
- `arithmetic_mode(mode)`: Context manager with explicit mode

### Constants
- `bottom()`: Create bottom element (⊥)
- `is_bottom(x)`: Check if value is bottom

### Configuration
- `ArithmeticMode.TRANSREAL`: Transreal mode enum
- `ArithmeticMode.WHEEL`: Wheel mode enum
- `WheelModeConfig`: Global mode configuration
