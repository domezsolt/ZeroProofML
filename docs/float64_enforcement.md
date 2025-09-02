# Float64 Enforcement Implementation

## Overview
This document describes the implementation of float64 enforcement in ZeroProof, ensuring maximum precision for transreal arithmetic operations by default.

## Implementation Details

### 1. Precision Configuration Module (`zeroproof/core/precision_config.py`)
Created a comprehensive precision management system with:

- **`PrecisionMode` enum**: Supports FLOAT16, FLOAT32, and FLOAT64
- **`PrecisionConfig` class**: Global precision management with:
  - Default precision: float64 (as specified)
  - Methods to get/set precision
  - Precision enforcement for all numeric values
  - Overflow detection for current precision
  - Machine epsilon, max, and min value queries
- **`precision_context`**: Context manager for temporary precision changes

### 2. Core Updates

#### TRScalar (`zeroproof/core/tr_scalar.py`)
- Updated `__init__` to enforce precision on REAL values
- Modified `real()` factory to:
  - Accept Union[float, int, np.floating]
  - Enforce precision before creating TRScalar
  - Check for overflow and return appropriate infinity

#### Arithmetic Operations (`zeroproof/core/tr_ops.py`)
- All operations now enforce precision on results:
  - `tr_add`: Result enforced to current precision
  - `tr_mul`: Result enforced to current precision
  - `tr_div`: Result enforced to current precision
  - `tr_log`: Result enforced to current precision
  - `tr_sqrt`: Result enforced to current precision
- Overflow detection added to return infinities when appropriate

#### IEEE Bridge (`zeroproof/bridge/ieee_tr.py`)
- `from_ieee`: Enforces precision when converting to TR
- `to_ieee`: Maintains precision when converting from TR

### 3. API Additions

```python
# Set global precision
PrecisionConfig.set_precision(PrecisionMode.FLOAT32)
# or
PrecisionConfig.set_precision('float32')

# Temporary precision change
with precision_context('float16'):
    x = real(1.0)  # Uses float16

# Query precision info
eps = PrecisionConfig.get_epsilon()  # Machine epsilon
max_val = PrecisionConfig.get_max()  # Maximum value
min_val = PrecisionConfig.get_min()  # Minimum positive value

# Check overflow
if PrecisionConfig.check_overflow(value):
    # Value would overflow in current precision
```

### 4. Key Features

1. **Default Float64**: All operations use float64 by default for maximum precision
2. **Overflow Handling**: Automatic conversion to ±∞ when values overflow
3. **Precision Preservation**: All intermediate results maintain the specified precision
4. **Optional Downgrade**: Can use float32 or float16 for performance when needed
5. **Seamless Integration**: Works transparently with existing code

### 5. Testing

Created comprehensive tests in `tests/unit/test_precision.py`:
- Default precision verification
- Precision switching and context management
- Overflow detection for different precisions
- Precision preservation through operations
- Edge cases (subnormals, epsilon, limits)

## Usage Examples

```python
import zeroproof as zp

# Default: float64 precision
x = zp.real(1.0 / 3.0)  # Full float64 precision

# Temporary float32 for performance
with zp.precision_context('float32'):
    # All operations use float32
    y = zp.real(2.0)
    z = zp.tr_mul(x, y)  # Result in float32

# Check current precision
mode = zp.PrecisionConfig.get_precision()
print(f"Current precision: {mode.bits} bits")

# Handle overflow gracefully
with zp.precision_context('float16'):
    # Float16 max is ~65504
    big = zp.real(100000.0)  # Automatically returns pinf()
```

## Benefits

1. **Numerical Stability**: Float64 by default ensures maximum precision
2. **Flexibility**: Can downgrade precision when needed for performance
3. **Safety**: Overflow automatically produces infinity tags (no NaN)
4. **Transparency**: Works with all existing TR operations
5. **Consistency**: All operations maintain the specified precision

## Next Steps

The remaining features to implement are:
1. Adaptive λ_rej (loss policy with Lagrange multipliers)
2. Saturating-grad (alternative to Mask-REAL)
3. Wheel mode (optional stricter algebra)
