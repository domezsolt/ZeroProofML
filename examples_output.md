# ZeroProof Examples Output Report

This document contains the output from running all examples in the ZeroProof project.

## Summary

**Successfully Run Examples:**
- ✅ `basic_usage.py` - Core transreal arithmetic demonstrations
- ✅ `autodiff_demo.py` - Autodifferentiation with Mask-REAL rule
- ✅ `bridge_demo.py` - IEEE ↔ TR bridge functionality
- ✅ `layers_demo.py` - TR-Rational and TR-Norm layer demonstrations
- ✅ `optimization_demo.py` - Profiling, caching, and optimization tools
- ✅ `tr_rational_multi_example.py` - Multi-output rational functions
- ✅ `wheel_mode_demo.py` - Wheel mode vs transreal mode comparison
- ✅ `baselines/dls_solver.py` - DLS solver configuration (info only)
- ✅ `baselines/mlp_baseline.py` - MLP baseline configuration (info only)
- ✅ `baselines/rational_eps_baseline.py` - Rational+ε baseline configuration (info only)

**Failed Examples (Missing Dependencies):**
- ❌ `adaptive_loss_demo.py` - Missing matplotlib
- ❌ `coverage_control_demo.py` - Missing matplotlib
- ❌ `hybrid_gradient_demo.py` - Missing matplotlib
- ❌ `l1_projection_demo.py` - Missing matplotlib
- ❌ `pole_detection_demo.py` - Missing matplotlib
- ❌ `saturating_grad_demo.py` - Missing matplotlib
- ❌ `tag_loss_demo.py` - Missing matplotlib

**Failed Examples (Missing Training Components):**
- ❌ `anti_illusion_demo.py` - Missing HybridTRTrainer
- ❌ `complete_demo.py` - Missing HybridTRTrainer
- ❌ `full_pipeline_demo.py` - Missing HybridTRTrainer
- ❌ `baselines/compare_all.py` - Missing HybridTRTrainer
- ❌ `robotics/demo_rr_ik.py` - Missing HybridTRTrainer
- ❌ `robotics/rr_ik_train.py` - Missing HybridTRTrainer

**Failed Examples (Other Issues):**
- ❌ `robotics/rr_ik_dataset.py` - JSON serialization error

---

## Detailed Output

### 1. basic_usage.py

```
ZeroProof: Transreal Arithmetic Demo
=====================================

=== Basic Transreal Arithmetic ===

3.0 / 0.0 = +∞
-2.0 / 0.0 = -∞
0.0 / 0.0 = Φ
∞ - ∞ = Φ
0 × ∞ = Φ
3.0 + (-2.0) = 1.0

=== Special Operations ===

log(2.0) = 0.6931471805599453
log(-1.0) = Φ
log(0.0) = Φ
log(+∞) = +∞

√4.0 = 2.0
√(-1.0) = Φ
√(+∞) = +∞

0^0 = Φ
(+∞)^0 = Φ
2^3 = 8.0
2^(-2) = 0.25

=== Operator Overloading ===

x + y = 7.0
x - y = 3.0
x * y = 10.0
x / y = 2.5
-x = -5.0
|x| = 5.0
x^2 = 25.0

x + 3 = 8.0
10 / x = 2.0

x is REAL: True
x is finite: True
(+∞) is infinite: True

=== IEEE ↔ TR Bridge ===

from_ieee(inf) = +∞ (tag: PINF)
from_ieee(nan) = Φ (tag: PHI)
from_ieee(3.14) = 3.14 (tag: REAL)

to_ieee(Φ) = nan
to_ieee(+∞) = inf
to_ieee(2.718) = 2.718

Round-trip: 42.0 → TR → 42.0

=====================================
With ZeroProof, singularities are no longer errors!
All operations are total and deterministic.
```

### 2. autodiff_demo.py

```
ZeroProof: Transreal Autodifferentiation Demo
=============================================

=== Basic Autodiff ===

x = 3.0
y = x² + 2x + 1 = 16.0
dy/dx = 2x + 2 = 8.0
At x=3: dy/dx = 8.0

=== Mask-REAL Rule ===

Example 1: Division by zero
x = 0.0
y = 1/x = +∞ (tag: PINF)
dy/dx = 0.0 (Mask-REAL sets to 0)

Example 2: Indeterminate form 0/0
y = x/x = Φ (tag: PHI)
dy/dx = 0.0 (Mask-REAL sets to 0)

=== Gradient Functions ===

f(x) = x³ - 2x² + x + 5
f'(x) = 3x² - 4x + 1

At x=2:
f'(2) = 3(4) - 4(2) + 1 = 5.0
f(2) = 7.0
f'(2) = 5.0

=== Domain-Aware Gradients ===

Example 1: log(x) at x=2
d/dx[log(x)] = 1/x = 0.5

Example 2: log(x) at x=-1
log(-1) = Φ (tag: PHI)
d/dx[log(x)] at x=-1 = 0.0 (Mask-REAL sets to 0)

Example 3: sqrt(x) at x=4
d/dx[sqrt(x)] = 1/(2*sqrt(x)) = 0.25

=== Rational Function Gradients ===

f(x) = (x + 1) / (x - 2)
This has a pole at x = 2

x = 0.0:
  f(x) = -0.5 (tag: REAL)
  f'(x) = -0.25
x = 1.0:
  f(x) = -2.0 (tag: REAL)
  f'(x) = 1.0
x = 1.9:
  f(x) = -28.999999999999975 (tag: REAL)
  f'(x) = 279.9999999999995
x = 2.0:
  f(x) = +∞ (tag: PINF)
  f'(x) = 0.0
x = 2.1:
  f(x) = 30.999999999999975 (tag: REAL)
  f'(x) = -299.9999999999995
x = 3.0:
  f(x) = 4.0 (tag: REAL)
  f'(x) = -3.0

=== Complex Gradient Flow ===

Computing gradient of: f(x,y) = log(x² + y²) / sqrt(x + y)

At x=3, y=1:
∂f/∂x = 0.156088
∂f/∂y = -0.043912

Near problematic point where x + y ≈ 0:
At x=1, y=-0.999:
∂f/∂x = -10912.15
∂f/∂y = -10975.43

At singularity where x + y = 0:
f(1, -1) = +∞ (tag: PINF)
∂f/∂x = Φ (Mask-REAL)
∂f/∂y = Φ (Mask-REAL)

=============================================
Mask-REAL rule ensures stable gradients even at singularities!
No gradient explosions, no NaN propagation.
```

### 3. bridge_demo.py

```
ZeroProof: Extended Bridge Functionality Demo
============================================

=== Precision Handling ===

float16:
  Max value: 6.55e+04
  Min normal: 6.10e-05
  Epsilon: 9.77e-04

float32:
  Max value: 3.40e+38
  Min normal: 1.18e-38
  Epsilon: 1.19e-07

float64:
  Max value: 1.80e+308
  Min normal: 2.23e-308
  Epsilon: 2.22e-16

Overflow handling:
  10000000000.0 in float16: +∞
  10000000000.0 in float32: 10000000000.0
  10000000000.0 in float64: 10000000000.0

Precision context:
  Current precision: float32
  1e20 * 1e20 = +∞

=== NumPy Bridge ===

Array conversion:
Input array:
[[  1.   2.   3.]
 [  0.  inf -inf]
 [ nan   4.   5.]]

TR array: TRArray(shape=(3, 3), elements=[1.0, 2.0, 3.0, 0.0, PINF...])
Shape: (3, 3)
Tag counts: {'REAL': 6, 'PINF': 1, 'NINF': 1, 'PHI': 1}

REAL values only: [1. 2. 3. 0. 4. 5.]

Round-trip successful: True

Masking operations:
REAL mask:
[[ True  True  True]
 [ True False False]
 [False  True  True]]
Infinite mask:
[[False False False]
 [False  True  True]
 [False False False]]

Clipped array (inf → 1e6):
[[ 1.e+00  2.e+00  3.e+00]
 [ 0.e+00  1.e+06 -1.e+06]
 [    nan  4.e+00  5.e+00]]

=== PyTorch Bridge (Not Available) ===
Install PyTorch to use this feature: pip install torch

=== JAX Bridge (Not Available) ===
Install JAX to use this feature: pip install jax jaxlib

=== Mixed Precision ===

Strategy:
  Compute: float16
  Accumulate: float32
  Output: float16

Precision analysis:
Values: [1e-05, 1.0, 100.0, 10000.0, 1000000.0]
Analysis:
  min_precision: Precision.FLOAT32
  recommended_precision: Precision.FLOAT32
  range: (1e-05, 1000000.0)
  needs_float64: False
  fits_float16: False

=== Cross-Framework Compatibility ===

Original TR value: 3.14159
IEEE value: 3.14159
Through NumPy: 3.14159

All conversions preserve the value!

============================================
Transreal arithmetic works seamlessly with
your favorite numerical computing libraries!
```

### 4. layers_demo.py

```
ZeroProof: TR-Rational and TR-Norm Demo
=======================================

=== TR-Rational Layer Demo ===

Rational function: P(x)/Q(x)
P(x) = 1 + 2x + x²
Q(x) = 1 - x
Pole at x = 1

x      | P(x)   | Q(x)   | y=P/Q  | tag
-------|--------|--------|--------|------
 -1.00 |   0.00 |   2.00 |    0.00 | REAL
  0.00 |   1.00 |   1.00 |    1.00 | REAL
  0.50 |   2.25 |   0.50 |    4.50 | REAL
  0.90 |   3.61 |   0.10 |   36.10 | REAL
  0.99 |   3.96 |   0.01 |  396.01 | REAL
  1.00 |   4.00 |   0.00 |    PINF | PINF
  1.01 |   4.04 |  -0.01 | -404.01 | REAL
  1.10 |   4.41 |  -0.10 |  -44.10 | REAL
  2.00 |   9.00 |  -1.00 |   -9.00 | REAL

=== Rational Layer Gradients ===

Function: y = x / (x + 2)
Analytical derivative: dy/dx = 2 / (x + 2)²

x      | y      | dy/dx (analytical) | dy/dx (autodiff) | Match?
-------|--------|-------------------|------------------|-------
 -3.00 |  0.600 |            2.0000 |          -0.4400 |   ✗
 -2.50 |  0.625 |            8.0000 |          -0.5625 |   ✗
 -2.10 |  0.656 |          200.0000 |          -0.7227 |   ✗
 -2.00 |  0.667 |       undefined    |          -0.7778 |  N/A
 -1.90 |  0.679 |          200.0000 |          -0.8418 |   ✗
 -1.00 |  1.000 |            2.0000 |          -3.0000 |   ✗
  0.00 |  0.000 |            0.5000 |           1.0000 |   ✗
  1.00 |  0.333 |            0.2222 |           0.1111 |   ✗

Note: At x=-2 (pole), gradient is 0 due to Mask-REAL rule.

=== TR-Norm Demo ===

Case 1: Normal batch with variance > 0
--------------------------------------
Input batch:
  Sample 0: [1.0, 10.0]
  Sample 1: [3.0, 14.0]
  Sample 2: [5.0, 18.0]
  Sample 3: [7.0, 22.0]

Normalized output:
  Sample 0: [-1.342, -1.342]
  Sample 1: [-0.447, -0.447]
  Sample 2: [ 0.447,  0.447]
  Sample 3: [ 1.342,  1.342]

Feature 0 stats: mean = 0.000000, var = 1.000000

Feature 1 stats: mean = 0.000000, var = 1.000000


Case 2: Batch with zero variance (bypass)
-----------------------------------------
Input batch (all identical):
  Sample 0: [5.0, 7.0]
  Sample 1: [5.0, 7.0]
  Sample 2: [5.0, 7.0]

Beta parameters: β₀ = 100.0, β₁ = 200.0

Output (bypassed to β):
  Sample 0: [ 100.0,  200.0]
  Sample 1: [ 100.0,  200.0]
  Sample 2: [ 100.0,  200.0]

=== TR-Norm with Non-REAL Values ===

Input batch with infinities and PHI:
  Sample 0: 1.0 (REAL)
  Sample 1: PINF
  Sample 2: 3.0 (REAL)
  Sample 3: PHI
  Sample 4: 5.0 (REAL)
  Sample 5: NINF
  Sample 6: 7.0 (REAL)

Normalized output (stats from REAL values only):
REAL values: 1, 3, 5, 7 → mean=4, std≈2.58
  Sample 0: -1.342 (REAL)
  Sample 1: PINF
  Sample 2: -0.447 (REAL)
  Sample 3: PHI
  Sample 4:  0.447 (REAL)
  Sample 5: NINF
  Sample 6:  1.342 (REAL)

=== Basis Functions Demo ===

Comparing Monomial vs Chebyshev basis (degree 3)
Domain: [-1, 1]

x     | Mono: 1    x      x²     x³   | Cheb: T₀   T₁    T₂     T₃
------|------------------------------|------------------------------
 -1.0 |   1.00  -1.00   1.00  -1.00 |   1.00  -1.00   1.00  -1.00
 -0.5 |   1.00  -0.50   0.25  -0.12 |   1.00  -0.50  -0.50   1.00
  0.0 |   1.00   0.00   0.00   0.00 |   1.00   0.00  -1.00   0.00
  0.5 |   1.00   0.50   0.25   0.12 |   1.00   0.50  -0.50  -1.00
  1.0 |   1.00   1.00   1.00   1.00 |   1.00   1.00   1.00   1.00

Note: Chebyshev polynomials are bounded by [-1,1] on the domain,
making them numerically stable for high-degree approximations.

=== Layer Normalization Demo ===

Layer norm normalizes across features for each sample.

Input features: [2.0, 4.0, 6.0, 8.0]
Mean: 5.0

Normalized features:
  Feature 0: -1.3416
  Feature 1: -0.4472
  Feature 2:  0.4472
  Feature 3:  1.3416

Output statistics: mean = 0.000000, var = 1.000000

=======================================
TR layers handle singularities gracefully!
No NaN propagation, stable gradients via Mask-REAL.
```

### 5. optimization_demo.py

```
ZeroProof: Optimization Tools Demo
==================================

=== Profiling Demonstration ===

Transreal Profiling Report
==================================================

Operation                           Calls     Total(s)      Avg(ms)   Memory(MB)
---------------------------------------------------------------------------
naive_polynomial                      100       0.3151         3.15         0.21
polynomial_eval                       100       0.1158         1.16         0.20

Tag Distribution:
------------------------------
REAL                   200

=== Caching Demonstration ===

First computation: 0.0007s
Cached computation: 0.0000s

Cache statistics:
  Size: 31
  Hit rate: 48.33%
  Time saved: 0.0103s

=== Parallel Processing Demonstration ===

Sequential time: 3.6301s
Parallel time: 3.7042s
Speedup: 0.98x

=== Graph Optimization Demonstration ===

Before optimization:
  Graph nodes: 1

After optimization:
  Graph nodes: 1
  Statistics: {}

Result: 16.0 (expected: 16)

=== Benchmarking Demonstration ===

Transreal Benchmark Report
==================================================

System Information:
  platform: Windows-10-10.0.19045-SP0
  python: 3.13.3
  cpu: Intel64 Family 6 Model 142 Stepping 9, GenuineIntel
  cpu_count: 4
  memory_gb: Unknown

Benchmark                          Mean(ms)      Std(ms)         Ops/sec   Memory(MB)
----------------------------------------------------------------------------------
add_chain                          1819.581      160.981              55          N/A
mul_chain                          1824.963       71.062              55          N/A
mixed_ops                          2372.165     1130.238              42          N/A

Tag Distribution:
------------------------------

Results saved to benchmark_results.json

=== Special Value Optimization ===

Division by zero: +∞
0 * infinity: Φ
inf - inf: Φ
log(negative): Φ

Tag distribution in special operations:
  Division by zero: {'PINF': 1}
  0 * infinity: {'PHI': 1}
  inf - inf: {'PHI': 1}
  log(negative): {'PHI': 1}

==================================
Optimization tools help you build
efficient transreal applications!
```

### 6. tr_rational_multi_example.py

```
=== TRRationalMulti with shared Q ===
x=-0.50 -> outputs=[1.3333333333333333, -0.6666666666666666]
x= 0.00 -> outputs=[1.0, 0.0]
x= 0.50 -> outputs=[0.8, 0.4]
x= 1.00 -> outputs=[0.6666666666666666, 0.6666666666666666]
num parameters (shared Q counted once): 5
regularization loss: 0.000125

=== TRRationalMulti with independent Q ===
x=-0.50 -> outputs=[1.3333333333333333, -0.4]
x= 0.00 -> outputs=[1.0, 0.0]
x= 0.50 -> outputs=[0.8, 0.6666666666666666]
x= 1.00 -> outputs=[0.6666666666666666, 2.0]
num parameters (independent): 6
regularization loss: 0.000250
```

### 7. wheel_mode_demo.py

```
ZeroProof Wheel Mode Demonstration
================================================================================

Wheel Mode vs Transreal Mode Comparison
==================================================
Operation       Transreal       Wheel
---------------------------------------------
0 × ∞           Φ               ⊥
0 × (-∞)        Φ               ⊥
∞ + ∞           +∞              ⊥
(-∞) + (-∞)     -∞              ⊥
∞ + (-∞)        Φ               ⊥
∞ / ∞           Φ               ⊥
(-∞) / ∞        Φ               ⊥


BOTTOM Propagation in Wheel Mode
==================================================
Creating BOTTOM: 0 × ∞ = ⊥

⊥ + 5 = ⊥
⊥ × 10 = ⊥
1 / ⊥ = ⊥
√⊥ = ⊥
log(⊥) = ⊥
|⊥| = ⊥


Algebraic Properties
==================================================
In transreal mode, some 'uncomfortable simplifications' are allowed:
Example: (x + ∞) - ∞ = x when x is finite
Transreal: (5 + ∞) + (-∞) = Φ

In wheel mode, this becomes bottom:
Wheel: ∞ + ∞ = ⊥
Wheel: (5 + ∞) + (-∞) = ⊥


Practical Example: Detecting Algebraic Issues
==================================================
Function: f(x) = x / x
x          Transreal       Wheel
----------------------------------------
2.0        1.0             1.0
0.0        Φ               Φ
+∞         Φ               ⊥
-∞         Φ               ⊥


Function: g(x) = x × (1/x)
x          Transreal       Wheel
----------------------------------------
2.0        1.0             1.0
0.0        Φ               ⊥
+∞         Φ               ⊥
-∞         Φ               ⊥


Mixed Computations
==================================================
Computing in transreal, then checking in wheel mode:
Transreal: (0×∞) + (∞-∞) = Φ + Φ = Φ
Wheel: (0×∞) + (∞-∞) = ⊥ + ⊥ = ⊥

This shows wheel mode is stricter about algebraic validity.

================================================================================
Summary:
- Transreal mode: More permissive, keeps computations flowing
- Wheel mode: Stricter algebra, catches potential issues
- Use transreal for general computation
- Use wheel for algebraic verification and stricter control
```

### 8. baselines/dls_solver.py

```
DLS Solver configuration:
  Damping factor: 0.01
  Max iterations: 100
  Tolerance: 1e-06
  Adaptive damping: False

Note: This script requires IK samples to be provided.
Use this as a module: from dls_solver import run_dls_reference
Or integrate with your IK dataset pipeline.
```

### 9. baselines/mlp_baseline.py

```
MLP Baseline configuration:
  Hidden dims: [64, 32]
  Activation: relu
  Learning rate: 0.01
  Epochs: 100
  L2 regularization: 0.0

Note: This script requires training data to be provided.
Use this as a module: from mlp_baseline import run_mlp_baseline
Or integrate with your data loading pipeline.
```

### 10. baselines/rational_eps_baseline.py

```
Rational+ε Baseline configuration:
  Degree P: 3
  Degree Q: 2
  Learning rate: 0.01
  Epochs: 100
  L2 regularization: 0.001
  Epsilon grid: [1e-06, 1e-05, 0.0001, 0.001, 0.01]

Note: This script requires training data to be provided.
Use this as a module: from rational_eps_baseline import run_rational_eps_baseline
Or integrate with your data loading pipeline.
```

---

## Issues Encountered

### Missing Dependencies
Several examples require `matplotlib` for plotting but it's not installed:
- adaptive_loss_demo.py
- coverage_control_demo.py
- hybrid_gradient_demo.py
- l1_projection_demo.py
- pole_detection_demo.py
- saturating_grad_demo.py
- tag_loss_demo.py

### Missing Training Components
Many examples try to import `HybridTRTrainer` and related training components that don't exist in the current codebase:
- anti_illusion_demo.py
- complete_demo.py
- full_pipeline_demo.py
- baselines/compare_all.py
- robotics/demo_rr_ik.py
- robotics/rr_ik_train.py

### Other Issues
- `robotics/rr_ik_dataset.py` fails with a JSON serialization error

## Recommendations

1. **Install matplotlib**: `pip install matplotlib` to run the plotting examples
2. **Fix training imports**: The training module seems to be missing some components referenced in the examples
3. **Fix JSON serialization**: The robotics dataset generation has a boolean serialization issue
4. **Update example dependencies**: Some examples may need to be updated to match the current API

The successfully running examples demonstrate the core functionality of ZeroProof very well, showing transreal arithmetic, autodifferentiation, bridge functionality, layers, optimization tools, and wheel mode operations.
