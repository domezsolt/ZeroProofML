# ZeroProof

[![Tests (Ubuntu)](https://img.shields.io/github/actions/workflow/status/domezsolt/zeroproofml/tests-ubuntu.yml?branch=main&label=Tests%20(Ubuntu))](https://github.com/domezsolt/zeroproofml/actions/workflows/tests-ubuntu.yml)
[![Tests (Windows)](https://img.shields.io/github/actions/workflow/status/domezsolt/zeroproofml/tests-windows.yml?branch=main&label=Tests%20(Windows))](https://github.com/domezsolt/zeroproofml/actions/workflows/tests-windows.yml)
[![Lint](https://img.shields.io/github/actions/workflow/status/domezsolt/zeroproofml/lint.yml?branch=main&label=Lint)](https://github.com/domezsolt/zeroproofml/actions/workflows/lint.yml)
[![Coverage](https://img.shields.io/github/actions/workflow/status/domezsolt/zeroproofml/coverage.yml?branch=main&label=Coverage)](https://github.com/domezsolt/zeroproofml/actions/workflows/coverage.yml)
[![Import Smoke Test](https://img.shields.io/github/actions/workflow/status/domezsolt/zeroproofml/import-smoke.yml?branch=main&label=Import%20Smoke%20Test)](https://github.com/domezsolt/zeroproofml/actions/workflows/import-smoke.yml)
[![Property Test Suite](https://img.shields.io/github/actions/workflow/status/domezsolt/zeroproofml/property.yml?branch=main&label=Property%20Test%20Suite)](https://github.com/domezsolt/zeroproofml/actions/workflows/property.yml)
[![Benchmarks (mini suite)](https://img.shields.io/github/actions/workflow/status/domezsolt/zeroproofml/benchmarks.yml?branch=main&label=Benchmarks%20(mini%20suite))](https://github.com/domezsolt/zeroproofml/actions/workflows/benchmarks.yml)
[![Determinism & Safety](https://img.shields.io/github/actions/workflow/status/domezsolt/zeroproofml/determinism.yml?branch=main&label=Determinism%20%26%20Safety)](https://github.com/domezsolt/zeroproofml/actions/workflows/determinism.yml)
[![Framework Integration](https://img.shields.io/github/actions/workflow/status/domezsolt/zeroproofml/test-frameworks.yml?branch=main&label=Framework%20Integration)](https://github.com/domezsolt/zeroproofml/actions/workflows/test-frameworks.yml)
[![E2E No‑NaN](https://img.shields.io/badge/e2e%20no‑NaN-✔-brightgreen)](#)

<div align="center">

 

**Transreal arithmetic for stable machine learning without epsilon hacks**

ZeroProof replaces epsilon hacks with a principled number system that makes
all operations total (no NaNs), keeps gradients stable at singularities, and
stays deterministic by design. Drop it into your stack to train near poles
without fragile thresholds.

[Getting Started](docs/topics/00_getting_started.md) | [Docs Index](docs/index.md) | [Examples](examples/)

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

See also: docs/topics/00_getting_started.md for an end‑to‑end sketch.

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

# Use in neural networks (layer forward)
from zeroproof.layers import TRRational, ChebyshevBasis
from zeroproof.autodiff.tr_node import TRNode
x_node = TRNode.constant(zp.real(0.3))
layer = zp.layers.TRRational(d_p=3, d_q=2, basis=ChebyshevBasis())
y_node, tag = layer.forward(x_node)
```

## Installation

ZeroProof targets Python 3.9+ and keeps heavy backends optional.

Until the first PyPI release, install from source:

```bash
git clone https://github.com/domezsolt/zeroproofml.git
cd zeroproof

# Minimal install (no optional backends)
pip install -e .

# Development (tests, linters, typing)
pip install -e .[dev]
# or install tooling from requirements file
# pip install -r requirements-dev.txt && pip install -e .

# With PyTorch support
pip install -e .[torch]

# With JAX support
pip install -e .[jax]

# All optional features
pip install -e .[all]
```

Once published on PyPI, installation will be as simple as:

```bash
pip install zeroproof
pip install zeroproof[torch]
pip install zeroproof[jax]
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

### Result Verification (Defaults)

Use the verification helper to enforce paper‑parity metrics on seed runs:

```bash
# Verify across seeds (90th percentile aggregation) for the 2R RR dataset
python3 scripts/verify_results.py \
  --path results/robotics/paper_suite \
  --method "ZeroProofML-Full" \
  --max-ple 0.30 \
  --max-b0 0.010 \
  --max-b1 0.010 \
  --percentile 90 \
  --require-nonempty-b03

# Strict per-run bounds (no percentile aggregation)
python3 scripts/verify_results.py \
  --glob 'results/robotics/paper_suite/seed_*/comprehensive_comparison.json' \
  --method "ZeroProofML-Full" \
  --max-ple 0.30 \
  --max-b0 0.010 \
  --max-b1 0.010 \
  --no-percentile \
  --require-nonempty-b03
```

Notes:
- Recommended defaults above target the CPU‑friendly RR 2R suite in this repo.
- Thresholds are dataset‑dependent; tighten/relax as appropriate for your setup.
- The guardrail `--require-nonempty-b03` promotes empty near‑pole buckets (B0–B3) to failures.

### Debug Logging

Enable console logging and capture structured training metrics for
troubleshooting and reproducibility:

```python
import logging
from zeroproof.utils.logging import StructuredLogger

logging.basicConfig(level=logging.INFO)
logger = StructuredLogger(run_dir="runs/demo")
# ... log per‑step metrics and save JSON/CSV/summary
```

See docs/debug_logging.md for details, per‑module log levels, and field
reference.

## Advanced Features

### TR-Rational Layers

Learn rational functions P(x)/Q(x) that can model poles:

```python
from zeroproof.layers import ChebyshevBasis

layer = zp.layers.TRRational(
    d_p=4,                 # Numerator degree
    d_q=3,                 # Denominator degree
    basis=ChebyshevBasis(),# Polynomial basis
    lambda_rej=1.0,        # Penalty for non-REAL outputs
)
```

### Policy & Guard Bands (TRPolicy)

ZeroProof provides a central TR policy that defines guard bands around poles and optional deterministic reductions:

```python
from zeroproof.policy import TRPolicy, TRPolicyConfig

# Enable ULP‑scaled thresholds with pairwise reductions
TRPolicyConfig.set_policy(TRPolicy(
    tau_Q_on=0.0, tau_Q_off=0.0,  # resolved automatically if desired
    tau_P_on=0.0, tau_P_off=0.0,
    keep_signed_zero=True,
    deterministic_reduction=True,
))

# Model‑aware thresholds (recommended)
from zeroproof.training import enable_policy_from_model
enable_policy_from_model(model, ulp_scale=4.0)
```

Deterministic reductions (pairwise trees) are honored throughout: P/Q in TR‑Rational, TR‑Norm mean/var, TR‑softmax normalization, dense sums, and regularizers when `deterministic_reduction=True`.

### Epsilon-Free Normalization

Batch normalization without epsilon hacks:

```python
norm = zp.layers.TRNorm(num_features=128)
# Handles zero-variance features deterministically
# No more eps=1e-5 parameters!
```

### TR-Softmax (Rational Surrogate)

ZeroProof includes a TR‑safe softmax built from rational operations (no exp). It preserves autodiff paths and avoids NaN/Inf propagation.

```python
from zeroproof.layers import tr_softmax
from zeroproof.autodiff import TRNode

logits = [TRNode.constant(zp.real(0.0)),
          TRNode.constant(zp.real(1.5)),
          TRNode.constant(zp.real(-0.5))]

probs = tr_softmax(logits)  # List[TRNode], sums to 1 in REAL regions
```

Policy: one‑hot on +∞ (optional)

By default, if a logit is `+∞`, the shift‑by‑max can yield non‑REAL tags in the surrogate (still TR‑safe). To force a deterministic one‑hot distribution when any `+∞` is present, enable the policy toggle:

```python
from zeroproof.policy import TRPolicy, TRPolicyConfig

pol = TRPolicy(softmax_one_hot_infinity=True)
TRPolicyConfig.set_policy(pol)

# Now tr_softmax([…, +∞, …]) returns one‑hot at the first +∞ index
```

Internals

- The surrogate uses a monotone rational decay after a max‑shift; it is stable on extreme logits and compatible with Mask‑REAL/Hybrid modes.

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

Browse runnable scripts in `examples/`:
- examples/complete_demo.py — end‑to‑end showcase
- examples/pole_1d_tutorial.py — 1D pole tutorial (y = 1/(x−a))
- examples/hybrid_gradient_demo.py — Hybrid gradient schedule
- examples/coverage_control_demo.py — adaptive coverage (λ_rej)
- examples/layers_demo.py — layer basics (TR‑Rational, TR‑Norm)
- examples/evaluator_cli.py — run integrated evaluator on a simple model
- examples/robotics/rr_ik_quick.py — quick RR IK run (dataset + baselines)

### Plot Styles

ZeroProof ships light/dark Matplotlib styles. Use them via:

```python
from zeroproof.utils.plotting import use_zeroproof_style
use_zeroproof_style("light")  # or "dark"
```

### Robotics IK (RR arm) — Parity Runner

Generate a dataset with bucket metadata and run all baselines on identical splits. Quick mode stratifies the test subset by |det(J)|≈|sin θ2| and aligns DLS to the same subset.

```bash
# Dataset
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

# Parity runner (quick)
python experiments/robotics/run_all.py \
  --dataset data/rr_ik_dataset.json \
  --profile quick \
  --models tr_basic tr_full rational_eps mlp dls \
  --max_train 2000 --max_test 500 \
  --output_dir results/robotics/quick_run
```

Outputs include bucketed MSE (with counts) and 2D pole metrics; a compact console table and comprehensive JSON are saved under the chosen output directory.

## Documentation

- Start here: docs/topics/00_getting_started.md
- Topics index: docs/index.md
- Concepts: docs/topics/01_overview.md, docs/topics/02_foundations.md, docs/topics/03_autodiff_modes.md
- Layers: docs/topics/04_layers.md, docs/layers.md
- Training: docs/topics/05_training_policies.md, docs/adaptive_loss_guide.md
- Sampling: docs/topics/06_sampling_curriculum.md
- Evaluation: docs/topics/07_evaluation_metrics.md
- How‑To Checklists: docs/topics/08_howto_checklists.md
- Benchmarks: docs/benchmarks.md

## Backend Status

| Backend | Status        | Extra           | Minimal Version(s)          |
|---------|---------------|-----------------|-----------------------------|
| NumPy   | Supported     | —               | —                           |
| PyTorch | Supported     | `[torch]`       | torch >= 1.12               |
| JAX     | Experimental  | `[jax]`         | jax >= 0.4.14, jaxlib >= 0.4.14 |

## System Requirements

- Python: 3.9 or newer (tested on 3.9–3.13)
- CPU: Any 64‑bit CPU; GPU not required
- RAM: 1–2 GB sufficient for examples; robotics demos may benefit from 4+ GB
- Disk: < 200 MB for repository checkout; datasets generated on‑the‑fly unless specified

## Comparison (When to Use ZeroProof)

- Standard autodiff with epsilons
  - Pros: widely supported, fast
  - Cons: requires arbitrary epsilon thresholds; NaNs/Inf can propagate; behavior near poles is non‑deterministic
- Symbolic/interval approaches
  - Pros: rigorous bounds, formal guarantees
  - Cons: heavy/slow for ML workloads; integration friction
- ZeroProof (Transreal arithmetic)
  - Pros: total operations (no exceptions), deterministic tag semantics, stable gradients (Mask‑REAL/Saturating/Hybrid), epsilon‑free normalization
  - Cons: new number system and policies to learn; some backend features are experimental

## Documentation Hosting

- Short‑form docs are in `docs/` within this repo.
- Full documentation site to be hosted on GitHub Pages/ReadTheDocs in a future release.

## Reproducibility & Bench Metrics

- Set global seeds (Python/NumPy/PyTorch) with `zeroproof.utils.seeding.set_global_seed(seed)`.
- Robotics datasets save `metadata.seed` and bucket metadata (`bucket_edges`, `train_bucket_counts`, `test_bucket_counts`).
- Hybrid training records per‑epoch timings: `avg_step_ms`, `data_time_ms`, `optim_time_ms`, `batches` (see `bench_history` in training summaries). Logging cadence is controlled by `log_interval` (CLI `--log_every`).

## Gradient Checking (Utilities)

- Use `zeroproof.autodiff.grad_funcs.check_gradient(func, x)` for finite‑difference checks on REAL paths.
- `tr_grad` and `tr_value_and_grad` lift scalar functions into gradient evaluators.
- See tests under `tests/unit/test_tr_autodiff.py` and the robotics gradcheck at `tests/unit/test_robotics_gradcheck.py`.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Clone the repository
git clone https://github.com/domezsolt/zeroproofml.git
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
### Packed NumPy Arrays (bit‑packed tags)

For memory efficiency, you can use a packed representation of TR arrays with bit‑packed tag masks (REAL, ±∞) and PHI implied by the remainder:

```python
import numpy as np
from zeroproof.bridge import from_numpy_packed, to_numpy, TRArrayPacked

arr = np.array([1.0, np.inf, -np.inf, np.nan])
packed = from_numpy_packed(arr)      # TRArrayPacked
restored = to_numpy(packed)          # IEEE round‑trip
```

This complements the standard `TRArray` struct‑of‑arrays layout for values+tags.

### Second‑Order Safeguards & Contracts

ZeroProof logs a conservative curvature bound and layer contract during training:

- Contract: `{B_k, H_k, G_max, H_max, depth_hint}` (published by layers)
- Curvature bound: logged per‑batch/epoch and included in summaries

Optional contract‑safe LR clamp (off by default) can be enabled in the IK runner via CLI:

```bash
python examples/robotics/rr_ik_train.py \
  --dataset data/rr_ik_dataset.json --model tr_rat \
  --use_contract_safe_lr --contract_c 1.0 --loss_smoothness_beta 1.0 \
  --output_dir runs/ik_experiment
```

Training summaries/results include policy and safeguard metrics: `flip_rate`, `saturating_ratio`, `tau_q_on/off`, `q_min_epoch`, `curvature_bound`, and `layer_contract`.

### Plot Training Curves

Use the compact utility to visualize flip rate, thresholds, and curvature over epochs:

```bash
python scripts/plot_training_curves.py \
  --results runs/ik_experiment/results_tr_rat.json \
    --outdir runs/ik_experiment
```

## Dependency Policy

- Runtime dependencies: kept minimal and specified as version ranges to allow
  compatibility with distro and cloud environments.
- Optional backends (PyTorch/JAX): provided via extras; not required for
  `import zeroproof`.
- Development tooling (tests/linters/typing): grouped under the `dev` extra.
  Use `pip install -e .[dev]`.
  
We aim to test Python 3.9–3.13 in CI. If an upper Python version shows issues,
we will document it in the changelog and README until resolved.

## Backward Compatibility & Deprecation

- Versioning: We follow SemVer pre‑1.0 conventions. Patch releases in the
  `0.1.x` line aim to be backward compatible; breaking changes, if any, land in
  the next minor `0.x` release with notes in the changelog.
- Public API: The stable surface is what `import zeroproof as zp` re‑exports
  from `zeroproof/__init__.py` (types, factory functions, core ops, modes,
  selected utilities, and protocols). Internal modules may change without
  notice.
- Deprecations: We announce deprecations in the changelog and README. Where
  feasible, runtime warnings are issued before removal.

## Troubleshooting

- Externally managed environment (PEP 668): If `pip install -e .` fails with
  “externally managed environment”, create a virtual environment:
  `python3 -m venv .venv && . .venv/bin/activate && pip install -e .[dev]`.
- Optional backends missing: Importing `zeroproof` does not require torch/jax.
  Backend‑specific features need the corresponding extra: `pip install -e .[torch]`
  or `pip install -e .[jax]`.
- Python version: Ensure Python ≥ 3.9. Check with `python -V`.
- Apple Silicon (JAX): Some JAX/JAXLIB versions may not be available for your
  platform. Consult JAX’s installation guide for platform‑specific wheels.
- NumPy bridge: If `from_numpy` is `None`, NumPy isn’t available in your env.
  Install it explicitly or use the IEEE bridge (`from_ieee`/`to_ieee`).
- CI import failures: Verify you can `python -c "import zeroproof"` in a fresh
  environment without extras before enabling optional backends.
 - RecursionError during training: Very deep linear graphs (e.g., summing many
   per‑sample losses) can overflow recursion limits during backprop. ZeroProof’s
   trainer aggregates per‑sample losses with a pairwise (tree) reduction to bound
   graph depth; if writing custom loops, use a balanced sum for loss aggregation.

## FAQ

- What does `1/0`, `-1/0`, and `0/0` return?
  - `zp.real(1)/zp.real(0)` → `PINF` (+∞)
  - `zp.real(-1)/zp.real(0)` → `NINF` (−∞)
  - `zp.real(0)/zp.real(0)` → `PHI` (nullity)
  - Examples follow TR division tables; no exceptions are raised.

- What about domain errors like `log(x≤0)` or `sqrt(x<0)`?
  - `zp.tr_log(zp.real(0.0))` → `PHI`
  - `zp.tr_log(zp.pinf())` → `PINF`
  - `zp.tr_sqrt(zp.real(-1.0))` → `PHI`

- How do tags propagate through computations and gradients?
  - Forward: `PHI` propagates; `±∞` obeys TR tag tables deterministically.
  - Gradients (default Mask‑REAL): paths that produce non‑REAL tags contribute zero gradient.
  - Deterministic reductions: when enabled in policy, sums/products use pairwise trees for stable results.

- When should I pick Hybrid or Saturating gradients instead of Mask‑REAL?
  - Hybrid: when you want zero gradients exactly at singularities but smooth, bounded gradients near them (with hysteresis control).
  - Saturating: when you need bounded, continuous gradients across singular regions (e.g., for continuous control).
  - Default Mask‑REAL is recommended for strict stability and identifiability.

- Do you respect IEEE signed zero (−0.0 vs +0.0)?
  - Yes. Division by `+0.0` vs `−0.0` yields opposite infinities in accordance with IEEE sign rules.
  - Policy has `keep_signed_zero` to preserve signed zero in backends that expose it.

- How do I convert to/from IEEE NaN/Inf?
  - `zp.from_ieee(float('nan'))` → `PHI`; `zp.from_ieee(float('inf'))` → `PINF`.
  - `zp.to_ieee(zp.phi())` → `nan`; `zp.to_ieee(zp.pinf())` → `inf`.

- What’s different in Wheel mode?
  - Stricter algebra with a `BOTTOM (⊥)` element: `0×∞=⊥`, `∞+∞=⊥`; `⊥` propagates.
  - Enable with `with zp.wheel_mode(): ...` when you need formal‑verification‑style behavior.
### Evaluator CLI

You can also run the integrated evaluator via module invocation:

```bash
python -m zeroproof.eval --xmin -2 --xmax 2 --n 201 --true-pole 0.5 --out results/eval.json
```

### RR‑IK Quick Runner

The quick runner generates a bucketed RR‑IK dataset and trains selected models.
On constrained environments, use small sizes to keep runtimes short:

```bash
python examples/robotics/rr_ik_quick.py \
  --out runs/rr_ik_quick \
  --models tr_rat \
  --profile quick \
  --n 1000

# Or drive training directly with tighter limits
python examples/robotics/rr_ik_dataset.py --n_samples 800 --stratify_by_detj --output data/rr_quick.json
python examples/robotics/rr_ik_train.py \
  --dataset data/rr_quick.json --model tr_rat --profile quick \
  --epochs 3 --limit_train 300 --limit_test 100 --output_dir runs/rr_quick
```
### Benchmarks

Run a small benchmark suite and save JSON results:

```bash
python -m zeroproof.bench --suite all --out benchmark_results
```

For a larger suite with more scenarios, see `benchmarks/run_benchmarks.py`.

Render a compact summary from a JSON result:

```bash
python -m zeroproof.bench_summary benchmark_results/bench_....json
```

### Hybrid Overhead

Compare Mask‑REAL vs Hybrid per‑batch timing and activation stats:

```bash
python -m zeroproof.overhead_cli --out runs/overhead.json
```

Outputs include baseline/hybrid avg_step_ms, slowdown_x, and hybrid mode stats.

### Compare Benchmark JSONs

Detect regressions between two benchmark runs (returns non‑zero on slowdown):

```bash
python -m zeroproof.bench_compare \
  --baseline benchmark_results/bench_old.json \
  --candidate benchmark_results/bench_new.json \
  --max-slowdown 1.15
```

Update CI baseline from latest local run:

```bash
python scripts/update_benchmark_baseline.py --src benchmark_results
```

## Acknowledgements

We gratefully acknowledge the open‑source projects and tools that make this work possible:

- Core libraries: NumPy, PyTorch, JAX
- Testing and property‑based fuzzing: pytest, Hypothesis
- Developer tooling: Black, Ruff, isort, MyPy, pre‑commit
- CI and reporting: GitHub Actions, Codecov, shields.io

Thanks as well to all contributors and users of ZeroProof for ideas, bug reports, and feedback.
