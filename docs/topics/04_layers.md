# Topic 4: Layers & Variants (TR‑Rational, TR‑Norm, Enhanced)

This topic covers ZeroProof’s core layers, how they differ, and when to use each. For API details see `docs/layers.md`.

## TR‑Rational (P/Q)
- Purpose: Learn rational functions with explicit poles; total under TR semantics.
- Form: y = P_θ(x) / Q_φ(x) with Q leading 1 for identifiability.
- Tags: REAL when Q≠0; ±∞ when Q=0 and P≠0 (sign from P); Φ when P=Q=0.
- Gradients: Mask‑REAL (default); zero grads when forward tag is non‑REAL.
- Stability aids: L2 on φ; optional L1 projection bound on ||φ||₁.
- Code: `zeroproof/layers/tr_rational.py:1`.

Usage
```python
from zeroproof.layers import TRRational, ChebyshevBasis
from zeroproof.autodiff.tr_node import TRNode
from zeroproof.core import real

layer = TRRational(d_p=3, d_q=2, basis=ChebyshevBasis())
y, tag = layer.forward(TRNode.constant(real(0.2)))
```

Choosing a basis
- Monomial: simple, good for low degree.
- Chebyshev: stable on bounded intervals; recommended default.
- Fourier: for periodic signals (if available in your build).

## TR‑Norm (Epsilon‑Free Normalization)
- Purpose: Batch/layer normalization with ε→0⁺ semantics; deterministically handles σ²=0.
- Behavior: If σ²>0 → classical normalization; if σ²=0 → bypass to β.
- Stats: Use DROP_NULL over REAL-only subset for μ, σ².
- Gradients: Regular branch = classical; bypass branch = ∂ŷ/∂x=0, ∂ŷ/∂β=1, ∂ŷ/∂γ=0.
- Code: `zeroproof/layers/tr_norm.py:1`.

## Enhanced & Variant Layers
When you need more control or explicit pole learning:

- SaturatingTRRational: Same P/Q with Saturating gradient mode baked in.
  - Code: `zeroproof/layers/saturating_rational.py:1`.

- HybridTRRational: Integrates Hybrid gradient schedule (Mask‑REAL far from poles, Saturating near poles) and optional Q tracking.
  - Code: `zeroproof/layers/hybrid_rational.py:1`.

- HybridRationalWithPoleHead: Adds auxiliary pole‑detection head to localize Q≈0.
  - Code: `zeroproof/layers/hybrid_rational.py:180`.

- EnhancedTRRational / EnhancedTRRationalMulti: Integrates pole detection and regularization with multi‑output options.
  - Code: `zeroproof/layers/enhanced_rational.py:1`.

- TagAwareRational / TagAwareMultiRational: Adds tag‑aware losses and outputs for training that supervises tag distribution.
  - Code: `zeroproof/layers/tag_aware_rational.py:1`.

- PoleAwareRational / FullyIntegratedRational: End‑to‑end stacks with pole metrics/regularizers.
  - Code: `zeroproof/layers/pole_aware_rational.py:1`.

- EnhancedPoleDetectionHead and regularizer components for custom assemblies.
  - Code: `zeroproof/layers/enhanced_pole_detection.py:1`.

## Multi‑Input / Multi‑Output Rationals

Some tasks (e.g., robotics IK) take vector inputs and produce multi‑dimensional outputs while sharing pole structure.

- TRRationalMulti: Multiple outputs with a shared denominator Q (optional) and separate numerators.
  - Use when outputs share the same pole locations (shared_Q=True).
  - Code: `zeroproof/layers/tr_rational.py:408` (`class TRRationalMulti`).

- TRMultiInputRational: Small TR‑MLP front end (R^D→K features) feeding TR‑Rational heads for multi‑output.
  - Forward (vector): `forward([TRNode|float]*input_dim) -> List[(TRNode, TRTag)]` (len = n_outputs).
  - Structured forward: `forward_fully_integrated(...) -> {outputs, tags, Q_abs_list?, pole_score?}`.
  - Supports enabling a simple pole head; provides `regularization_loss()` across heads.
  - Code: `zeroproof/layers/multi_input_rational.py:1`.

Usage (4D→2D)
```python
from zeroproof.layers import TRMultiInputRational, MonomialBasis

model = TRMultiInputRational(
    input_dim=4,
    n_outputs=2,
    d_p=3,
    d_q=2,
    basis=MonomialBasis(),
    hidden_dims=[8],
    shared_Q=True,
    enable_pole_head=False,
)

# Vector forward (floats or TRNodes)
outs = model.forward([0.1, -0.2, 0.05, -0.05])  # [(TRNode, TRTag), (TRNode, TRTag)]
```

When to use which
- Prefer TRRationalMulti when outputs share Q and the input is scalar or already projected.
- Prefer TRMultiInputRational when inputs are vectors and you want a lightweight front end with shared Q across heads.

## Practical Patterns
- Start simple: TRRational + Chebyshev basis; monitor tag distribution and q_min.
- For pole learning: switch to HybridTRRational with a gentle schedule; enable Q tracking to tune δ.
- Add a pole head when you have labels/weak‑labels for singularities; combine with coverage control in training.
- For normalization without ε: use TRNorm or TRLayerNorm to avoid tuning eps.
 - Benchmark training: Hybrid trainer prints and records per‑epoch timings — `avg_step_ms`, `data_time_ms`, `optim_time_ms`, `batches` — returned in training summaries under `bench_history`. Adjust logging cadence via `log_interval` (CLI `--log_every`).

## Interactions with Autodiff
- All layers use lifted TR ops (`tr_ops_grad`) and integrate with `TRNode`.
- Autodiff modes apply as configured globally (Mask‑REAL default, Saturating, or Hybrid via schedules).
- Code references: `zeroproof/autodiff/tr_ops_grad.py:1`, `zeroproof/autodiff/backward.py:1`.

## Diagnostics to Track
- q_min (batch/epoch): from Hybrid context or via layer Q tracking.
- Tag distribution: counts of REAL, PINF, NINF, PHI during training.
- Near‑pole ratio: fraction of samples triggering Saturating under Hybrid.
- Pole localization metrics: use `utils/pole_metrics.py` if applicable.

## See Also
- Doc: `docs/layers.md:1` for extended explanation and examples.
- Concepts: `docs/topics/02_foundations.md:1` for arithmetic rules.
- Autodiff: `docs/topics/03_autodiff_modes.md:1` for mode selection.
