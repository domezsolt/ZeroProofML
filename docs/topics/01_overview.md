# Topic 1: Overview & Principles

This overview orients you to ZeroProof’s goals, how it differs from conventional ML numerics, and where to look next.

## What ZeroProof Solves

- Singularities and indeterminate forms appear in many domains (e.g., division by zero, ∞−∞, 0×∞, log≤0, zero variance normalization).
- Conventional stacks rely on ε-hacks or avoid these regions, causing silent instability, NaNs, or brittle heuristics.
- ZeroProof uses Transreal (TR) arithmetic to make operations total: every op returns a value with an explicit tag, never throwing.

## Core Idea: Transreal Values

- Tags: REAL (finite), PINF (+∞), NINF (−∞), PHI (nullity for indeterminate forms). Wheel-mode adds BOTTOM (⊥).
- Total operations: +, −, ×, ÷, log, sqrt, pow_int are defined for all tag combinations with deterministic rules.
- Determinism: Tag decisions use exact predicates (e.g., denom==0). No ε thresholds in core semantics.

## Key Innovations

- Total operations: No exceptions; stable behavior at singularities.
- Mask-REAL gradients: Non-REAL forward tags produce zero parameter gradients (prevents gradient explosions).
- Saturating gradients (optional): Bounded gradients near poles without ε, for continuous alternatives.
- Coverage control: Adaptive λ_rej maintains target REAL/non-REAL ratios in training.
- Pole learning: Layers and metrics explicitly detect, supervise, and localize poles.

## Design Principles

- Totality first: All ops are total on TR; forward/backward never throw.
- Exactness over ε: No hidden thresholds in core; any evaluation-time τ is explicitly opt-in.
- Explicit tags: Values carry semantics; reductions declare STRICT vs DROP_NULL.
- Gradient safety: Mask-REAL ensures bounded updates when encountering non-REAL tags.
- Mode isolation: TR and Wheel modes never mix within an op; bridging is explicit.

## When to Use TR vs Wheel

- TR (default): Keeps arithmetic flowing with PHI for indeterminate forms; suitable for most ML tasks.
- Wheel (optional): Replaces certain indeterminate results with ⊥ for stricter algebraic control (e.g., 0×∞=⊥, ∞+∞=⊥). Useful for audits/verification.

## Where to Go Next

- Semantics and tables: See Spec Clarifications in `complete_v2.md`.
- Autodiff rules: See `docs/autodiff_mask_real.md` and `docs/saturating_grad_guide.md`.
- Layers overview: `docs/layers.md` (TR-Rational, TR-Norm, enhanced variants).
- Bridges and interop: `docs/bridge_summary.md` and `docs/bridge_extended.md`.
- Precision and float64 enforcement: `docs/float64_enforcement.md`.
- Adaptive loss & coverage: `docs/adaptive_loss_guide.md` and summaries in `docs/adaptive_loss_summary.md`.

## Glossary

- REAL: Finite real number; the standard slice of ℝ.
- PINF/NINF: +∞/−∞ as first-class values.
- PHI (Φ): Nullity; indeterminate forms (0/0, ∞−∞, 0×∞, log≤0, 0^0, (±∞)^0).
- BOTTOM (⊥): Wheel-mode’s error-like element; propagates strictly.
- STRICT/DROP_NULL: Reduction modes governing how non-REALs aggregate.
- Mask-REAL: Autodiff rule sending zero grads through non-REAL forward states.

For the full conceptual framework, see `concept_250908.md`.
