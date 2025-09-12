# Topic 2: Mathematical Foundations

This topic defines ZeroProof’s transreal scalar, arithmetic semantics, reductions, and precision behavior, with pointers to spec and code.

## TR Scalar & Tags
- Carrier: TR := (val: float, tag ∈ {REAL, PINF, NINF, PHI})
- Meaning: `val` is read only when `tag=REAL`. Non‑REAL tags carry semantics.
- Optional: Wheel mode introduces `BOTTOM (⊥)`.
- Code: `zeroproof/core/tr_scalar.py:1`

Factories and checks
```python
from zeroproof.core.tr_scalar import real, pinf, ninf, phi
x = real(3.0); y = real(0.0); inf = pinf(); nul = phi()
```

## Arithmetic Semantics (Totalized)
- Every op returns a valid TR value; no exceptions. Deterministic tag rules.
- Code: `zeroproof/core/tr_ops.py:1`

Addition (⊕)
| ⊕ | REAL | PINF | NINF | PHI |
|---|------|------|------|-----|
| REAL | REAL | PINF | NINF | PHI |
| PINF | PINF | PINF | PHI  | PHI |
| NINF | NINF | PHI  | NINF | PHI |
| PHI  | PHI  | PHI  | PHI  | PHI |

Multiplication (⊗)
| ⊗ | REAL≠0 | 0 | PINF | NINF | PHI |
|---|--------|---|------|------|-----|
| REAL≠0 | REAL | 0 | ±∞ | ±∞ | PHI |
| 0 | 0 | 0 | PHI | PHI | PHI |
| PINF | ±∞ | PHI | PINF | NINF | PHI |
| NINF | ±∞ | PHI | NINF | PINF | PHI |
| PHI | PHI | PHI | PHI | PHI | PHI |

Division (⊘)
- Finite/finite, denom≠0 → REAL
- x/0 → sign‑∞ (x>0→+∞, x<0→−∞), 0/0→PHI
- (±∞)/(±∞) → PHI; finite/∞ → 0 (REAL)
- Code paths: `tr_div` handles sign of ±0 and ∞ cases

## Unary Semantics (Domain‑Aware)
- log: REAL x>0 → REAL ln(x); else PHI; log(+∞)→+∞
- sqrt: REAL x≥0 → REAL √x; else PHI; sqrt(+∞)→+∞
- pow_int(x,k): integer power under TR laws; 0^0=PHI, (±∞)^0=PHI
- Code: `tr_log`, `tr_sqrt`, `tr_pow_int` in `zeroproof/core/tr_ops.py:400`

## Reduction Semantics
- STRICT: PHI dominates; ±∞ of conflicting signs → PHI; else ∞ or REAL
- DROP_NULL: Ignore PHI; if none remain → PHI (monitoring/metrics)
- Code: `zeroproof/core/reduction.py:1`

## Precision & Overflow
- Default precision: float64; conversions enforced centrally.
- Overflow mapping: If REAL arithmetic overflows numeric range → ±∞ under TR rules (sign‑aware).
- APIs: `PrecisionConfig.enforce_precision`, `PrecisionConfig.check_overflow`
- Code: `zeroproof/core/precision_config.py:1`, used in `tr_scalar` and `tr_ops`

Example
```python
from zeroproof.core.tr_ops import tr_div, tr_mul
from zeroproof.core.tr_scalar import real

# Division by zero is total
tr_div(real(1.0), real(0.0))   # → +∞
tr_div(real(0.0), real(0.0))   # → Φ

# 0 × ∞ is PHI in TR; ⊥ in Wheel mode
```

## Mode Isolation (TR vs Wheel)
- TR (default): PHI for indeterminate forms.
- Wheel (optional): ⊥ for cases like 0×∞, ∞±∞, ∞/∞; strict propagation.
- Isolation: Must not mix modes within an op; enforced via mode config.
- Code: `zeroproof/core/wheel_mode.py:1`, `zeroproof/core/mode_isolation.py:1`

## Determinism & No‑ε Invariant
- Tag decisions use exact predicates (e.g., denom==0). No ε in core ops.
- See spec clarifications: `complete_v2.md:1` (normative rules and tables)

## Cross‑References
- Full spec: `complete_v2.md:1`
- Conceptual framework: `concept_250908.md:1`
- Quick reference: `docs/quick_reference.md:1`
