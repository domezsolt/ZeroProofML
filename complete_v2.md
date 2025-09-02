# Revision 2025-08-26 — Spec Clarifications & Changes (Normative)

> **Scope.** This section is normative and overrides any conflicting statements elsewhere in this document.
> It clarifies PHI semantics, stability conditions, AD rules near poles, compiler rewrites, Wheel-mode isolation,
> evaluation-only thresholds, determinism guarantees, and loss policy for non‑REAL tags.

## 1) PHI semantics & `0^0`
- **Definition.** `Φ` (PHI) denotes **indeterminate forms** (e.g., `0/0`, `∞−∞`, `0·∞`, `log(x≤0)`, and also `0^0`, `(±∞)^0`).
- **General-purpose power.** In `pow_int(x,k)` these remain `Φ` to preserve totalized arithmetic across domains.
- **Domain-specific convention.** Contexts that conventionally use `0^0 = 1` (e.g., combinatorics) MUST use a dedicated function
  (e.g., `comb`, `pow0_for_combinatorics`) rather than override `pow_int` globally.

## 2) Stable region for TR-Rational layers
- The bound \(|Q(x)| \ge 1 - \lVert \phi \rVert_1\,B\) is a **sufficient, not necessary** condition for smooth REAL‑path training.
- Practical controls:
  1. L2 regularization on the denominator coefficients (\(\phi\)) + online monitoring of \(q_{\min}\).
  2. Optional \(\ell_1\) projection enforcing \(\lVert \phi \rVert_1 \le \rho < 1/B\) if desired.
  3. Chebyshev basis for \(Q\) on \([-1,1]\), where \(B=1\).

## 3) Gradients at singularities — Mask‑REAL and a soft alternative
- **Primary rule (default):** **Mask‑REAL** — for non‑REAL tags (±∞, Φ) the local Jacobian is **zero**.
- **Optional variant:** **Saturating‑grad** — replace the singular growth (e.g., \(1/Q^2\)) by a TR‑safe capped form
  such as \(\frac{1}{Q^2\oplus 1_R}\) to obtain **continuous** transitions near poles, **without** ε‑hacks.
  This variant is for ablations or tasks that prefer a soft approach; Mask‑REAL remains the default.

## 4) Totalization terminology
- Operations are **totalized over the TR carrier** (REAL, `±∞`, `Φ`). No operation throws; `Φ` is a first‑class **value**
  representing indeterminate forms, not an external “undefined”. This is an ADT view of TR arithmetic.

## 5) Safe‑Rewrite rule for compilers
- Algebraic rewrites relying on associativity/distributivity are sound **on the REAL slice**.
- When `Φ` can arise, **compilers must preserve evaluation order** unless a static proof establishes the expression
  is confined to the REAL slice. Optimizers SHOULD implement a REAL‑path analysis before rewriting.

## 6) Wheel mode isolation
- **Wheel mode** is a compile‑time semantic parameterization (mapping `Φ`→`⊥` and adopting wheel laws such as `0·∞=⊥`, `∞+∞=⊥`).
- **No mixing:** TR and Wheel modes MUST NOT be mixed **within a single operation**. Any bridging must be explicit.
- Implementations SHOULD expose mode as a global or module‑level switch that applies uniformly to a graph.

## 7) `τ` threshold is evaluation‑only (no ε in core)
- The core forward rules and autodiff are **ε‑free**. Any `τ>0` threshold appears **only** in evaluation‑time
  **risk–coverage** curves (e.g., gating predictions by small \(|Q|\)). `τ` MUST NOT influence training gradients
  or the forward rules.

## 8) Determinism statement and repro checklist
- Tag decisions (REAL vs `±∞`/`Φ`) are **threshold‑free** and order‑independent by construction.
- Bit‑level determinism is guaranteed **per backend** under deterministic settings. Cross‑platform bit‑equality is **not** guaranteed.
- **Repro checklist (recommended):**
  - Fix random seeds across framework + libraries.
  - Enable framework “deterministic”/“noTF32”/“noFMA” modes where applicable.
  - Pin BLAS/cuBLAS/cuDNN versions and set deterministic flags.
  - Fix compiler flags that alter floating‑point contraction and reassociation.

## 9) Loss policy for non‑REAL tags and adaptive λ₍rej₎
- Let \(\lambda_{\text{rej}}\) be the penalty applied when an output tag is non‑REAL (±∞, Φ).
- Treat \(\lambda_{\text{rej}}\) as a **Lagrange multiplier** to control target coverage \(c^*\):
  \[ \lambda \leftarrow \lambda + \eta_\lambda\, (c^* - c_{\text{actual}}) \]
- With Mask‑REAL, gradients w.r.t. model parameters remain unaffected on non‑REAL samples, avoiding instability,
  while \(\lambda\) adapts to meet the desired coverage.

---



---

# Scope & Invariants (Phase 0)

> **Applies to:** TR-Rational layer (P/Q), TR-AD (autodiff), TR-Norm (epsilon-free normalization), IEEE↔TR bridge.  
> **Scalar:** `TR = (val: float64, tag: {REAL, PINF, NINF, PHI})`.

---

## 0) Purpose & Non‑Goals
**Purpose.** Nail down semantics we preserve from math → code so every op is **total** (never throws) and **deterministic**.

**Non‑goals (for MVP foundations).** Complex numbers; generalized functions; GPU kernels/compilers; full wheel/meadow production track (wheel kept as optional experiment mode).

---

## 1) Core Data Model
- **REAL:** finite real number.  
- **PINF / NINF:** +∞ / −∞ as **first‑class values**.  
- **PHI:** “nullity / undefined form” (e.g., 0/0, ∞−∞, 0·∞, log≤0).

**Tensor semantics.** Elementwise TR values; reductions must state their **tag strategy**.

---

## 2) Arithmetic Semantics (Totality)
All ops are **pure** and **total** on TR. They return `(val, tag)` without raising exceptions.

### 2.1 Addition (a + b)
**Rules:**
- REAL ± REAL → REAL (IEEE addition on `val`).
- finite ± (±∞) → that infinity (sign dominates).
- +∞ + +∞ → PINF; −∞ + −∞ → NINF.
- +∞ + −∞ (or vice versa) → PHI.
- PHI + anything → PHI; anything + PHI → PHI.

**Tag table (addition)**
| a \ b | REAL | PINF | NINF | PHI |
|---|---|---|---|---|
| **REAL** | REAL | PINF | NINF | PHI |
| **PINF** | PINF | PINF | PHI  | PHI |
| **NINF** | NINF | PHI  | NINF | PHI |
| **PHI**  | PHI  | PHI  | PHI  | PHI |

> **Note.** Signs for REAL±∞ cases are inherited from the ∞ operand.

### 2.2 Multiplication (a × b)
**Rules (key cases):**
- nonzero REAL × (±∞) → (±∞) (sign product).  
- 0 × (±∞) → PHI.  
- (±∞) × (±∞) → (±∞) (sign product).  
- PHI × anything → PHI; anything × PHI → PHI.

**Tag table (multiplication)**
| a \ b | REAL≠0 | 0 | PINF | NINF | PHI |
|---|---|---|---|---|---|
| **REAL≠0** | REAL | 0  | PINF/NINF* | PINF/NINF* | PHI |
| **0**      | 0    | 0  | PHI        | PHI        | PHI |
| **PINF**   | PINF/NINF* | PHI | PINF | NINF | PHI |
| **NINF**   | PINF/NINF* | PHI | NINF | PINF | PHI |
| **PHI**    | PHI  | PHI | PHI | PHI | PHI |

\* **Sign rule:** sign(∞)·sign(real) for REAL≠0; sign(∞)·sign(∞) for ∞×∞.

### 2.3 Division (a ÷ b)
**Rules (prioritized):**
1. finite / finite:
   - denom≠0 → REAL;  
   - denom=0 → (num>0 → PINF; num<0 → NINF; num=0 → PHI).
2. (±∞) / finite nonzero → (±∞) (sign depends on denom sign).  
3. finite / (±∞) → 0 (REAL).  
4. (±∞) / 0 → (±∞) (sign of numerator).  
5. 0 / (±∞) → 0 (REAL).  
6. (±∞)/(±∞) → PHI.  
7. PHI / anything → PHI; anything / PHI → PHI.

**Tag table (division)**
| a \ b | REAL>0 | REAL<0 | 0 | PINF | NINF | PHI |
|---|---|---|---|---|---|---|
| **REAL>0** | REAL | REAL | PINF | 0 | 0 | PHI |
| **REAL<0** | REAL | REAL | NINF | 0 | 0 | PHI |
| **0**      | 0    | 0    | PHI  | 0 | 0 | PHI |
| **PINF**   | PINF | NINF | PINF | PHI | PHI | PHI |
| **NINF**   | NINF | PINF | NINF | PHI | PHI | PHI |
| **PHI**    | PHI  | PHI  | PHI  | PHI | PHI | PHI |

> **Note.** REAL>0 / REAL<0 split is to encode sign flips.

### 2.4 Unary ops (initial set)
| op | Input tag | Output |
|---|---|---|
| `abs(x)` | REAL | REAL: `|x|` |
|  | PINF | PINF |
|  | NINF | PINF |
|  | PHI  | PHI |
| `sign(x)` | REAL | REAL in {−1,0,1} |
|  | PINF | REAL `+1` |
|  | NINF | REAL `−1` |
|  | PHI  | PHI |
| `log(x)` | REAL `x>0` → REAL `ln x`; `x≤0` → PHI |
| `sqrt(x)` | REAL `x≥0` → REAL `√x`; `x<0` → PHI |
| `pow(x,k)` (k∈ℤ) | computed by repeated × and ÷ under TR rules; special: `0^0 → PHI`, `(±∞)^0 → PHI` |

**Determinism invariant.** Tag decisions use exact predicates (e.g., denom==0) on REAL values; no hidden ε.

---

## 3) Reduction Semantics (Aggregations)
Every reduction declares a **mode**:
- **`strict`**: if any element is PHI then result is PHI; else if any is ±∞, result is that ±∞ (if both ±∞ present with conflicting signs → PHI); else REAL.
- **`drop-null`**: ignore PHI elements; reduce over remaining; if none remain → PHI. (Use in metrics/monitoring; training defaults to `strict` unless explicitly justified.)

---

## 4) TR‑Rational Layer Invariants
**Definition.** \( y = P_\theta(x) / Q_\phi(x) \) on a low‑degree polynomial basis; identifiability via leading 1 in `Q` (e.g., `Q(0)=1`).

**Forward tag rules.**
- `Q(x) ≠ 0` → tag = REAL.
- `Q(x) = 0 ∧ P(x) ≠ 0` → tag = PINF/NINF by sign of `P(x)`.
- `Q(x) = 0 ∧ P(x) = 0` → tag = PHI.

**No‑ε invariant.** No path adds ε to `Q` or replaces `Q` by `Q+ε`.

---

## 5) TR‑AD (Autodiff) Invariants
**Primary rule — Mask‑REAL.**
- If forward tag = **REAL** → use standard quotient rule for grads.  
- If forward tag ∈ **{PINF, NINF, PHI}** → **stop‑gradient** to layer params (send zero grads to `θ, φ`); upstream grads remain defined (zeros for this node wrt params).

**Ablation — Saturating‑grad.** Optionally bound the `1/Q^2` factor via a transreal‑safe cap; still no ε.

**Chain‑rule invariant.** Composition of TR ops is total; backprop never throws; grads are finite or identically zero by construction.

**Quick reference (forward tag → param grad rule)**
| Forward tag | Grad to `θ, φ` |
|---|---|
| REAL | standard quotient rule |
| PINF/NINF | 0 |
| PHI | 0 |

---

## 6) TR‑Norm (Epsilon‑Free Normalization) Invariants
**Definition.** \( \hat{x} = (x - \mu) / \sqrt{\sigma^2} \) in TR.

**Bypass rule (σ²=0).** If variance is exactly zero for a feature/batch, set \(\hat{x}:=0\) deterministically; layer output becomes \(y = \gamma·0 + \beta = \beta\). Tags remain REAL.

**Limit‑equivalence.** For σ²>0, TR‑Norm equals the limit of BN with ε→0⁺; for σ²=0 it follows the affine bypass. Never produces NaN/Inf in outputs or grads.

---

## 7) IEEE ↔ TR Bridge Invariants
| IEEE value | TR tag/value |
|---|---|
| finite float | `(val, REAL)` |
| `+∞` | `(—, PINF)` |
| `−∞` | `(—, NINF)` |
| `NaN` | `(—, PHI)` |

**Round‑trip invariant.** `to_ieee(to_tr(x)) = x` and `to_tr(to_ieee(z)) = z` on supported subsets (up to usual float rounding on REAL values).

---

## 8) Determinism, Reproducibility, Precision
- All tag decisions are threshold‑free and order‑independent.
- Default numeric precision: float64; all randomness seeded.
- No implicit TR↔IEEE casts inside core ops; bridging is explicit.

---

## 9) Optional Wheel Mode (experimental)
Compile‑time switch to substitute **⊥** for PHI and enforce wheel control laws (`0·∞=⊥`, `∞+∞=⊥`, etc.). TR and wheel values are not mixed within a single op without explicit bridging.

---

## 10) Acceptance Tests (Phase‑0 gate)
- **Totality:** every entry in the op tables returns a defined tag.
- **No‑NaN:** random compositions (forward+backward) produce zero IEEE NaNs.
- **Bridge round‑trip:** fuzz passing on finite/±∞/NaN.
- **TR‑Rational poles:** synthetic `Q=0` inputs hit intended tags.
- **TR‑Norm limit:** matches BN(ε) as ε→0⁺ for σ²>0; bypass for σ²=0.

---

# Property‑Test Checklist (Hypothesis/pytest)

> **Goal:** executable tests that mirror the spec. Use float64; include edge cases (±0.0, subnormals).

## A) Generators
- `finite_reals()` → wide float range excluding ±∞/NaN.
- `tr_scalars()` → union of: `(REAL, random float)`, `(PINF)`, `(NINF)`, `(PHI)`.
- `expr_trees(depth)` → random binary trees of ops {+, ×, ÷} and unary {abs, log, sqrt, pow(±1,2)}.

## B) Arithmetic (Sections 2.1–2.4)
- **Totality:** for all a,b in `tr_scalars()`, `add/mul/div(a,b)` returns a TR with tag in {REAL,PINF,NINF,PHI} and never raises.
- **Tables:** parameterized tests enumerating the 3×3 or 5×5 cases in the tables; assert tag equality to the expected table entry.
- **Specials:** `0×∞→PHI`, `∞+−∞→PHI`, `0/0→PHI`, `finite/∞→0 (REAL)`.
- **Unary:** `log(x≤0)→PHI`, `sqrt(x<0)→PHI`, `0^0→PHI`, `(±∞)^0→PHI`.

## C) Reductions (Section 3)
- **strict:** if any PHI present → PHI; else if both PINF and NINF present → PHI; else if any infinity → that infinity; else REAL.
- **drop‑null:** dropping PHI equals reducing the filtered list; empty → PHI.

## D) TR‑Rational Layer (Section 4)
- **Forward tags:** construct `(P,Q)` so that `Q(x)=0` and test: `(P≠0)→±∞`, `(P=0)→PHI`; else REAL.
- **Identifiability:** scaling (cP,cQ) with leading‑1 in `Q` is disallowed; test constraint enforcement.

## E) TR‑AD (Section 5)
- **Mask‑REAL:** when forward tag is REAL, `∂y/∂θ, ∂y/∂φ` = quotient‑rule grads; when tag∈{±∞,PHI} both param‑grads **exactly zero**.  
- **No‑NaN grads:** backprop through random expr trees yields finite grads or zeros only.

## F) TR‑Norm (Section 6)
- **σ²>0:** for a batch, `TR‑Norm(x)` ≈ `BN(x, ε)` as `ε→0⁺` (numerically compare for `ε∈{1e−6,1e−8,1e−10}`).
- **σ²=0:** outputs equal affine bypass (`β`) and tags are REAL.

## G) IEEE↔TR Bridge (Section 7)
- **Round‑trip:** for samples in {finite,±∞,NaN}, `to_tr→to_ieee` and `to_ieee→to_tr` are identity (up to float rounding for REAL).

---

## Minimal pytest skeleton
```python
# tests/test_tr_totality.py
import pytest
from hypothesis import given, strategies as st
from trlib import add, mul, div, TR, Tag

@st.composite
def tr_scalars(draw):
    tag = draw(st.sampled_from([Tag.REAL, Tag.PINF, Tag.NINF, Tag.PHI]))
    if tag == Tag.REAL:
        val = draw(st.floats(allow_nan=False, allow_infinity=False, width=64))
        return TR(val, tag)
    else:
        return TR(float('nan'), tag)

@given(tr_scalars(), tr_scalars())
def test_totality_add(a,b):
    out = add(a,b)
    assert out.tag in {Tag.REAL, Tag.PINF, Tag.NINF, Tag.PHI}

# … replicate for mul/div and table-parameterized cases
```

```python
# tests/test_tr_norm.py
import numpy as np
from trlib import tr_norm, bn_eps

def test_tr_norm_sigma_zero_bypass():
    x = np.ones((8, 4))  # variance=0 per feature
    y = tr_norm(x)
    # Expect affine bypass → normalized part = 0
    assert np.allclose(y, 0.0)  # before gamma/beta
```

---

### Glossary
- **REAL, PINF, NINF, PHI:** TR tags (finite, +∞, −∞, nullity).  
- **strict / drop‑null:** reduction modes.  
- **mask‑REAL:** only REAL forward states produce non‑zero param grads.  
- **bypass:** TR‑Norm’s deterministic σ²=0 branch.

 
# Phase 1 — Algebraic Foundation

## 1) TR scalar (definition)
**Carrier.** `TR := { (val: ℝ, tag) | tag ∈ {REAL, PINF, NINF, PHI} }` with the convention that `val` is only semantically read when `tag=REAL`.

**Intuition.**
- `REAL`: finite reals; embed ℝ via `ι(r)=(r,REAL)`.
- `PINF/NINF`: +∞/−∞ as first‑class values.
- `PHI`: nullity (undefined forms such as 0/0, ∞−∞, 0·∞, log≤0).

**Goal.** Make `+, −, ×, ÷` **total** on TR and deterministic.

---

## 2) Operations (closed tables)
All rules are **pure** (no ε hacks). Write tags as R, +∞, −∞, Φ for brevity.

### 2.1 Addition (⊕)
- R ⊕ R → R (usual real addition on `val`).
- R ⊕ (±∞) → (±∞) ; (±∞) ⊕ R → (±∞).
- (+∞) ⊕ (+∞) → +∞ ; (−∞) ⊕ (−∞) → −∞.
- (+∞) ⊕ (−∞) → Φ ; (−∞) ⊕ (+∞) → Φ.
- Φ ⊕ x → Φ ; x ⊕ Φ → Φ.

**Tag table.**
| ⊕ | R | +∞ | −∞ | Φ |
|---|---|---|---|---|
| **R** | R | +∞ | −∞ | Φ |
| **+∞** | +∞ | +∞ | Φ  | Φ |
| **−∞** | −∞ | Φ  | −∞ | Φ |
| **Φ**  | Φ  | Φ  | Φ  | Φ |

### 2.2 Multiplication (⊗)
- (nonzero R) ⊗ (±∞) → (±∞) with sign product; (±∞) ⊗ (±∞) → (±∞) with sign product.
- 0 ⊗ (±∞) → Φ ; (±∞) ⊗ 0 → Φ.
- Φ ⊗ x → Φ ; x ⊗ Φ → Φ.

**Tag table.**
| ⊗ | R≠0 | 0 | +∞ | −∞ | Φ |
|---|---|---|---|---|---|
| **R≠0** | R | 0 | ±∞ | ±∞ | Φ |
| **0**    | 0 | 0 | Φ  | Φ  | Φ |
| **+∞**   | ±∞ | Φ | +∞ | −∞ | Φ |
| **−∞**   | ±∞ | Φ | −∞ | +∞ | Φ |
| **Φ**    | Φ  | Φ | Φ  | Φ  | Φ |

### 2.3 Division (⊘)
- R / R: if denom≠0 → R; if denom=0 → sign‑∞; 0/0 → Φ.
- (±∞)/R≠0 → (±∞) (sign flip by denom); R/(±∞) → 0 (R).
- (±∞)/0 → (±∞) (sign by numerator); (±∞)/(±∞) → Φ.
- Φ / x → Φ ; x / Φ → Φ.

**Tag table.**
| ⊘ | R⁺ | R⁻ | 0 | +∞ | −∞ | Φ |
|---|---|---|---|---|---|---|
| **R⁺** | R | R | +∞ | 0 | 0 | Φ |
| **R⁻** | R | R | −∞ | 0 | 0 | Φ |
| **0**  | 0 | 0 | Φ  | 0 | 0 | Φ |
| **+∞** | +∞ | −∞ | +∞ | Φ | Φ | Φ |
| **−∞** | −∞ | +∞ | −∞ | Φ | Φ | Φ |
| **Φ**  | Φ  | Φ  | Φ  | Φ | Φ | Φ |

### 2.4 Selected unary ops
- `abs`: R→R (|x|); +∞→+∞; −∞→+∞; Φ→Φ.
- `sign`: R→{−1,0,1}; +∞→(+1 as R); −∞→(−1 as R); Φ→Φ.
- `log`: R with x>0 → R; else Φ.
- `sqrt`: R with x≥0 → R; else Φ.
- `pow(x,k)` (k∈ℤ): by repeated ⊗ and ⊘ under TR rules; special cases: `0^0=Φ`, `(±∞)^0=Φ`.

---

## 3) Algebraic laws (what holds, what changes)
- **Commutativity**: ⊕ and ⊗ are commutative (tag‑aware).
- **Associativity**: holds over REAL terms; in full TR, associativity can fail for expressions involving Φ (documented and tested).
- **Identity elements**: `0_R=(0,REAL)` for ⊕; `1_R=(1,REAL)` for ⊗ on REAL slice.
- **Distributivity**: holds on REAL slice; at Φ, distributivity does not generally hold (by design; Φ is absorptive for ± operations in many cases).

---

## 4) Closure & Embedding (key lemmas)
**Lemma 1 (Closure).** For all TR a,b, the results `a⊕b`, `a⊗b`, `a⊘b` are in TR (by tables above).  
*Proof sketch.* Case split on tags; each rule returns one of {R, ±∞, Φ}.

**Lemma 2 (Embedding).** The map `ι: ℝ → TR`, `ι(r)=(r,REAL)` is an injective homomorphism and its image is a subfield isomorphic to ℝ under the restricted operations.  
*Proof sketch.* For r,s∈ℝ: `ι(r)+ι(s)=ι(r+s)`, `ι(r)×ι(s)=ι(rs)`, and for s≠0, `ι(r)/ι(s)=ι(r/s)`. Injectivity is immediate from equality on the `val` coordinate. Thus REAL‑slice of TR is a field and ≅ ℝ.

**Remark (ordered/complete transfield).** TR realizes a transfield where +,−,×,÷ are total; under suitable axioms (absorptive, extremal, order, completeness), the transreals are the **smallest ordered, complete transfield** (background note; formalized later in theory doc).

---

## 5) Optional: Wheel (Qw) contrast (for experiments)
**Wheel signatures/axioms (essentials).**
- Constants: `0,1,∞,⊥`; operations: `+,·,−, (·)^{-1}`; division `x/y := x·y^{-1}`.
- Control identities prevent “uncomfortable simplifications”:
  - `1/0=∞`, `1/∞=0`.
  - `0·∞=⊥`.
  - `∞+∞=⊥`.
  - Propagation: `x+⊥=⊥`, `−⊥=⊥`, `x·⊥=⊥`.
- Equational spec `(Σ_w,E_w)` has an **initial algebra** isomorphic to a concrete rational wheel `Q_w`.

**TR vs Wheel (semantic delta).**
- **At ∞+∞:** TR returns ∞ (models “limit‑like” intuition), Wheel returns ⊥ (signals unsafe sum).
- **At 0·∞:** TR returns Φ (nullity), Wheel returns ⊥.
- **Design intent:** TR keeps more arithmetic flowing with explicit Φ, Wheel is stricter (error‑element) for algebraic control.

**Use in our stack.** Default to TR; keep an optional "wheel‑mode" (compile‑time) for mask‑as‑algebra ablations and safety studies.

---

## 6) Artifacts to produce now
- **TR spec PDF:** this section + operation tables + Lemma proofs.
- **Wheel vs TR (2‑page note):** side‑by‑side examples (∞+∞, 0·∞, 0/0, (±∞)/(±∞), distributivity corner cases), guidance for when to use wheel‑mode.

---

## 7) Acceptance (proof sketches / property tests)
- **Property‑based tests:**
  - *Totality:* random TR pairs → ops return a TR tag (never raise).
  - *Tables:* parameterized tests exactly matching 2.1–2.3 tables.
  - *Embedding:* for random reals r,s, TR ops equal real ops on REAL slice.
- **(Optional) Lean/Coq sketch:**
  - Inductive `tag := REAL | PINF | NINF | PHI`.
  - Record `TR := {val:ℝ; tag:tag}`.
  - Define `tr_add, tr_mul, tr_div` by tag case; prove `closure : ∀ a b, tag(tr_op a b) ∈ {R,±∞,Φ}`.
  - Define `ι : ℝ → TR`; prove field homomorphism on REAL.

---

## 8) Worked micro‑examples (sanity)
- `3 / 0 → +∞`; `−2 / 0 → −∞`; `0 / 0 → Φ`.
- `(+∞) + (−∞) → Φ` ; `(+∞) + 7 → +∞` ; `0 × (+∞) → Φ`.
- `log(−5) → Φ`; `sqrt(−1) → Φ` (real domain).

---

## 9) Notes for implementers
- Store TR as `(float64 val, uint2 tag)`; branch on tag only.
- All reductions must choose a `strict` or `drop‑null` strategy explicitly.
- No implicit TR↔IEEE casts; bridge functions handle finite/±∞/NaN.

 
# Phase 2 — TR‑AD (Analysis & Rules)

> **Goal.** Define autodiff over transreal (TR) computations so that (i) derivatives on `REAL` paths match classical calculus, and (ii) gradients are **well‑defined and stable** when forwards produce non‑REAL tags (`PINF/NINF/PHI`). This phase fixes the tag‑lifted rules, clarifies composition, and states the Mask‑REAL composition lemma.

---

## 1) Setup & notation

- **Tags:** `REAL`, `PINF`, `NINF`, `PHI`.
- **Operations:** `+`, `−`, `×`, `/` follow TR’s total arithmetic tables. Division is total; forward always returns a tag in {REAL, PINF, NINF, PHI}.
- **Nodes:** Each node carries `(value, tag)` in forward pass. Backward pass propagates parameter/input partials.

---

## 2) Tag‑lifted differential rules

**Principle.**
- On `REAL` paths, TR‑AD **equals classical AD** (same formulas and chain rule).
- If a node’s **forward tag is non‑REAL**, then **all its output‑w.r.t‑input/param partials are 0** (Mask‑REAL).

**Elementary REAL‑path rules (scalars).** For `y = u ⊕ v` with `⊕ ∈ {+, −, ×, /}` and both inputs `REAL`:
- `+`/`−`: `∂y/∂u = 1`, `∂y/∂v = ±1`.
- `×`: `∂y/∂u = v`, `∂y/∂v = u`.
- `/`: `∂y/∂u = 1/v`, `∂y/∂v = −u/v²`.
- Unary (REAL domain): `d/dx[log x]=1/x`, `d/dx[sqrt x]=1/(2√x)`, `d/dx[x^p]=p x^{p−1}` with their usual domain guards encoded by tags in forward.

**Mask‑REAL clause.** If the forward tag of `y` is in `{PINF, NINF, PHI}`, then for every input `x` of that op, `∂y/∂x := 0`.

---

## 3) **Mask‑REAL composition lemma** (core)

> **Lemma (one‑liner).** In a TR‑AD graph with Mask‑REAL, if **any intermediate node** on a path from an input/parameter to an output has a **non‑REAL** forward tag, then **the entire path’s Jacobian contribution is the zero map**; equivalently, all upstream partials on that path are **exactly zero**.

**Proof sketch (one paragraph).** By structural induction on path length. Base: at the first non‑REAL node `z`, Mask‑REAL sets all partials `∂z/∂· = 0`. Inductive step: for any successor `y=g(z,…)` on that path, the chain rule multiplies by `∂y/∂z`, yielding `∂y/∂(·) = (∂y/∂z)(∂z/∂(·)) = (·)*0 = 0`. Summing over paths in a DAG leaves only **all‑REAL** paths contributing. Hence singular subgraphs **silence** their gradients; training signal flows exclusively through REAL subgraphs.

**Operational corollary.** Non‑REAL forwards cannot cause gradient explosions or undefined backprop; they contribute zero, aligning with Phase 6 bounds and loss policies.

---

## 4) Worked examples

1) **Quotient node.** `y=P/Q`. If `Q=0, P≠0` → tag `±INF` → `∂y/∂θ=0`, `∂y/∂φ=0`. If `Q≠0` (REAL), use classical quotient rule.
2) **Nested comp.** `y = f(g(h(x)))`. If `tag(h)≠REAL`, then `∂y/∂x=0` by the lemma. If all tags REAL, use classical chain `f'·g'·h'`.
3) **Mixture graph.** If two branches feed a sum and one branch is non‑REAL at some node, only the REAL branch contributes to the gradient of the sum.

---

## 5) Vector/Jacobian form

For vector outputs `y∈ℝ^m`, inputs `x∈ℝ^n`, write the Jacobian as a sum over all computation paths. The lemma zeroes every term containing a non‑REAL intermediate, so `J_y(x)` equals the classical Jacobian **restricted to REAL‑tagged subgraphs**.

---

## 6) Interface contract

- Backward returns zeros for any upstream edge whose forward path encountered a non‑REAL node.
- Mixed REAL/non‑REAL batches: automatic masking acts **per sample**.
- Compatible with **TR‑Norm** (its zero‑variance branch is REAL deterministic), and with the **IEEE↔TR bridge** (no NaN backprop).

---

## 7) Test hooks (Phase 7 references)

- **Composition property.** Random DAGs with an injected non‑REAL node must yield zero gradients on all upstream leaves for that path.
- **Chain‑rule equivalence (REAL slice).** Compare TR‑AD to closed‑form/finite‑diff when all tags are REAL.
- **Silencing subgraph.** Force `P=Q=0` at an internal node → verify all upstream partials are zero while siblings continue to train.

 
# Phase 3 — TR‑Rational Layer (Spec + Proof Sketch + Tests)

> **Goal.** Define and validate a rational layer \(y=P_\theta(x)/Q_\phi(x)\) that (i) is **total** under transreal arithmetic (TR), (ii) has **stable AD** via Mask‑REAL, and (iii) is **identifiable** without ε‑hacks. This phase fixes the interface, forward/grad rules, loss policy, and test plan.

---

## 1) Model & parameterization

- **Basis.** Let \(\psi(x)=(\psi_0(x),\dots,\psi_d(x))\) be a fixed feature basis (e.g., monomials, Chebyshev, RBF). We assume inputs and basis are scaled so \(\|\psi(x)\|_\infty\le B\) on the data domain.
- **Polynomials.**
  \[
  P_\theta(x)=\sum_{k=0}^{d_P}\theta_k\,\psi_k(x),\qquad
  Q_\phi(x)=1+\sum_{k=1}^{d_Q}\phi_k\,\psi_k(x)\quad\text{(leading‑1 for identifiability).}
  \]
- **Layer output (scalar case).** \(y(x)=\dfrac{P_\theta(x)}{Q_\phi(x)}\). (Vector/matrix outputs use independent rational heads or shared \(Q\) per group.)

**Identifiability.** Leading‑1 (or alternatively a pinning constraint \(Q(x_0)=1\)) removes trivial scaling \((cP)/(cQ)\equiv P/Q)\.

---

## 2) TR forward semantics (total)

All arithmetic is executed in TR with tags: `REAL`, `PINF`, `NINF`, `PHI`.

- If \(Q(x)\neq 0\) and both \(P(x),Q(x)\) are finite → `REAL` with value \(P/Q\).
- If \(Q(x)=0\) and \(P(x)\ne 0\) → `PINF` if \(P>0\), `NINF` if \(P<0\).
- If \(P(x)=0\) and \(Q(x)=0\) → `PHI` (nullity).
- IEEE export/import uses the bridge: `REAL`↔finite, `±INF`↔`±∞`, `PHI`↔`NaN`. The layer **never** emits NaN internally; TR keeps it as a tag.

**Determinism.** For fixed \((x,\theta,\phi)\) and basis \(\psi\), outputs (value+tag) are deterministic.

---

## 3) Gradients (TR‑AD, Mask‑REAL)

On `REAL` paths, TR‑AD reduces to classical calculus; on non‑REAL, **grads are zero** (Mask‑REAL):
\[
\frac{\partial y}{\partial \theta_k}=\frac{\psi_k(x)}{Q(x)},\qquad
\frac{\partial y}{\partial \phi_k}=-\,\frac{P(x)\,\psi_k(x)}{Q(x)^2}=-\,y\,\frac{\psi_k(x)}{Q(x)}.
\]
- If the node’s forward tag is `PINF/NINF/PHI`, **all partials w.r.t. inputs and params are 0**.
- Composition lemma (operational): if any subnode on a path is non‑REAL, upstream Jacobians are zeroed by Mask‑REAL.

**Optional ablation (Saturating‑grad).** Replace factors \(1/Q^2\) by a TR‑bounded form \(1/(Q^2\oplus 1_R)\) to cap local Lipschitz constants **without ε**. Use only for ablation studies; the default is Mask‑REAL.

---

## 4) Losses (tag‑aware) — **Call‑site reduction policy**

**Per‑sample MSE (scalar):**
\[
\mathcal L_i=\begin{cases}
\tfrac12\,(y_i-y_i^*)^2,& \text{if tag}_i=\text{REAL}\\
\lambda_{\text{rej}},& \text{if tag}_i\in\{\text{PINF,NINF,PHI}\}
\end{cases}
\]

**Policy (explicit).** **Reduction uses _strict_ mode.** Each per‑sample loss is a **REAL** scalar by construction: non‑REAL forward tags map to the constant reject penalty \( \lambda_{\text{rej}}\ge 0 \) (possibly 0). **No PHI can enter aggregations.**

**Classification (CE) sketch.** Apply CE on REAL logits; map non‑REAL logits to a constant \(\lambda_{\text{rej}}\) (or mask out with zero weight). The reduction remains **strict**.

---

## 5) Regularization & stable region

- **L2 on denominator.** \(\Omega(\phi)=\frac{\alpha}{2}\|\phi\|_2^2\) discourages widespread small \(|Q|\).
- **Optional projection.** Enforce \(\|\phi\|_1\le \rho<1/B\) to maintain a **stable region** with \(|Q(x)|\ge 1-\|\phi\|_1 B>0\) over the domain.
- **Practical note.** Track \(q_{\min}=\min_i |Q(x_i)|\) during training; warn if \(q_{\min}\) drifts toward 0.

---

## 6) Well‑posedness & learnability (sketch)

- **Well‑posed REAL slice.** If \(q_{\min}>0\) on the data domain, \(y\) is smooth and the TR‑AD gradients equal classical ones everywhere; optimization is standard.
- **Learning poles.** In 1D with data on both sides of a simple pole, gradients from near‑pole REAL samples adjust \(\phi\) to place zeros of \(Q\) near the true pole. Mask‑REAL prevents unstable contributions from exact singular samples.
- **Identifiability check.** With leading‑1 (or pinning), trivial rescalings are eliminated; nontrivial degenerate solutions (\(Q\equiv 0\)) are discouraged by L2/projection.

---

## 7) Interface & shape contract

```python
class TRRational(nn.Module):
    def __init__(self, d_p:int, d_q:int, basis:Basis, shared_Q=False,
                 lambda_rej:float=0.0, alpha_phi:float=1e-3): ...
    def forward(self, x):
        # returns (y, tag) with TR semantics; no NaNs
        ...
```
- **Inputs.** `x: (..., D)`; basis expands to `(..., K)`.
- **Params.** `theta: (..., d_p+1)`, `phi: (..., d_q+1)` with `phi[0] == 1` (enforced).
- **Outputs.** `(y, tag)` where `y` is numeric (REAL path) and `tag∈{REAL,PINF,NINF,PHI}`.

---

## 8) Tests (this phase)

1) **Forward totality & tagging.** Random \((\theta,\phi,x)\) → op returns one of four tags; determinism holds. Edge cases: `Q=0, P≠0` → ±∞; `P=Q=0` → PHI.
2) **Gradients.** On REAL paths, TR‑AD grads match closed‑form; on non‑REAL, grads are exactly zero (Mask‑REAL). Saturating‑grad ablation stays finite.
3) **Loss policy.** Per‑sample loss is always REAL; reduction with mean/sum/none never sees PHI; gradients from non‑REAL samples are zero.
4) **Regularization effects.** With \(\alpha>0\) (and optional projection), \(\|\phi\|\) does not diverge; \(q_{\min}\) remains bounded away from 0 on synthetic tasks.
5) **Bridge checks.** IEEE export of layer outputs contains no NaN; ±∞ round‑trip to PINF/NINF.

---

## 9) Acceptance (Phase‑3 gate)

- Forward is **total** and deterministic across randomized tests; tag rules match spec.
- REAL‑path gradients equal classical formulas within tolerance; non‑REAL grads are **exact zeros**.
- Loss policy is enforced (**strict reduction**; per‑sample loss always REAL).
- No NaNs in exported streams; basic training on synthetic 1D runs without ε.

---

## 10) Engineering notes

- Prefer numerically stable bases (Chebyshev) for high degrees; monomials fine for low‑degree 1D.
- Keep \(d_Q\) modest initially; use shared‑\(Q\) heads for multi‑output to economize parameters and stabilize \(Q\).
- Log \(q_{\min}\), \(\|\phi\|_1\), and the count of non‑REAL tags per batch to diagnose pole learning vs. instability.

 
# Phase 4 — TR‑Norm (Epsilon‑Free Normalization)

> **Goal.** Define an epsilon‑free normalization with a deterministic bypass at zero variance; prove limit‑equivalence to BN(ε) as ε→0⁺; specify AD rules and tests.

---

## 1) Definition (forward semantics)
**Input.** A tensor `x` of TR scalars along a chosen **feature axis** (e.g., last dim). Let the per‑feature index be `j`, batch/sample indices be `i`.

**REAL subset for stats.** For each feature `j`, define the index set \(S_j=\{ i \mid x_{ij}.tag=\mathrm{REAL}\}\).

**Per‑feature mean/variance (drop‑null).**
\[
\mu_j = \frac{1}{|S_j|} \sum_{i\in S_j} x_{ij}.val,\qquad
\sigma_j^2 = \frac{1}{|S_j|} \sum_{i\in S_j} (x_{ij}.val-\mu_j)^2.
\]
If \(|S_j|=0\) we set \(\mu_j:=0,\ \sigma_j^2:=0\) by convention (forces bypass).

**Normalization (TR arithmetic).** For each element:
\[
\hat x_{ij} = \begin{cases}
\dfrac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2}} & \text{if }\sigma_j^2>0 \\
0 & \text{if }\sigma_j^2=0 \quad \text{(deterministic bypass)}
\end{cases}
\]
where subtraction/division/sqrt are **TR ops** (total). The affine output is \(y_{ij}=\gamma_j\,\hat x_{ij}+\beta_j\) with learnable REAL parameters \(\gamma_j,\beta_j\).

**Tag behavior.**
- If \(\sigma_j^2>0\): REAL inputs remain REAL; PINF/NINF/PHI inputs produce tags via TR arithmetic (e.g., \((\pm\infty - \mu)/\sqrt{\sigma^2}=(\pm\infty)\)).
- If \(\sigma_j^2=0\): **bypass**: we set \(\hat x_{ij}:=0\) with tag REAL for *all* i (so \(y_{ij}=\beta_j\)).

**No ε invariant.** No term of the form `σ² + ε` appears; the branch on `σ²=0` is exact and deterministic.

---

## 2) Proposition — Limit‑equivalence to BN(ε)
**Claim.** For any feature `j` with \(\sigma_j^2>0\), TR‑Norm equals the pointwise limit of standard batch normalization
\[\hat x^{(\varepsilon)}_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\varepsilon}}\]
as \(\varepsilon\to 0^+\). When \(\sigma_j^2=0\), TR‑Norm yields \(\hat x_{ij}=0\) (and \(y_{ij}=\beta_j\)), which equals \(\lim_{\varepsilon\to 0^+}\hat x^{(\varepsilon)}_{ij}\) on any batch where all REAL members are identical (hence \(x_{ij}.val=\mu_j\) and numerator is 0).

**Proof (case analysis).**
- If \(\sigma_j^2>0\), continuity of \(t\mapsto 1/\sqrt{t}\) at \(t=\sigma_j^2\) gives \(\lim_{\varepsilon\to0^+}\frac{1}{\sqrt{\sigma_j^2+\varepsilon}}=\frac{1}{\sqrt{\sigma_j^2}}\), hence \(\hat x^{(\varepsilon)}_{ij}\to \hat x_{ij}\).
- If \(\sigma_j^2=0\) with \(|S_j|\ge 1\), then all REAL entries equal \(\mu_j\), so \(x_{ij}.val-\mu_j=0\) and \(\hat x^{(\varepsilon)}_{ij}=0\) for all \(\varepsilon>0\); the limit is 0 and matches the bypass. If \(|S_j|=0\) the convention forces bypass; BN(ε) is undefined (no REAL stats), but TR‑Norm remains total by design.

---

## 3) Backward (TR‑AD rules)
Let the loss be \(\mathcal{L}=\sum_{i} \ell(y_{ij})\). Gradients are per‑feature `j`.

**Case A — Regular branch (\(\sigma_j^2>0\)).**
- Use **classical BN grads** on REAL paths:
\[
\frac{\partial \hat x_{ij}}{\partial x_{ij}}=\frac{1}{\sqrt{\sigma_j^2}} - \frac{(x_{ij}.val-\mu_j)}{|S_j|\,(\sigma_j^2)^{3/2}} - \frac{1}{|S_j|\sqrt{\sigma_j^2}},
\]
with the usual batch couplings (full formula in code). Then
\(\partial y_{ij}/\partial x_{ij}=\gamma_j\,\partial \hat x_{ij}/\partial x_{ij}\),
\(\partial y_{ij}/\partial \gamma_j = \hat x_{\cdot j}\),
\(\partial y_{ij}/\partial \beta_j = 1\).
- **TR masking.** If a particular element’s forward tag is non‑REAL (PINF/NINF/PHI), then per **Mask‑REAL** (Phase‑2) its local Jacobian **to inputs** is 0.

**Case B — Bypass (\(\sigma_j^2=0\)).**
- We define \(\hat x_{ij}=0\) constant ⇒ **\(\partial \hat x_{ij}/\partial x_{kl}=0\)** for all \(i,k\). Thus
\(\partial y_{ij}/\partial x_{kl}=0\), \(\partial y_{ij}/\partial \gamma_j=0\), and \(\partial y_{ij}/\partial \beta_j=1\) accumulated over the batch.
- This matches the **limit** of BN(ε): since \(\hat x^{(\varepsilon)}_{ij}=0\) for all ε, input and \(\gamma\) gradients vanish.

**No‑NaN guarantee.** All expressions use TR arithmetic or constants; the only potential singularity is \(1/\sqrt{\sigma^2}\), which is guarded by the bypass.

---

## 4) Stability & determinism guarantees
1) **Total forward.** For every input tensor, each feature takes either the regular branch (\(\sigma^2>0\)) or the bypass (\(\sigma^2=0\)); both produce defined TR outputs.
2) **No NaNs.** Neither branch can generate IEEE NaN; bridge only appears at I/O edges.
3) **Deterministic branching.** The branch condition is exact (no ε thresholds), hence reproducible.
4) **Compatibility.** On REAL paths with \(\sigma^2>0\), TR‑Norm ≡ BN(ε→0⁺); at \(\sigma^2=0\), TR‑Norm ≡ affine \(\beta\) path (the continuous limit of shrinking variance).

---

## 5) Microbench: variance→0 stress
Design a one‑parameter family `x(t)` with per‑feature variance \(\sigma^2(t)\downarrow 0\) as \(t\to 0^+\): e.g., `x_i(t)=c + t·u_i` with zero‑mean `u`. Verify numerically:
- \(\|\hat x_{\mathrm{TR}}(t) - \hat x_{\mathrm{BN}(\varepsilon)}(t)\| \to 0\) as \(\varepsilon\to 0^+\) for fixed `t>0`, and \(\hat x_{\mathrm{TR}}(0)=0\).
- Backward: input and \(\gamma\) grads → 0 as \(t\to 0^+\); \(\beta\) grad matches batch size.

---

## 6) Property tests (acceptance)
**A) Equality to BN(ε) as ε→0⁺ (regular branch)**
- Sample random REAL batches with \(\sigma^2>0\). Check \(\max\) abs diff between TR‑Norm and BN(ε) for ε∈{1e−4,1e−6,1e−8} goes to 0 (monotone).

**B) Bypass correctness (σ²=0)**
- Construct batches with constant REAL feature values (or |S|=0). Assert outputs equal \(\beta\); input and \(\gamma\) grads are 0; \(\beta\) grad equals batch size.

**C) No‑NaN fuzz**
- Fuzz with mixtures of REAL/±∞/PHI entries; assert no IEEE NaNs in outputs or grads.

**D) Axis semantics**
- For different feature axes (channels, per‑row), confirm identical behavior modulo reshaping.

---

## 7) Implementation notes
- Compute stats using **drop‑null** over REAL subset (Phase‑0), materialize \(|S_j|\) and guard \(|S_j|=0\) → bypass.
- Keep \(\gamma,\beta\) in REAL; initialize \(\gamma=1,\beta=0\).
- Provide switch `tag_passthrough=False|True` for non‑REAL inputs on regular branch: either compute via TR ops (default) or copy input tag to output directly.
- In frameworks (PyTorch/JAX), implement as a custom autograd primitive using the two‑branch rule above.

---

## 8) Notes & caveats
- When many non‑REALs appear in a feature, \(|S_j|\) may be small; stats become noisy—consider minimum sample guard (optional future work).
- Running‑average (eval mode) is orthogonal; for MVP we operate in **pure batch mode**.

 
# Phase 5 — IEEE↔TR Bridge & ADT View

> **Goal.** Define a precise, invertible mapping between IEEE‑754 double and our TR scalar, and give an ADT‑style specification of TR (with a short wheel contrast). Provide testable properties.

---

## 1) IEEE↔TR Bridge (definitions & properties)

### 1.1 Mapping functions
Let `Float64` denote IEEE‑754 binary64.

**to_tr : Float64 → TR**
```
if isNaN(x):        return (NaN, PHI)
elif x == +∞:       return (NaN, PINF)
elif x == −∞:       return (NaN, NINF)
else:               return (x,   REAL)   # finite (incl. subnormals, ±0.0)
```

**to_ieee : TR → Float64**
```
if tag == REAL:     return val          # must be finite (incl. ±0.0)
elif tag == PINF:   return +∞
elif tag == NINF:   return −∞
elif tag == PHI:    return NaN          # quiet NaN (payload not preserved)
```

**Notes.**
- We *preserve* signed zeros through `val` (±0.0 Round‑trip).  
- NaN payloads are **not** preserved (collapsed to a quiet NaN).
- Subnormals are treated as ordinary finite reals.

### 1.2 Totality & round‑trip laws
For all `x: Float64` and `z: TR`:
- **Totality.** `to_tr(x)` and `to_ieee(z)` are defined for all inputs (no exceptions).
- **Round‑trip (IEEE→TR→IEEE).** `to_ieee(to_tr(x)) = x` for all finite and ±∞ and NaN (up to NaN payload; for NaN we compare via `isNaN`).
- **Round‑trip (TR→IEEE→TR).** `to_tr(to_ieee(z)) = z` for all `z` such that `(tag==REAL ⇒ val is finite)`.

### 1.3 Operation‑awareness (commuting diagrams)
Let `⊕, ⊗, ⊘` be TR ops; let `+_ieee, ×_ieee, ÷_ieee` be IEEE‑754 arithmetic (default rounding) with its exception semantics. For all `a,b ∈ TR` and `x,y ∈ Float64`:

- **Finite REAL slice:** if `a,b` are REAL and all intermediate IEEE ops are defined (i.e., no exceptional case), then
  `to_ieee(a ⊕ b) = to_ieee(a) +_ieee to_ieee(b)` (and similarly for `⊗, ⊘`).
- **Exceptional cases:** whenever IEEE yields NaN or ±∞, TR yields the corresponding tag/value so that
  `to_ieee(op_TR(to_tr(x),to_tr(y)))` and `op_IEEE(x,y)` agree **modulo NaN payload and signed‑zero conventions**. Examples:  
  `x/0 (x>0) → +∞`, `0/0 → NaN↔PHI`, `∞−∞ → NaN↔PHI`, `0×∞ → NaN↔PHI`.

> **Interpretation.** The bridge makes the TR and IEEE computations commute wherever IEEE is defined; in exceptional cases, TR’s tags map to the same IEEE results.

### 1.4 Order‑awareness
Define a partial order `≤_TR`:
- `NINF <_TR REAL values <_TR PINF` with the usual real order inside REAL.
- `PHI` is **unordered**: no comparisons to PHI are true (except `is_phi`).

Compatibility:
- On finite reals, `≤_TR` coincides with IEEE comparisons.
- IEEE NaN is unordered; via the bridge, this matches PHI’s unordered status.

### 1.5 Determinism & precision
- The bridge is *pure* and deterministic; it does not depend on rounding modes.  
- No value changes on REAL inputs: `val` is the exact IEEE value (including ±0.0 bit).  
- Infinity signs are preserved.

---

## 2) TR as an ADT (algebraic data type)

### 2.1 Signature
Constructors: `Real(r)` for `r∈ℝ` (finite), `Pinfty`, `Ninfty`, `Phi`.
Operations: `add, sub, mul, div, abs, sign, log, sqrt, pow_int`.

### 2.2 Equations (selected; total semantics)
Let `R,S` denote `Real` values with payloads `r,s ∈ ℝ`.
- **Addition**
  - `add(R,S) = Real(r+s)`
  - `add(R,Pinfty)=Pinfty`, `add(R,Ninfty)=Ninfty` (and symmetric)
  - `add(Pinfty,Pinfty)=Pinfty`, `add(Ninfty,Ninfty)=Ninfty`
  - `add(Pinfty,Ninfty)=Phi` (and symmetric)
  - `add(Phi,_) = Phi = add(_,Phi)`
- **Multiplication**
  - `mul(R,S)=Real(r·s)`; `mul(R≠0, Pinfty)=Pinfty` with sign; `mul(0,Pinfty)=Phi`
  - `mul(Pinfty,Ninfty)=Ninfty` (sign), `mul(Pinfty,Pinfty)=Pinfty`
  - `mul(Phi,_) = Phi = mul(_,Phi)`
- **Division**
  - `div(R,S≠0)=Real(r/s)`; `div(R,0)=Pinfty/Ninfty` by sign; `div(0,0)=Phi`
  - `div(Pinfty,S≠0)=Pinfty` with sign; `div(R,Pinfty)=Real(0)`
  - `div(Pinfty,Pinfty)=Phi`; `div(Phi,_) = Phi = div(_,Phi)`
- **Unary** (domain‑aware, total via Phi)
  - `log(Real(x>0))=Real(ln x)`, else `Phi`
  - `sqrt(Real(x≥0))=Real(√x)`, else `Phi`
  - `pow_int(x,k)` via repeated `mul/div`; `pow_int(0,0)=Phi`, `pow_int(Pinfty,0)=Phi`

**Remarks.** Associativity/distributivity hold on the REAL slice; Phi is absorptive for most mixed cases.

---

## 3) Wheel (Σ_w, E_w) — short contrast (optional appendix)
**Wheel constants & ops.** `0,1,∞,⊥ ; +,·,−,( )^{-1}` with `x/y := x·y^{-1}`.

**Control axioms (highlights).**
- `1/0 = ∞`, `1/∞=0`.
- `0·∞ = ⊥` (strict error instead of TR’s Phi).
- `∞+∞ = ⊥` (wheel forbids adding infinities coherently).
- Propagation: `x+⊥=⊥`, `x·⊥=⊥`, `−⊥=⊥`.

**Design delta vs TR.**
- Wheel is **stricter** (more cases collapse to ⊥), useful for mask‑as‑algebra experiments.  
- TR is **flow‑oriented** (keeps ±∞ as first‑class values; uses Phi only when necessary), better for autodiff and keeping computations alive.

---

## 4) Tests & acceptance (repo checklist)

### 4.1 Bridge round‑trip tests
- **IEEE→TR→IEEE:** sample floats from bins {finite (incl. subnormals, ±0), ±∞, NaN(payloads)}; assert identity modulo NaN payload.  
- **TR→IEEE→TR:** sample TR values; assert identity when `tag==REAL ⇒ val` is finite.

### 4.2 Operation‑commutation tests
For random `x,y: Float64` and corresponding `a=to_tr(x), b=to_tr(y)`:
- Compute `u = to_ieee(add(a,b))` and `v = x +_ieee y`; assert `u==v` using a comparator that treats NaN≈NaN, +0≈−0. Repeat for mul/div and selected unary ops.

### 4.3 Order tests
- On random finite reals, `compare_TR(a,b)` equals IEEE comparison.
- Any comparison involving `Phi` returns **false** (except tag checks).

### 4.4 Fuzz (acceptance gate)
- Generate random expression trees over {add,mul,div,log,sqrt,pow_int} with IEEE leaves; check that bridging in/out at the **ends** yields the same IEEE outputs as computing purely in IEEE (mod NaN payload and signed‑zero) while TR never raises and internal states are tagged.

---

## 5) Implementation notes
- Use explicit QNaN construction in `to_ieee(PHI)`; do not reuse payloads.
- Keep bridge functions centralized; forbid implicit casts in core ops.
- Provide an IEEE comparator utility: `eq_ieee(u,v)` that treats NaN≈NaN and +0≈−0 for testing.
- Vectorize bridge functions for tensors; ensure they are JIT/Numba friendly.

 
# Phase 6 — Optimization & Convergence Near Poles (Spec + Proof Sketch + Tests)

> **Goal.** Give crisp conditions for stable training of TR‑rational layers near singularities; prove bounded per‑step updates; derive a **safe LR window** in 1D; restate identifiability/regularization needed to avoid degenerate denominators.

---

## 1) Setting & notation

- **Model (1D for clarity).** \(y(x)=\dfrac{P_\theta(x)}{Q_\phi(x)}\), with
  \(P_\theta(x)=\sum_k \theta_k\,\psi_k(x)\), \(Q_\phi(x)=1+\sum_{k\ge 1}\phi_k\,\psi_k(x)\) (leading‑1 in \(Q\) for identifiability).
- **Basis scaling.** Assume inputs/basis scaled so \(\|\psi(x)\|_\infty\le B\) on the data domain (e.g., \([-1,1]\)). This basis assumption is used to bound \(|Q|\) and the Lipschitz surrogate below.
- **TR forward.** Returns tags `REAL/±INF/PHI`; operations are total.
- **TR‑AD (Mask‑REAL).** On `REAL` paths, use classical calculus; if the node’s forward tag is `±INF/PHI`, **all partials to inputs and params are 0**.
- **Param grads on REAL paths.**
  \[
  \frac{\partial y}{\partial \theta_k}=\frac{\psi_k(x)}{Q(x)},\qquad
  \frac{\partial y}{\partial \phi_k}=-\,\frac{P(x)\,\psi_k(x)}{Q(x)^2}=-\,y\,\frac{\psi_k(x)}{Q(x)}.
  \]
- **Per‑sample loss (MSE).**
  \[
  \mathcal L_i=\begin{cases}
  \tfrac12\,(y_i-y_i^*)^2,& \text{if tag}_i=\text{REAL}\\
  \lambda_{\text{rej}},& \text{if tag}_i\in\{\text{PINF,NINF,PHI}\}
  \end{cases}
  \]
  **Reduction policy (call‑site).** Use **strict** reduction; each per‑sample loss is **REAL** by construction (non‑REAL forwards map to the constant \( \lambda_{\text{rej}}\ge 0 \), possibly 0). **No PHI can enter aggregations.**

Define per batch (over REAL indices \(S_R\)):
\[
q_{\min} := \min_{i\in S_R} |Q(x_i)|,\quad
B_\psi := \max_{i\in S_R}\|\psi(x_i)\|_2,\quad
y_{\max} := \max_{i\in S_R}|y(x_i)|,\quad
\|e\|_2 := \Big(\sum_{i\in S_R} (y_i-y_i^*)^2\Big)^{1/2}.
\]

---

## 2) Boundedness of one‑step updates (Mask‑REAL)

For full‑batch gradient descent with step \(\eta>0\),
\[
\|\Delta\theta\|_2 \le \eta\,\frac{\|e\|_2\,B_\psi}{q_{\min}},\qquad
\|\Delta\phi\|_2 \le \eta\,\frac{\|e\|_2\,B_\psi\,y_{\max}}{q_{\min}}.
\]
*Sketch.* Bound each coordinate by the gradient formulas above and sum in quadrature over \(S_R\). **Non‑REAL samples contribute zero** under Mask‑REAL, so poles cannot cause gradient explosions.

---

## 3) Stable region away from poles

**Basis assumption used here (‖ψ‖∞ ≤ B).** Because sup_x Σ_{k≥1} |φ_k|·|ψ_k(x)| ≤ ‖φ‖₁·B, it follows that |Q_φ(x)| ≥ 1 − ‖φ‖₁·B for all in-domain x. The bounds below use this assumption.

From the basis assumption, for all \(x\) in domain: \(|Q_\phi(x)|\ge 1-\|\phi\|_1 B\). Hence if \(\|\phi\|_1 < 1/B\) then \(q_{\min}>0\) uniformly; the model is smooth on the domain and TR‑AD equals classical AD along these paths. We call this the **stable region**.

**Keeping the region:**
- **Identifiability:** leading‑1 fixes \(Q\)’s scale.
- **Regularization:** \(\Omega(\phi)=\tfrac{\alpha}{2}\|\phi\|_2^2\) discourages tiny \(|Q|\); optionally project \(\|\phi\|_1\le \rho<1/B\).

---

## 4) Smoothness of the REAL‑slice loss

**Basis assumption used here (‖ψ‖∞ ≤ B).** With K basis terms, ‖ψ(x)‖₂ ≤ √K·B for all in-domain x; hence a worst-case bound B_ψ ≤ √K·B can be used in (★) if you don’t want to compute B_ψ empirically per batch. This makes the dependence of L_batch on the basis scaling explicit.

With \(S_R\) fixed, MSE+\(\alpha\|\phi\|_2^2\) is \(L\)-smooth in \((\theta,\phi)\). A **batch‑wise Lipschitz surrogate**:
\[
L_{\text{batch}}
\;\;\le\;\;
\frac{1}{|S_R|}\sum_{i\in S_R}\frac{\|\psi(x_i)\|_2^2}{|Q(x_i)|^2}\,\Big(1+y(x_i)^2\Big)
\;+\;\alpha
\;\;\le\;\;
\frac{B_\psi^2}{q_{\min}^2}\,\Big(1+y_{\max}^2\Big) \;+\; \alpha.
\tag{★}
\]

---

## 5) Safe learning‑rate window (monotone descent)

For gradient descent on the REAL subset,
\[
0<\eta<\tfrac{2}{L_{\text{batch}}}\ \Rightarrow\ \mathcal{L}(\Theta-\eta\nabla\mathcal{L})\le \mathcal{L}(\Theta)-\eta\Big(1-\tfrac{\eta L_{\text{batch}}}{2}\Big)\|\nabla\mathcal{L}\|_2^2.
\]
Choose
\[
\boxed{\ \eta_{\text{safe}} = 1/L_{\text{batch}}\ }
\]
for guaranteed **monotone early** decrease. For Adam/AdamW, a conservative engineering rule is \(\eta_{\text{safe,Adam}}\approx (1-\beta_1)/L_{\text{batch}}\).

**Runtime recipe (per batch).**
1) Collect REAL indices \(S_R\).  
2) Compute \(q_{\min}, y_{\max}, B_\psi\).  
3) Form \(L_{\text{batch}}\) by (★); set \(\eta=\min(\eta_{\text{user}}, 1/L_{\text{batch}})\).

Mask‑REAL ensures if a point flips to `±INF/PHI`, it **drops from** \(S_R\) and cannot destabilize the step.

---

## 6) Worked 1D example (single‑pole family)

Let \(\psi(x)=(1,x,\dots,x^{d})\) on \([-1,1]\) so \(B_\psi\le\sqrt{d{+}1}\), \(B=1\). With \(Q(x)=1+\sum_{k=1}^{d_Q}\phi_k x^k\), we have \(q_{\min}\ge 1-\|\phi\|_1\). If \(q_{\min}\) shrinks, (★) inflates via \(1/q_{\min}^2\) → \(\eta\) auto‑shrinks → bounded updates.

**Ablation (saturating‑grad).** Replace \(1/Q^2\) by \(1/(Q^2\oplus 1_R)\) to cap local Lipschitz terms without ε; keep default as Mask‑REAL.

---

## 7) Identifiability & anti‑degeneracy

- **Scaling ambiguity removed** by the leading‑1 in \(Q\) (or pinning \(Q(x_0)=1\)).
- **Avoid “everywhere‑zero \(Q\)”**: use \(\alpha>0\) and optional \(\ell_1\) projection to keep \(q_{\min}>0\) on the training support.

---

## 8) Loss on TR outputs — design notes

- Compute loss **only on REAL predictions**; non‑REAL tags map to a constant reject penalty \(\lambda_{\text{rej}}\). Under Mask‑REAL its gradient contribution is 0.
- **Reduction policy:** **strict** (per‑sample values are always REAL). No PHI can enter reductions or metrics.
- Upstream **TR‑Norm** is ε‑free and total; at zero variance it deterministically bypasses to \(\beta\), so Phase‑6 analyses never see NaNs or undefined branches.

---

## 9) Acceptance tests (Phase‑6 gate)

**A) Monotone descent under \(\eta_{\text{safe}}\).** On 1D synthetic tasks (e.g., \(f(x)=\tfrac{2x+1}{x-3}\) on \([-1,6]\)), use \(\eta=\min(\eta_0,1/L_{\text{batch}})\). Assert non‑increasing batch loss for the first \(K\) epochs; fewer restarts/NaNs than ε‑baseline.

**B) Bounded updates.** Log \n\|\Delta\theta\|_2, \n\|\Delta\phi\|_2 and verify the bounds of §2 using measured \(q_{\min},y_{\max},B_\psi,\|e\|_2\) (within slack).

**C) Stable‑region invariance.** Track \(q_{\min}\) and \(\|\phi\|_1\). With \(\alpha>0\) (and optional projection), verify \(\|\phi\|_1<1/B\) hence \(q_{\min}>0\).

**D) Mask‑REAL safety.** Build batches with exact singulars (`Q=0`, or `P=Q=0`); confirm zero gradient from those samples while REAL samples update normally.

**E) Ablation.** With saturating‑grad enabled, confirm tighter empirical \(L_{\text{batch}}\) and step sizes, still ε‑free.

---

## 10) Drop‑in code (runtime guard)

```python
# Given: batch {x_i}, REAL mask M, features psi(x), y=P/Q
q_min   = min(abs(Q[i]) for i in M)
y_max   = max(abs(y[i]) for i in M)
Bpsi2   = max(np.linalg.norm(psi(x[i]))**2 for i in M)
L_batch = (Bpsi2 / (q_min**2)) * (1 + y_max**2) + alpha
eta     = min(user_eta, 1.0 / L_batch)
# step with eta (GD) or (1-beta1)/L_batch for Adam‑ish
```

---

## 11) What this buys us

- **Total, NaN‑free** training (TR ops + IEEE bridge + TR‑Norm bypass).
- **No ε‑hacks**: singular forwards flip tags; gradients from those samples are zero; updates remain bounded.
- **Concrete LR rule** from batch observables \((q_{\min},y_{\max},B_\psi)\) that **shrinks adaptively** near poles while still allowing the model to **learn pole locations** from near‑pole REAL samples.

 
# Phase 7 — Verification Harness (Spec + Property Tests + CI)

> **Goal.** Build a property‑based test harness that mirrors the math: algebraic laws on the REAL slice, Wheel‑mode sanity checks, TR‑AD equivalences, TR‑Norm limits, and end‑to‑end “no‑NaN” guarantees.

---

## 0) Scope & definitions

- **TR tags:** `REAL`, `PINF`, `NINF`, `PHI` (nullity).
- **Wheel‑mode (optional ADT track):** a stricter algebra where certain expressions evaluate to bottom `⊥` (e.g., `0·∞=⊥`, `∞+∞=⊥`).
- **Mask‑REAL AD:** classical grads on REAL paths; zero‑grad when the forward tag is non‑REAL.
- **Stable‑region guard:** for derivative comparisons, require `|Q(x)| ≥ τ` with small `τ>0` to avoid ill‑conditioned finite differences.

---

## 1) Repo layout

```
polenet/
  tests/
    property/
      test_tr_ops.py          # +, −, ×, ÷ totality & laws (REAL slice)
      test_ieee_bridge.py     # IEEE↔TR round‑trip & totality
      test_wheel_mode.py      # optional wheel checks (⊥ cases)
      test_tr_ad.py           # chain rule equivalence & mask‑REAL
      test_tr_norm.py         # ε‑free limit tests vs BN(ε→0⁺)
    e2e/
      test_train_1d.py        # random 1D runs: no‑NaN, monotone early loss
      test_train_tabular.py   # TR‑Norm end‑to‑end, label noise stress
  pyproject.toml              # pytest, hypothesis, coverage
  .github/workflows/ci.yml    # CI matrix + badges
```

---

## 2) Hypothesis strategies

Core building blocks (Python):

```python
from hypothesis import strategies as st

# Finite reals in a safe numeric band
st_real = st.floats(allow_nan=False, allow_infinity=False,
                    width=64, min_value=-1e6, max_value=1e6)

# TR tagged values (REAL or special)
st_tag = st.sampled_from(["REAL","PINF","NINF","PHI"])  # for tags only

# Structured TR value; if tag==REAL carry a float payload
@st.composite
def st_tr(draw):
    tag = draw(st.sampled_from(["REAL","PINF","NINF","PHI"]))
    if tag == "REAL":
        return (tag, draw(st_real))
    return (tag, None)

# Random polynomials for P/Q with leading-1 in Q
@st.composite
def st_rational(draw, deg_p=st.integers(0,4), deg_q=st.integers(1,4)):
    dp, dq = draw(deg_p), draw(deg_q)
    theta = [draw(st_real) for _ in range(dp+1)]
    phi   = [1.0] + [draw(st_real) for _ in range(dq)]  # leading-1
    return theta, phi
```

Notes:
- Add `assume(|Q(x)| ≥ τ)` when comparing derivatives; shrink `τ` gradually (e.g., `1e-4`).
- Use `@settings(deadline=None, max_examples=2000)` for heavy properties in CI nightlies; lighter profiles for PR runs.

---

## 3) Algebraic properties (REAL slice)

Implement these **only when all involved intermediates are REAL**; otherwise skip via `hypothesis.assume`.

- **Commutativity:** `x⊕y == y⊕x` and `x⊗y == y⊗x` (on REALs).
- **Associativity:** `(x⊕y)⊕z == x⊕(y⊕z)` and `(x⊗y)⊗z == x⊗(y⊗z)` (REALs).
- **Distributivity (restricted):** `x⊗(y⊕z) == x⊗y ⊕ x⊗z` if all 3 sums/products are REAL.
- **Embedding:** `embed(r)` maps `r∈ℝ` to TR with `tag=REAL` and preserves +, ×.
- **Totality:** For all TR `a,b`, each op returns a **well‑formed** TR value (never throws; tag∈{REAL,PINF,NINF,PHI}).

Example (sketch):

```python
from hypothesis import given, assume

def is_real(tv): return tv.tag=="REAL"

@given(x=st_tr(), y=st_tr())
def test_totality_add(x,y):
    z = tr_add(x,y)
    assert z.tag in {"REAL","PINF","NINF","PHI"}

@given(x=st_tr(), y=st_tr(), z=st_tr())
def test_assoc_add_real_slice(x,y,z):
    assume(is_real(x) and is_real(y) and is_real(z))
    left  = tr_add(tr_add(x,y), z)
    right = tr_add(x, tr_add(y,z))
    assume(is_real(left) and is_real(right))
    assert left.value == pytest.approx(right.value)
```

---

## 4) Wheel‑mode checks (optional ADT track)

Enable the wheel code path and verify distinctive bottoms:

- `0·∞ = ⊥`
- `∞+∞ = ⊥`
- Propagate `⊥` through all ops (`⊥ ⊕ a = ⊥`, `⊥ ⊗ a = ⊥`, etc.).
- Round‑trip consistency: disabling wheel restores TR semantics.

```python
@given()
def test_wheel_zero_times_inf_bottom():
    with wheel_mode():
        assert tr_mul(TR.real(0.0), TR.pinfty()).is_bottom()
```

---

## 5) IEEE↔TR bridge tests

- **Round‑trip:** `x_float → TR → float` preserves finite numbers; `±∞ ↔ PINF/NINF`; `NaN ↔ PHI`.
- **Operation‑aware mapping:** The bridge refuses to emit IEEE `NaN` on any total TR op; instead, the TR value carries the tag.
- **No‑NaN invariant:** For **all** randomized op pipelines, the exported IEEE stream contains no `NaN` (use a metamorphic “compose random ops then export” test).

---

## 6) AD checks

### (A) Chain‑rule equivalence on REAL paths

For random `(θ,φ)` and inputs with `|Q(x)|≥τ`, verify that TR‑AD equals classical derivatives (either analytic or finite‑diff):

```python
@given(st_rational(), st.lists(st_real, min_size=8, max_size=64))
def test_chain_rule_real_slice(rational, xs):
    theta, phi = rational
    for x in xs:
        if abs(Q(phi,x)) < 1e-4: continue  # skip near-pole
        y, tag = forward_tr(theta, phi, x)
        assert tag=="REAL"
        g_classic = grads_classic(theta, phi, x)  # closed-form
        g_tr_ad   = grads_tr_ad(theta, phi, x)    # our AD
        assert_allclose(g_tr_ad.theta, g_classic.theta, rtol=1e-5, atol=1e-7)
        assert_allclose(g_tr_ad.phi,   g_classic.phi,   rtol=1e-5, atol=1e-7)
```

### (B) Mask‑REAL triggers exactly at non‑REAL tags

Construct inputs yielding `P/Q` with tags in `{PINF,NINF,PHI}` and assert **zero** parameter‑grads from those samples while REAL samples contribute normally.

```python
def test_mask_real_zero_grads_on_nonreal():
    batch = make_batch_mixed_tags()  # contains REAL + PINF/NINF/PHI
    loss, grads = loss_and_grads(batch)
    assert grads.from_nonreal_samples_is_zero()
    assert grads.from_real_samples_is_nonzero()
```

### (C) TR‑Norm limit tests

- For batches with variance `> 0`, TR‑Norm output equals the **ε→0⁺ limit** of standard BN/LayerNorm (within tolerance).
- For zero‑variance batches, TR‑Norm **bypasses** to `β` (deterministic), while BN(ε) converges to the same limit as ε→0⁺.

```python
@given(st_batch_normal_like(min_var=1e-3))
def test_trnorm_equals_bn_limit(batch):
    y_tr  = tr_norm(batch, gamma, beta)
    y_bn1 = bn(batch, gamma, beta, eps=1e-3)
    y_bn2 = bn(batch, gamma, beta, eps=1e-6)
    # Extrapolate eps→0 with Richardson-like refinement
    y_extrap = (1e-3*y_bn2 - 1e-6*y_bn1)/(1e-3-1e-6)
    assert_allclose(y_tr, y_extrap, rtol=1e-5, atol=1e-6)

@given(st_batch_all_equal())
def test_trnorm_zero_variance_bypass(batch):
    y_tr = tr_norm(batch, gamma, beta)
    assert_allclose(y_tr, beta)
```

---

## 7) End‑to‑end properties

- **E2E‑No‑NaN:** Random training runs (synthetic 1D rational target; modest width) produce **zero NaNs** in activations, params, and losses. Check every step.
- **Monotone early loss:** With the Phase‑6 `η_safe`, the first `K` epochs have non‑increasing batch loss on the REAL subset.
- **Pole‑robustness:** Randomly sprinkle near‑pole inputs; ensure learning still proceeds (REAL samples update) and the run terminates without restarts.

```python
def test_e2e_no_nan_and_monotone():
    stats = train_random(seed=0, steps=500, eta_rule="safe")
    assert not stats.any_nan
    assert is_monotone_nonincreasing(stats.loss_first_K)
```

---

## 8) Metrics, coverage & CI

- **Coverage:** `pytest --cov=polenet --cov-report=xml` (target ≥95%). Property coverage = % of lines/branches touched by Hypothesis suites.
- **Hypothesis profiles:**
  - `dev`: fast checks, `max_examples=200`, `derandomize=True` (seeded).
  - `ci`: thorough, `max_examples=2000`, `deadline=None`.
- **CI (GitHub Actions):** Python {3.10, 3.11, 3.12}; cache pip; run `ruff`/`mypy`; then tests.
- **Badges (README):** Build, Coverage, Property‑Suite Pass, No‑NaN E2E ✔.

Example workflow:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix: { python-version: ["3.10","3.11","3.12"] }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: ${{ matrix.python-version }} }
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
      - run: pip install -e .[dev]
      - run: pytest -q --maxfail=1 --disable-warnings
      - run: pytest -q -m property --cov=polenet --cov-report=xml
```

Badge snippet (README):

```markdown
![Build](https://img.shields.io/github/actions/workflow/status/ORG/REPO/ci.yml?branch=main)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/USER/GIST/raw/coverage.json)
![Property Suite](https://img.shields.io/badge/property%20tests-passing-brightgreen)
![E2E No‑NaN](https://img.shields.io/badge/e2e%20no‑NaN-✔-brightgreen)
```

---

## 9) Acceptance criteria (Phase‑7 gate)

- **≥95%** line coverage on `core/`, `layers/`, `bridge/` via property suites.
- **Zero NaNs** across randomized E2E runs (exported IEEE streams must be NaN‑free).
- **Wheel track** passes bottom checks when enabled; TR track unaffected.
- **AD checks**: chain‑rule equivalence holds on REAL slice within numeric tolerances; mask‑REAL exact zeros on non‑REAL.
- **TR‑Norm** matches BN(ε→0⁺) on positive‑variance batches and bypasses to `β` when variance=0.

---

## 10) Stretch tests (nice‑to‑have)

- **Metamorphic testing:** invariants under input scaling for rational layers (given leading‑1 in `Q`).
- **Fuzzed IEEE bridge:** random op graphs (depth 1–5) over TR values, then export → assert `not any_nan`.
- **Differential testing:** compare TR‑rational (REAL slice) vs standard rational with small ε on batches far from poles; results should match within tolerance.

