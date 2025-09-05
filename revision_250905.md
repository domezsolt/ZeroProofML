# Revision 2025-09-05 — Spec Clarifications & Changes (Normative)

> **Scope.** This section is normative and overrides any conflicting statements elsewhere in this document.
> It clarifies PHI semantics, stability conditions, AD rules near poles, compiler rewrites, Wheel-mode isolation,
> evaluation-only thresholds, determinism guarantees, and loss policy for non‑REAL tags.

---

# Extensions: Addressing Critiques

To strengthen the methodology against common reviewer criticisms, we explicitly add two enhancement packages:

## A) Anti “Dropped Sample” Package (Mask‑REAL criticism)

**Issue.** Mask‑REAL zeroes gradients on non‑REAL tags (±∞, Φ). A reviewer may claim this ignores the hardest examples.

**Enhancements.**
1. **Hybrid gradient schedule.** Training starts with Mask‑REAL. After N₁ epochs, selectively enable Saturating‑grad *only* for inputs with |Q(x)| ≤ δ. δ decays (e.g. 1e−2 → 1e−6). This lets near‑pole points contribute finite gradients while keeping initial stability.
2. **Tag‑loss.** Non‑REAL outputs incur an auxiliary classification loss, predicting PINF/NINF/PHI. This injects information from non‑REAL samples instead of discarding them.
3. **Coverage control.** Oversample near‑pole regions and adapt λ₍rej₎ to maintain a target REAL coverage c*. Prevents the model from “escaping” singular zones by trivially rejecting too much.
4. **Pole head.** Add a small auxiliary network that predicts where Q(x) ≈ 0. Train it with either analytical teacher signals (e.g. det J in robotics) or self‑supervised proxies. This explicitly learns pole locations.

**Effect.** Non‑REAL samples are no longer “silent”; they provide supervision via tag‑loss and pole localization, while still avoiding gradient explosions.

---

## B) Anti “Extrapolation Illusion” Package

**Issue.** A reviewer may argue that TR just re‑labels undefined cases, without truly learning pole behavior.

**Enhancements.**
1. **Pole Localization Error (PLE).** Introduce metrics: distance between learned pole set (zeros of Q) and ground‑truth poles (e.g. det J=0). Report PLE in benchmarks.
2. **Sign‑consistency checks.** On paths crossing a pole, verify the model flips between +∞ and −∞ correctly. Penalize inconsistencies with a dedicated loss.
3. **Asymptotic slope loss.** Near simple poles, enforce log|y| ∼ −log|Q| behavior, encouraging the model to approximate the true asymptotic.
4. **Residual consistency.** Penalize deviation of R(x)=Q(x)·y(x)−P(x). REAL near‑pole samples must satisfy R≈0, forcing structural coherence.
5. **Teacher/proxy signals.** Where available (robotics: det J, physics: singular mass matrices), supervise the pole‑head with analytical pole indicators. In domains without analytic labels, use proxies (e.g. DLS instability) as weak supervision.

**Effect.** The model does not merely categorize infinities but actively learns the geometry and asymptotics of singularities, addressing the “illusion” concern.

---

# Integration into Core Spec

These packages integrate with existing sections as follows:
- **TR‑AD (§5).** Mask‑REAL remains default; Saturating‑grad can be scheduled locally near poles.
- **Loss policy (§9).** Add auxiliary tag‑loss for non‑REAL outputs, alongside λ₍rej₎.
- **Training (§6).** CoverageTracker and Adaptive λ₍rej₎ are extended to enforce target near‑pole coverage.
- **Evaluation (§7).** Add pole‑specific metrics: PLE, sign‑consistency, asymptotic slope, residual error.
- **Layers (§3).** Optionally attach pole‑head modules for Q‑zero detection.

---

# Summary

By embedding these anti‑critique packages into the core design, ZeroProofML ensures that:
- Non‑REAL samples contribute useful supervision (not ignored).
- Extrapolation near poles is empirically verified and structurally enforced.
- Benchmarks can directly rebut “dropped sample” and “extrapolation illusion” objections with quantitative evidence.

