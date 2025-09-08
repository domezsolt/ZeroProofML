# ZeroProofML Library To‑Do List (Revision 2025‑09‑07)

This to‑do list integrates earlier core requirements with the new normative changes from **Revision 2025‑09‑05**, insights from example testing, and critical gaps identified in the implementation.

---

## 0. Critical Fixes (Immediate Priority)

### 0.1 Dataset Generation Issues ✅
- [x] **Fix singular point generation**: Created SingularDatasetGenerator that includes actual singular points
- [x] **Implement forced singularity exploration**: Samples placed exactly at Q(x) = 0 positions
- [x] **Improve near-pole sampling**: Using exponential distribution for distance from poles
- [x] **Add ground-truth pole annotations**: Metadata includes true pole locations and tag distribution

### 0.2 Coverage Control Effectiveness ✅
- [x] **Fix coverage control**: λ_rej now has minimum threshold and asymmetric updates
- [x] **Implement minimum λ_rej**: Never drops below 0.1 to maintain exploration pressure
- [x] **Add coverage violation penalties**: Dead-band control and faster increase when coverage too high

---

## 1. Core (Normative) Tasks

### 1.1 Arithmetic & Autodiff
- [ ] **Mask‑REAL rule** remains the default: gradients flow only on REAL outputs, zero otherwise.
- [ ] **Hybrid gradient schedule** (needs improvement):
  - Training starts with Mask‑REAL.
  - After N₁ epochs, enable Saturating‑grad *only* for inputs with |Q(x)| ≤ δ.
  - Implement δ‑decay (e.g., 1e−2 → 1e−6).
  - **Add**: Force exploration of δ-neighborhoods around detected poles
- [ ] Expose schedule parameters (`N1`, initial δ, decay rate) in training config.
- [ ] **Add q_min tracking**: Monitor minimum |Q(x)| per batch to verify near-pole exploration

### 1.2 Loss Policy
- [ ] Maintain **λ₍rej₎** as rejection penalty (strict reduction).
- [ ] Add **auxiliary tag‑loss** (currently underutilized):
  - Classification of non‑REAL outputs (PINF, NINF, PHI).
  - Integrated into the loss function alongside λ₍rej₎.
  - **Add**: Increase tag_loss weight when coverage is too high
- [ ] **Add minimum rejection penalty**: λ_rej_min parameter to prevent complete avoidance

### 1.3 Training & Coverage ✅
- [x] Extend **CoverageTracker**: (Completed)
  - Track both global REAL coverage and **near‑pole coverage**.
  - **Add**: Track actual non-REAL outputs (currently always 0%)
  - **Add**: Monitor distance to nearest singularity for each sample
- [x] Update **Adaptive λ₍rej₎** (currently too weak): (Completed)
  - Control target REAL coverage `c*`.
  - Prevent trivial solutions where the model rejects too many near‑pole samples.
  - **Add**: Asymmetric updates (faster increase, slower decrease)
  - **Add**: Dead-band around target coverage to prevent oscillation
- [x] Add support for **oversampling near poles** in training data pipelines: (Completed)
  - **Priority**: This is critical - current sampling is inadequate
  - Implement adaptive grid refinement near detected poles
  - Weight sampling by 1/|Q(x)| proximity

### 1.4 Layers ✅
- [x] Extend **TRRational layer**: (Completed)
  - Optional **pole‑head module** predicting regions where Q(x) ≈ 0.
  - Interface to attach pole‑head to TRRational or TRRationalMulti.
  - **Improve pole detection accuracy** (currently only 40%):
    - Better initialization strategy for pole head
    - Increase pole loss weight in total loss
- [x] Ensure pole‑head integrates with loss (tag‑loss, PLE) and supervision (teacher/proxy). (Completed)
- [x] **Add pole regularization**: Encourage Q(x) to have poles at specific locations during training (Completed)

### 1.5 Evaluation & Metrics ✅
- [x] Implement new metrics (currently missing): (Completed)
  - **Pole Localization Error (PLE):** distance between learned poles and ground truth.
  - **Sign‑consistency:** check correct +∞ ↔ −∞ flips across poles.
  - **Asymptotic slope loss:** enforce log|y| ∼ −log|Q|.
  - **Residual consistency:** R(x) = Q(x)·y(x) − P(x) ≈ 0 near poles.
  - **Add**: Actual singularity count vs predicted count
  - **Add**: Coverage breakdown by distance from pole (near/mid/far)
- [x] Integrate metrics into evaluation API and training logs. (Completed)
- [x] **Add visualization tools**: (Completed)
  - Plot learned Q(x) vs ground truth
  - Visualize pole locations in input space
  - Show gradient flow near singularities

### 1.6 Wheel Mode ✅
- [x] Preserve strict **isolation of Wheel mode**: (Completed)
  - TR and Wheel semantics must never mix inside a single operation.
  - Wheel remains a compile‑time switch mapping Φ→⊥ and enforcing wheel axioms.

---

## 2. Optional / Experimental Tasks

### 2.1 Loss & Control
- [ ] **Dead‑band or PI controller** for λ₍rej₎ (stability improvement).
- [ ] Ablations comparing proportional vs PI λ‑update.
- [ ] **Add curriculum learning**: Start with easy (far from poles) samples, gradually introduce harder ones

### 2.2 Supervision
- [ ] **Teacher / proxy supervision** for pole‑head (critical for improving 40% accuracy):
  - Robotics: det(J) as analytic label.
  - Physics: singular mass matrices.
  - Proxy: instability signals as weak supervision.
  - **Add**: Pre-train pole head on synthetic data with known poles

### 2.3 Sampling & Diagnostics
- [ ] Active sampling in pole neighborhoods (adaptive x‑grid refinement).
- [ ] **Critical**: Implement importance sampling with weight ∝ 1/|Q(x)|²
- [ ] Diagnostic API to export **P(x), Q(x), q_min** for monitoring.
- [ ] Logging improvements:
  - Stable keys in training history (`lambda_rej`, `coverage_train`, `coverage_eval_tau`, etc.).
  - Batch‑wise q_min, tag distributions, and pole metrics.
  - **Add**: Log actual number of PINF/NINF/PHI outputs per epoch
  - **Add**: Track gradient magnitudes near poles vs far from poles

---

## 3. Testing & CI

### 3.1 Unit Tests
- [ ] Property‑based tests for TR ops (totality, closure, embedding).
- [ ] Gradient equivalence on REAL paths vs analytic formulas.
- [ ] Zero‑grad property for non‑REAL paths (Mask‑REAL, hybrid schedule).
- [ ] Tag‑loss correctness (classification of non‑REAL outputs).
- [ ] **Add**: Test that models actually produce non-REAL outputs when expected
- [ ] **Add**: Verify gradient flow through pole detection head

### 3.2 Integration Tests
- [ ] End‑to‑end run with synthetic rational regression:
  - Coverage adapts to target c*.
  - λ₍rej₎ stabilizes.
  - **Critical**: Verify coverage < 100% (actual singularities encountered)
- [ ] Pole reconstruction demo with ground‑truth poles:
  - Verify PLE, sign consistency, slope, residual metrics.
  - **Add**: Test should fail if pole accuracy < 60%
- [ ] **Add**: Robotics IK test with actual singular configurations

### 3.3 Continuous Integration
- [ ] Add CI jobs for:
  - Basic TR property tests.
  - Regression tasks with pole metrics.
  - No‑NaN invariant on IEEE export streams.
  - At least one "good spike" reproduction run.
  - **Add**: Coverage control effectiveness test (must achieve target ± 5%)
  - **Add**: Pole detection accuracy test (must exceed baseline)

---

## 4. Documentation

- [ ] Update `docs_en.md` and `complete_v2.md`:
  - Add hybrid gradient schedule details.
  - Document tag‑loss, pole‑head, and pole‑specific metrics.
  - Clarify coverage extensions (near‑pole handling).
  - **Add**: Document why 100% coverage indicates a problem
  - **Add**: Provide guidelines for dataset generation with actual singularities
- [ ] Provide reviewer‑oriented notes:
  - Explicit rebuttal to "Dropped Sample" criticism (Mask‑REAL + tag‑loss).
  - Explicit rebuttal to "Extrapolation Illusion" criticism (pole metrics, asymptotic enforcement).
  - **Add**: Include example results showing < 100% coverage as evidence
  - **Add**: Show pole detection accuracy improvements with proper training
- [ ] **Add troubleshooting guide**:
  - What to do when coverage is always 100%
  - How to verify singularities are being encountered
  - Debugging pole detection accuracy issues

---

## 5. Priority Order (Based on Testing Results)

### Immediate (Blocking Issues):
1. **Fix dataset generation** (0.1) - Without actual singularities, nothing else matters
2. **Fix coverage control** (0.2) - λ_rej must maintain pressure
3. **Implement near-pole oversampling** (1.3) - Critical for learning

### High Priority (Core Functionality):
4. **Implement evaluation metrics** (1.5) - Needed to measure success
5. **Improve pole detection accuracy** (1.4) - Currently too low at 40%
6. **Fix hybrid gradient schedule** (1.1) - Not engaging near poles

### Medium Priority (Enhancements):
7. **Add tag-loss utilization** (1.2) - Currently unused due to 100% coverage
8. **Teacher supervision for poles** (2.2) - Would improve accuracy
9. **Visualization tools** (1.5) - Help debug and demonstrate

### Low Priority (Nice to Have):
10. **Documentation updates** (4) - After implementation is working
11. **CI improvements** (3.3) - After core issues resolved
12. **Curriculum learning** (2.1) - Optimization after basics work

---

## Success Criteria

The implementation will be considered successful when:
- [ ] Coverage achieves target ± 5% (not stuck at 100%)
- [ ] At least 10% of outputs are non-REAL during training
- [ ] Pole detection accuracy exceeds 60%
- [ ] PLE (Pole Localization Error) < 0.1 for synthetic data
- [ ] No NaN errors during training
- [ ] Robotics IK demo shows better performance near singularities than baselines
