# ZeroProofML – Implementation To‑Do (for Cursor agents)

> Goal: implement the two enhancement packages added to *Complete V2*: (A) Anti “Dropped Sample”, (B) Anti “Extrapolation Illusion”. This to‑do is grounded in `docs_en.md` modules and uses existing APIs where possible.

---

## 0) Repo setup & conventions
- **Branching:** create feature branches `feat/pole-learning` and `feat/hybrid-grad`. Merge via PRs with CI tests.
- **Coding style:** follow existing module organization in `zeroproof/` and tests in `tests/`.
- **Acceptance criteria (global):**
  1) All unit tests pass; add ≥ 12 new tests covering tags, gradients, and metrics.
  2) Examples run end‑to‑end without NaNs/uncaught exceptions.
  3) New metrics appear in logs and are saved to `runs/<exp>/metrics.json`.

---

## 1) Hybrid Gradient Schedule (Mask‑REAL → Saturating near poles)
**Why:** Allow non‑REAL neighborhoods to contribute finite gradients late in training without early instabilities.

**Files to modify:**
- `zeroproof/autodiff/grad_mode.py`
- `zeroproof/autodiff/saturating_ops.py`
- `zeroproof/layers/saturating_rational.py`
- `zeroproof/training/trainer.py`

**Tasks:**
1. **Expose schedule config** in `GradientModeConfig`:
   - Add fields: `schedule={"warmup_epochs": N1, "delta_init": 1e-2, "delta_final": 1e-6}`.
   - Add getter/setter and `with gradient_mode(...)` context to accept `local_threshold=delta`.
2. **Local gating** inside division gradient:
   - In `saturating_ops.py`, implement `is_near_pole = (abs(Q_value) <= delta)` and apply saturating gradient only when `is_near_pole` *and* global mode is `SATURATING`.
   - Outside this region, keep `MASK_REAL` behavior.
3. **Trainer integration:**
   - In `trainer.py`, add epoch hook: after epoch ≥ N1, set mode to SATURATING and linearly/log‑scale `delta` from `delta_init → delta_final` over remaining epochs.
4. **Unit tests:**
   - New test in `tests/test_grad_schedule.py` to verify: (i) early epochs produce zero grad on non‑REAL, (ii) late epochs produce finite, bounded grads only for `|Q| ≤ delta`.

**Acceptance:** gradient norms stay bounded; training logs show `mode=MASK_REAL → SATURATING`, and `delta` decays.

---

## 2) Tag‑Loss for non‑REAL outputs (PINF/NINF/PHI)
**Why:** Non‑REAL samples contribute supervision rather than being silently ignored.

**Files:**
- `zeroproof/training/adaptive_loss.py`
- `zeroproof/layers/tr_rational.py`
- `zeroproof/training/trainer.py`

**Tasks:**
1. **Add tag‑loss API** in `adaptive_loss.py`:
   - Implement `compute_tag_loss(tags_pred, tags_true, weights)` returning CE loss over {PINF,NINF,PHI}.
2. **Model outputs:**
   - In `TRRationalMulti`, optionally return an auxiliary logits head `tag_logits` when `enable_tag_head=True`.
   - Map predicted logits → tags with softmax for reporting.
3. **Trainer:**
   - When any output tag is non‑REAL, compute tag‑loss with small weight (e.g., `lambda_tag=0.05`). Combine with main loss and `λ_rej` policy.
4. **Tests:**
   - `tests/test_tag_loss.py`: ensure gradients flow into the tag head; verify CE decreases on synthetic labeled infinities and Φ.

**Acceptance:** training reports `tag_loss`; confusion matrix over PINF/NINF/PHI improves over epochs.

---

## 3) Coverage Control (oversampling + adaptive λ₍rej₎)
**Why:** Prevent trivial strategy of rejecting too many near‑pole points.

**Files:**
- `zeroproof/training/coverage.py`
- `zeroproof/training/adaptive_loss.py`
- `zeroproof/training/trainer.py`

**Tasks:**
1. **Coverage target `c*`** in `CoverageTracker`:
   - Add `target_coverage` and `coverage_gap()`; expose to trainer.
2. **Adaptive λ₍rej₎**:
   - Update policy: if REAL coverage < `c*`, decrease λ₍rej₎; else increase slightly. Provide min/max clamps.
3. **Sampler hooks:**
   - In data loader utilities, add `NearPoleSampler` that up‑weights samples with `|Q| ≤ δ_samp` (or domain‑specific proxy).
4. **Tests:**
   - `tests/test_coverage.py`: simulate batches with varying tag rates and verify λ₍rej₎ adapts to move coverage toward `c*`.

**Acceptance:** logs show coverage moving toward target, and λ₍rej₎ adjustments are bounded and stable.

---

## 4) Pole‑Head (auxiliary predictor for Q≈0)
**Why:** Explicitly learn where poles are instead of only inferring from outcomes.

**Files:**
- `zeroproof/layers/tr_rational.py`
- `zeroproof/layers/basis.py`
- `zeroproof/training/trainer.py`

**Tasks:**
1. **Add optional pole‑head** in `TRRationalMulti`:
   - Small MLP/Chebyshev head `pole_score(x)∈R`; produce `σ(pole_score)` as `p_pole`.
2. **Teacher/proxy wiring:**
   - Trainer accepts `pole_labels` (binary) per sample. If provided, compute `BCE(p_pole, pole_labels)` with weight `lambda_pole`.
3. **Shared‑Q option:**
   - Keep the existing `shared_Q=True` to enforce coherent poles across outputs.
4. **Tests:**
   - `tests/test_pole_head.py`: with analytic labels (e.g., synthetic function), verify AUC>0.9 after few epochs.

**Acceptance:** AUC/PR for pole detection reported; improves with training.

---

## 5) Anti‑Illusion Metrics & Losses
**Why:** Prove we learn pole geometry & asymptotics, not just labels.

**Files:**
- `zeroproof/training/trainer.py`
- `zeroproof/utils/metrics.py` (new)
- `zeroproof/layers/tr_rational.py`

**Tasks:**
1. **Metrics module:** implement
   - **PLE** (Pole Localization Error): distance between `{Q(x)=0}` estimates and ground‑truth set. Provide 1D/2D helpers (Chamfer/Hausdorff approximations on sampled grids).
   - **Sign‑consistency@cross:** for parametrized paths crossing poles, check correct +∞↔−∞ flipping.
   - **Asymptotic slope error:** fit slope of `log|y|` vs `−log|Q|` in near‑pole window; penalize |slope−1|.
2. **Residual consistency loss:**
   - Implement optional `lambda_resid * mean((Q·y−P)^2)` on REAL near‑pole samples.
3. **Trainer logging:**
   - Save all metrics per epoch; add plots in `runs/<exp>/plots/`.
4. **Tests:**
   - Synthetic functions with known poles; verify metrics improve when model capacity/budget increases.

**Acceptance:** metrics appear and trend in the expected direction during training.

---

## 6) Data utilities for robotics (IK near singular Jacobians)
**Why:** Provide ready‑to‑use dataset generators.

**Files:**
- `examples/robotics/rr_ik_dataset.py` (new)
- `examples/robotics/rr_ik_train.py` (new)

**Tasks:**
1. **Dataset generator:** RR‑arm kinematics; sample `θ1, θ2`, compute `J(θ)`, `det J`, mark near‑singular if `|det J|≤τ`. Provide `Δx→Δθ*` targets from a DLS teacher for off‑pole regimes.
2. **Training script:**
   - CLIs: choose model (`mlp`, `rat_eps`, `tr_rat`), toggle tag‑loss/pole‑head/residual loss, set coverage target and gradient schedule.
3. **Evaluation:**
   - PLE, sign‑consistency, asymptotic slope, tracking error on a path crossing the singular region.

**Acceptance:** script runs end‑to‑end and produces comparison tables/plots in `runs/`.

---

## 7) Baselines & Ablations
**Why:** Transparent comparison.

**Files:**
- `examples/baselines/` (new directory)

**Tasks:**
1. **Baselines:**
   - `mlp_baseline.py` (ReLU/Tanh MLP)
   - `rational_eps_baseline.py` (P/(Q+ε), grid ε∈{1e−6,1e−4,1e−3})
   - `dls_solver.py` (reference IK step)
2. **Ablations:**
   - Flags to disable: tag‑loss, residual loss, pole‑head, hybrid schedule, coverage control.
3. **Reporting:**
   - Save CSV with all metrics and config; include wall‑clock time and any manual ε tuning.

**Acceptance:** reproducible scripts and CSV tables for the paper.

---

## 8) Logging, plots, and reports
**Files:**
- `zeroproof/utils/logging.py` (extend)
- `zeroproof/utils/plotting.py` (new, matplotlib)

**Tasks:**
1. **Structured logs:** include tag counts (REAL/PINF/NINF/PHI), coverage, λ₍rej₎, gradient mode, delta, metric scores.
2. **Plots:** training curves, pole heatmaps, sign‑flip along paths, histogram of residuals near poles.

**Acceptance:** visual artifacts saved per run; paper‑ready figures can be exported as SVG/PNG.

---

## 9) Documentation updates
**Files:**
- `docs/` (add a section "Pole Learning & Hybrid Gradient")

**Tasks:**
1. **API docs:** parameters for schedules, tag‑loss, residual‑loss, coverage target, pole‑head.
2. **Tutorial:** end‑to‑end RR‑arm example with commands.

**Acceptance:** docs build without warnings; new section links to examples.

---

## 10) CI & quality gates
**Files:**
- `.github/workflows/ci.yml`

**Tasks:**
1. Run tests on Python 3.10/3.11; cache datasets; fail on coverage < 85% for new modules.
2. Lint/format checks.

**Acceptance:** green CI on both feature branches and after merge.

---

## Quick command plan (pseudo)
```bash
# 1) RR dataset & baselines
python examples/robotics/rr_ik_dataset.py --n 50000 --tau 1e-3
python examples/robotics/rr_ik_train.py --model tr_rat --enable-tag-head --enable-pole-head \
  --lambda-tag 0.05 --lambda-pole 0.1 --lambda-resid 0.01 \
  --grad-warmup 20 --delta-init 1e-2 --delta-final 1e-6 --coverage-target 0.7

# Baselines
python examples/baselines/mlp_baseline.py ...
python examples/baselines/rational_eps_baseline.py --eps 1e-4 ...
python examples/baselines/dls_solver.py ...
```

---

## Done definition (per package)
- **Anti Dropped‑Sample:** tag‑loss + coverage control + (optional) pole‑head implemented; tests pass; coverage approaches target in logs.
- **Anti Extrapolation‑Illusion:** PLE, sign‑consistency, asymptotic slope metrics + residual loss implemented; synthetic and RR tasks show improving scores vs baselines.

