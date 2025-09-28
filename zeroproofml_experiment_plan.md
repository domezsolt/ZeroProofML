
# ZeroProofML — Robotics Experiment Plan (6–7 DOF, CPU-friendly)

**Goal:** Validate Transreal (TR) rational layers with Mask‑REAL, guard bands, and hybrid switching on **6–7 DOF** manipulators, including **multiple simultaneous singularities**, against **strong ε‑ and smooth baselines** on a **7th‑gen i5** (CPU only).

---

## 1) Hypotheses

- **H1 — Stability & accuracy near poles.** TR achieves **bounded**, NaN‑free updates and **lower pose error** than ε/smooth baselines in the closest‑to‑singularity bins.
- **H2 — Multi‑singularity robustness.** When at least **two singular directions** are small (rank deficiency ≥ 2), TR maintains stable performance; ε‑methods either over‑damp or destabilize.
- **H3 — Reproducibility.** Under a fixed policy (ULP bands, signed zeros, deterministic reductions), TR shows **lower seed variance** near poles than baselines.

**Acceptance criteria:**
- H1: TR improves pose error by **≥10–20%** in tight bins; **≤5%** gap in far bins; bounded updates; no NaNs.
- H2: TR outperforms best baseline on the **multi‑singularity** subset; stable closed loop (no chatter/divergence).
- H3: Lower **seed variance** in near‑pole bins; outside guard bands tags are identical across seeds.

---

## 2) Robots & Tasks

**Robots (URDFs available):** 7‑DOF Franka Panda / KUKA iiwa7; 6‑DOF UR5e.  
**EE (end‑effector) frame:** e.g., `panda_hand`, `tool0`.

**Tasks:**
- **A) Resolved‑rate IK (supervised):** Learn `(q, Δx) → Δq`, labels from **DLS** with λ≈1e‑6 (near pseudoinverse). Rational structure exposes poles.
- **B) Short closed‑loop tracking (analysis):** Integrate `q_{t+1}=q_t+Δq_t` for 200 steps; trajectories pass near and dwell within singular neighborhoods; include **intersection** cases.

**Kinematics stack:** **Pinocchio** for kinematics/Jacobians (CPU). *(Optional:* PyBullet for pose validation.)

---

## 3) Singularity Metrics & Binning

Let `J(q)` be the 6×n geometric Jacobian; let `σ1≥…≥σ6` be singular values.

- **Primary distance:** `d1(q)=σ6(J(q))` (smallest singular value).
- **Multi‑singularity:** `d2(q)=σ5(J(q))` (**rank deficiency ≥2** when small).
- **Type tags (optional, recommended):**  
  `d1_v = σmin(J_v)` translational; `d1_w = σmin(J_ω)` rotational; tag as **trans/rot/mixed**.

**Bins for d1 (default edges):** `[1e-3, 1e-2, 1e-1] ⇒ 4 bins` → `[0,1e-3), [1e-3,1e-2), [1e-2,1e-1), [1e-1,∞)`.

**Held‑out extrapolation bin (recommended):** train without **smallest bin** (e.g., `[0,1e-3)`), test on it to probe near‑pole generalization.

**Multi‑singularity threshold:** `d2 < 5e-3` (tunable per robot).

---

## 4) Data Generation (script + configuration)

Use the provided script:

- **Download:** `generate_robotics_dataset.py` (Pinocchio‑based; CPU‑friendly).  
- **Outputs (`.npz`):** `q`, `dx`, `dq_target`, `d1`, `d2`, `bin_idx`, `is_multi`, `meta` (and optionally `J`).

**Examples:**
```bash
# Franka Panda (7 DOF)
python generate_robotics_dataset.py \
  --urdf /path/to/franka_panda.urdf \
  --ee-frame panda_hand \
  --out-prefix panda_sing \
  --n-train 30000 --n-val 3000 --n-test 3000 \
  --bins 1e-3,1e-2,1e-1 --d2-threshold 5e-3

# UR5e (6 DOF)
python generate_robotics_dataset.py \
  --urdf /path/to/ur5e.urdf \
  --ee-frame tool0 \
  --out-prefix ur5e_sing
```

**Dataset sizes (per robot, CPU‑feasible):**
- A‑task: **30k / 3k / 3k** train/val/test (default).  
- B‑task: **100** episodes × **200** steps (per robot).

**File contents per split:**
- `q: (N,nq)`, `dx: (N,6)`, `dq_target: (N,nq)`, `d1: (N,)`, `d2: (N,)`, `bin_idx: (N,)`, `is_multi: (N,)`, `meta: JSON bytes`.

---

## 5) Models & TR Policy

**TR model:** small MLP with **TR‑rational** blocks \(P/Q\), **Mask‑REAL** gradients, **hybrid switching** (MR↔SAT), and **guard bands**.  
**Policy:** ULP‑scaled `τ_Q, τ_P` with hysteresis (`on/off`), **signed zeros** for Q, **deterministic reductions**, bit‑packed **REAL mask** (1 bit), and compact tag payload (2–3 bits + sign + signed‑zero latch).

**Baselines (strong, CPU‑safe):**
- **Classical IK:** DLS (λ ∈ {1e‑1…1e‑5}); Truncated‑SVD pseudoinverse (τ ∈ {1e‑1,1e‑2,1e‑3}).
- **ε‑regularization:** Fixed‑ε (ε ∈ {1e‑1…1e‑5}); **learnable ε** (global/per‑layer); **ε‑ensemble** (M∈{3,4}).
- **Smooth substitutes:** tanh/softplus/exponential regularizations (α ∈ {1e‑1,1e‑2,1e‑3}).
- **Spectral/conditioning:** spectral‑norm control + mild Jacobian penalty (Hutchinson probes k∈{2,4}; λ_J∈{1e‑4,1e‑3}).
- **Optimizer safety:** gradient clipping (global/per‑layer/AGC/percentile) + **batch‑safe LR** for Adam/HB.
- **Structure‑mismatch controls:** plain MLP; polynomial network (no divisions).
- **TR ablations:** TR w/o hybrid; w/o guard bands; per‑node vs global hybrid; w/wo signed‑zero & deterministic reductions.

**Capacity parity:** keep parameter counts within ±5% across learned methods.

---

## 6) Training Protocol (CPU)

- **Optimizer:** Adam with **batch‑safe LR cap:**  
  `η_safe = min( α / L̂_batch , c / (L_ell * ∏_k max{B_k,G_max}) )`, quantile‑robust `L̂_batch` recommended.  
  Momentum bounds (sufficient):  
  `HB: η ≤ 2(1−β1)/L̂_batch`, `Adam: η ≤ (1−β1)/(√(1−β2) * L̂_batch)`.
- **Schedule:** 30–50 epochs (A‑task), 10 fine‑tune epochs (B‑task). Batch size 256 (A).  
- **Clipping (if enabled):** globals {0.5,1.0,2.0}; per‑layer {0.25,0.5}; AGC {0.01,0.05}.  
- **Logging:** loss/val; **min|Q|**, **flip_rate**, **%SAT_time**, gradient norms, **switches/1k steps** (chatter flag).  
- **Seeds:** 5 per method (fixed list); policy constants logged (ULP bands, hysteresis, rounding mode).

**Runtime notes (i5):**  
- Data gen: <30 min per robot for 60k states.  
- A‑task: ~1–2 h per baseline per robot (≤300k params).  
- B‑task: <1 h for rollouts + fine‑tuning.

---

## 7) Metrics

**A‑task (supervised):**
- Pose error (pos mm, orient deg) **per bin** of `d1`; separate report on **multi‑singularity** subset (`d2<δ2`).  
- Stability: NaN/INF counts; **descent fraction**; update‑norm spikes.

**B‑task (closed loop):**
- RMS tracking error, **overshoot**, **settling time**, **control energy** (∑‖Δq‖²).  
- Failures: divergence, oscillation (chatter near poles), constraint violations.

**TR diagnostics:** per‑bin **flip_rate**, **%SAT_time**, **min|Q|**, #switches.

**Reproducibility:** per‑bin **seed variance**; outside guard bands, TAG equality across seeds.

**Overhead:** time/epoch; %SAT_time (TR); ε‑ensemble overhead (M).

---

## 8) Analysis

- **Per‑bin curves:** pose error vs `d1` (log); panels for **multi‑singularity** subset and **extrapolation bin** (held‑out).  
- **Pareto near poles:** tracking error vs energy.  
- **Ablations:** TR w/wo hybrid/guard bands/signed‑zero; ε fixed vs learnable; smooth α forms; clipping variants.  
- **Switching density:** switches per 1k steps; chatter counts; histogram of dwell times in guard band.

---

## 9) Reproducibility Policy

- Declare **policy‑determinism**: ULP bands, hysteresis, rounding mode, deterministic reductions.  
- Outside guard bands, tags must match across seeds and (optionally) across BLAS backends in a micro‑test.

---

## 10) Deliverables

- **Datasets:** `<prefix>_{train,val,test}.npz` for each robot.  
- **Scripts:** training runner; evaluation; plotting.  
- **Tables/figures:** metrics by bin; multi‑singularity results; extrapolation; runtime/overhead; ablation grids.  
- **Appendix:** hyper‑ranges; per‑robot Jacobian stats (d1/d2 histograms); implementation details of TR policy.

---

## 11) Minimal Checklist

- [ ] Generate datasets for **Panda** and **UR5e** with bins `[1e-3,1e-2,1e-1]`; mark multi‑singularity (`d2<5e-3`).  
- [ ] Create **held‑out** smallest bin for testing only.  
- [ ] Train **TR**, **ε** (fixed/learnable/ensemble), **smooth**, **spectral**, **clipping**, **MLP/polynomial**, **IK** (DLS/tSVD).  
- [ ] Log **flip_rate**, **%SAT_time**, **min|Q|**, **switching density**.  
- [ ] Report **per‑bin** errors (including multi‑singularity subset); closed‑loop metrics; seed variance; overhead.

---

## 12) Quick Commands (RR 2R + optional 3R)

Use the orchestration script to reproduce paper-style artifacts on the planar robots (CPU-only):

```bash
# Quick 2R baselines + TR across 3 seeds (paper-ready CSV/LaTeX + bars)
python scripts/run_paper_suite.py --quick --seeds 3 \
  --out-root results/robotics/paper_suite --rollout

# Add 3R TR-only evidence (multi-singularity) in the same root folder
python scripts/run_paper_suite.py --quick --seeds 3 --include-3r \
  --out-root results/robotics/paper_suite --rrr-epochs 60

# Heavier (more epochs) 2R run
python scripts/run_paper_suite.py --seeds 5 \
  --mlp-epochs 50 --rat-epochs 50 --zp-epochs 100 \
  --out-root results/robotics/paper_suite_full
```

Artifacts of interest (paths are echoed at the end):
- Seed-wise comprehensive results: `results/robotics/paper_suite/quick_s*/comprehensive_comparison.json`
- Across-seed summary CSV/LaTeX: `results/robotics/paper_suite/aggregated/seed_summary.(csv|tex)`
- Bucket bars (B0–B2): `results/robotics/paper_suite/figures/b012_bars.png`
- Optional closed-loop rollout (2R): `results/robotics/paper_suite/rollout_summary.json`
- Optional 3R TR evidence: `results/robotics/paper_suite/e3r/e3r_results.json`

---

## 13) 6R Synthetic (No URDF)

For a CPU‑friendly 6‑DOF proxy without external kinematics libraries, use the synthetic 6R generator and TR trainer:

```bash
# Generate 6R dataset (d1 binning; ensures B0–B3 presence)
python examples/robotics/ik6r_dataset.py \
  --output data/ik6r_dataset.json \
  --n_samples 24000 --singular_ratio 0.35 \
  --ensure_buckets_nonzero --seed 1

# Train TR on 6R (input: q(6)+twist(6) → dq(6))
python examples/robotics/ik6r_train.py \
  --dataset data/ik6r_dataset.json \
  --output_dir results/robotics/ik6r_s1 \
  --epochs 40 --batch_size 256 --learning_rate 1e-2 --seed 1

# Repeat for a few seeds (e.g., 3):
for s in 1 2 3; do \
  python examples/robotics/ik6r_train.py \
    --dataset data/ik6r_dataset.json \
    --output_dir results/robotics/ik6r_s${s} \
    --epochs 40 --batch_size 256 --learning_rate 1e-2 --seed ${s}; \
done

# Aggregate across seeds into CSV (+ optional LaTeX)
python scripts/aggregate_ik6r_seeds.py \
  --glob 'results/robotics/ik6r_s*/ik6r_results.json' \
  --out results/robotics/ik6r_agg/summary.csv \
  --latex results/robotics/ik6r_agg/summary.tex
```

This synthetic 6R setup reports overall MSE and d1‑bucketed MSE; it also flags rank≥2 cases via `d2` in the dataset. It serves as a scalable proxy for H1/H3 and a partial proxy for H2 without requiring URDF/Pinocchio.
