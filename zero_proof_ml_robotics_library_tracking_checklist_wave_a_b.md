# ZeroProofML – Robotics & Library Tracking Checklist (Wave A & B)

> **Purpose.** A single, repo‑ready tracker for implementing and validating ZeroProofML’s goals on the 2‑DOF robotics example (4D→2D), with apples‑to‑apples baselines and near‑pole evidence.

---

## 0) Conventions & Shared Artifacts

- **Repo paths (suggested):**
  - Library: `zeroproof/`
  - Examples: `examples/robotics/`
  - Experiments: `experiments/robotics/`
  - Results: `results/robotics/<date>/<run_name>/`
- **Profiles:** `quick` (dev, fewer epochs, light logging), `full` (paper). See §7.
- **Seed utility:** `zeroproof/utils/seeding.py` → `set_global_seed(seed: int)`
  - Sets: `random`, `numpy`, `torch` (if installed), `PYTHONHASHSEED`, Deterministic cudnn.
- **Bucket edges by |det(J)| (B0 includes exact zeros):**
  ```
  B0: [0, 1e-5]
  B1: (1e-5, 1e-4]
  B2: (1e-4, 1e-3]
  B3: (1e-3, 1e-2]
  B4: (1e-2, inf)
  ```
- **Loss convention (2 outputs):** MSE averaged over outputs: \( L = (\text{MSE}(y_1, \hat y_1) + \text{MSE}(y_2, \hat y_2))/2 \).
- **Diagnostics to log each epoch:**
  - `q_min`, `near_pole_ratio` (fraction with |Q| ≤ 1e−3), `saturating_ratio` (fraction of grads using saturating mode in Hybrid), `bucket_counts` by |det(J)| on train/val batches.
- **Uniform JSON schema (results):**
  ```json
  {
    "global": {
      "seed": 123,
      "profile": "quick|full",
      "dataset": {"name": "rr2d", "n_train": 20000, "n_test": 5000,
                   "stratify_by_detj": true, "bucket_edges": [1e-5,1e-4,1e-3,1e-2],
                   "singular_ratio": {"train":0.05, "test":0.10},
                   "min_detj": 1e-6, "force_exact_singularities": true},
      "model": {"name": "tr_rational|tr_full|eps_rational|mlp|dls", "params": {...}},
      "trainer": {"name": "HybridTRTrainer", "batch_size": 1024, "epochs": 20,
                   "optimizers": {"head_main":"Adam", "head_pole":"Adam"}}
    },
    "metrics": {
      "overall": {"mse": 0.4411, "mae": 0.5},
      "buckets": [
        {"range": [0.0,1e-5], "count": 312, "mse": 1.22},
        {"range": [1e-5,1e-4], "count": 480, "mse": 0.95},
        {"range": [1e-4,1e-3], "count": 790, "mse": 0.70},
        {"range": [1e-3,1e-2], "count": 1150, "mse": 0.55},
        {"range": [1e-2,"inf"], "count": 2268, "mse": 0.40}
      ],
      "pole_metrics": {"PLE": 0.012, "sign_consistency": 0.98,
                        "slope_error": 0.07, "residual_consistency": 0.91}
    },
    "diagnostics": {
      "q_min": 2.3e-4,
      "near_pole_ratio": 0.14,
      "saturating_ratio": 0.18,
      "timings": {"train_epoch_sec": 18.2, "eval_sec": 1.9}
    }
  }
  ```

---

## 1) Wave A — MUST items (P0)

### A1) Multi‑input TR model (4D→2D) + multi‑output API
- [x] **Implement** `TRMultiInputRational` (TR‑MLP front + TR‑Rational heads) or `TRRationalMulti` with optional **shared‑Q** for both outputs.
- [x] **Forward signature (library)**: `forward(x: List[TRNode|float] (len=4)) -> List[(TRNode, TRTag)] (len=2)`.
      For baselines/wrappers, provide tensor variants: `forward(x: array[...,4]) -> (y: array[...,2], tags: empty)`.
- [x] **Gradient checks** on random inputs (finite‑difference in TR; or torch.autograd gradcheck in wrappers).
      Added dedicated TR finite‑difference gradcheck for 4D→2D model: `tests/unit/test_robotics_gradcheck.py`.
- **DoD:** One script `examples/robotics/rr_ik_train.py` can switch models via `--model` and trains 4D→2D consistently.

### A2) Hybrid trainer: true mini‑batching + persistent per‑head optimizers
- [x] Keep **one optimizer per head** (instantiate once in `__init__`).
- [x] Batch loop updates **all heads** without per‑step re‑construction.
- [x] Toggle metrics cadence: `--log-every N` steps.
      Implemented via `log_interval` in trainers; exposed as `--log_every` in `examples/robotics/rr_ik_train.py`.
      Per‑epoch bench table implemented and saved: `avg_step_ms`, `data_time_ms`, `optim_time_ms`, `batches`.
      Printed from `HybridTRTrainer._log_epoch` and persisted to `bench_history` in trainer summary; also captured in robotics driver (`IKTrainer`).
- **Target:** Step time with metrics off ≤ **1.3×** TR‑basic at same batch size.
- **DoD:** Bench table printed at end of epoch: `avg_step_ms`, `data_time_ms`, `optim_time_ms`.

### A3) Stratified split & generator knobs (by |det(J)|)
- [x] Add flags: `--stratify-by-detj`, `--force-exact-singularities`, `--min-detj`, `--singular_ratio <train>:<test>`.
- [x] Print and save histograms of |det(J)| per split; ensure **non‑zero counts in B0–B3**.
- **DoD:** JSON includes per‑bucket **counts** for train/val/test; console shows a compact bucket table.

Notes:
- Implemented CLI in `examples/robotics/rr_ik_dataset.py` with:
  - `--stratify_by_detj`, `--force_exact_singularities`, `--min_detj`
  - `--singular_ratio_split <train>:<test>` (name differs slightly from stub, functionally identical)
  - `--bucket-edges ...` (optional override; defaults from `zeroproof/utils/config.py::DEFAULT_BUCKET_EDGES`)
  - `--ensure_buckets_nonzero` augments near‑pole samples to populate B0–B3 in both splits.
- JSON metadata fields when stratified: `bucket_edges`, `stratified_by_detj`, `train_ratio`, `train_bucket_counts`, `test_bucket_counts`, optional `singular_ratio_split`, `ensured_buckets_nonzero`, and `seed`.
- Tests: `tests/unit/test_generator.py` asserts non‑empty B0–B3; `tests/unit/test_serialization.py` covers JSON/NPZ round‑trip.

### A4) Comparator parity & bucketed metrics for all methods
- [x] Single driver `experiments/robotics/run_all.py` running: `mlp`, `rational_eps`, `tr_basic`, `tr_full`, `dls` with identical splits & loss.
      Thin wrapper forwards to `examples/baselines/compare_all.py::run_complete_comparison`; supports `--dataset`, `--profile quick|full`, `--models`, `--seed`, `--output_dir`.
- [x] Compute **per‑bucket MSE** (B0–B4) and **overall**, and include **bucket counts**; save to JSON & print a table for every method.
- [x] Record **params**, **epochs**, **step time**, **total train time**.
- **DoD:** One line per method, same buckets, same loss, same dataset hash; console summary shows bucket counts per method.

### A5) Reproducibility (global seeding)
- [x] `zeroproof/utils/seeding.py` with `set_global_seed(seed)` called by: dataset generator, trainers, comparators.
- [x] Record `seed` in every results JSON.
- **DoD:** Re‑running the same command yields identical metrics within floating‑point tolerance.

### A6) Performance tuning of "Full" mode
- [x] Profiles: `--profile quick|full` controlling epochs, logging cadence, extra metrics (`--log_every`).
- [x] Optional toggles to reduce overhead: `--no_structured_logging`, `--no_plots`.
- **Target:** `tr_full(quick)` wall‑clock ≤ **2×** `tr_basic(quick)` (profiling shows near‑linear scale at same batch size).
- **DoD:** Per‑epoch bench printed and saved (`avg_step_ms`, `data_ms`, `optim_ms`, `batches`); quick runs finish significantly faster at the same batch size.

---

## 2) Wave B — SHOULD items (P1)

### B1) 2D near‑pole metrics (PLE/sign/slope/residual)
- [x] Implement helpers in `zeroproof/metrics/pole_2d.py`:
  - **PLE (Pole Localization Error)** against analytic θ₂∈{0,π} lines (from det(J)).
  - **Sign consistency** across a θ₂‑crossing path.
  - **Slope error** near the pole lines.
  - **Residual consistency** via forward kinematics with predicted Δθ.
- [x] Integrate into comparator outputs.
- **DoD:** JSON `pole_metrics` populated for TR/ε/MLP (DLS where applicable).

### B2) Teacher signals for pole head
- [x] Supervise pole head using analytic det(J) signals (distance/sign to θ₂ lines).
- [x] Track PLE vs ground truth each epoch.
- **DoD:** `pole_head_loss` curve and final PLE in JSON.

### B3) DLS enhancements & vectorized perf
- [x] Save per‑sample errors and |det(J)|; log failure reasons (divergence, iterations cap).
- [x] Vectorize dataset gen & DLS evaluation.
- **DoD:** DLS JSON includes `per_sample: {idx, detj, err, status}`; run time reduced vs baseline.

### B4) Serialization safety & interface hygiene
- [x] Centralize NumPy↔Python converters; add unit tests for JSON/NPZ round‑trip.
- [x] Normalize `forward`/`forward_fully_integrated`; ensure vector input returns tags consistently.
- **DoD:** `pytest -k serialization` and `pytest -k interface` green.

### B5) Docs refresh (venv/PEP‑668, baselines, buckets)
- [x] Update `docs/08_howto_checklists.md` with venv/PEP‑668 guidance, baseline parity, bucket interpretation.
      Note: The doc is updated and includes PEP‑668 venv steps and bucket guidance; it references `experiments/robotics/run_all.py`
      which functionally maps to `examples/baselines/compare_all.py` in this repo.
- **DoD:** Docs lints pass; example commands reproduce a quick run end‑to‑end.

---

## 3) CLI Stubs (copy‑paste)

```bash
# Dataset with stratified buckets and forced singularities
python examples/robotics/make_dataset.py \
  --n-train 20000 --n-test 5000 \
  --stratify-by-detj --bucket-edges 1e-5 1e-4 1e-3 1e-2 \
  --force-exact-singularities \
  --min-detj 1e-6 \
  --singular_ratio 0.05:0.10 \
  --seed 123 \
  --out results/robotics/2025-09-12/datasets/rr2d_seed123.npz

# TR-basic vs TR-full vs eps-rational vs MLP vs DLS (quick)
python experiments/robotics/run_all.py \
  --dataset results/robotics/2025-09-12/datasets/rr2d_seed123.npz \
  --profile quick --epochs 20 --batch-size 1024 \
  --models tr_basic tr_full rational_eps mlp dls \
  --seed 123 \
  --out results/robotics/2025-09-12/quick_seed123/

# Single TR-full run with pole head supervised, extra logs (full)
python examples/robotics/rr_ik_train.py \
  --model tr_full --profile full --epochs 100 --batch-size 2048 \
  --supervise-pole-head --log-every 50 \
  --dataset results/robotics/2025-09-12/datasets/rr2d_seed123.npz \
  --seed 987 \
  --out results/robotics/2025-09-12/tr_full_seed987/
```

---

## 4) Compact Console Table (per method)

```
Method         Params  Epochs  Time(s)  Overall MSE  B0 MSE  B1 MSE  B2 MSE  B3 MSE  B4 MSE  B0 Cnt  B1 Cnt  B2 Cnt  B3 Cnt  B4 Cnt
tr_basic       12      20      180      0.4411       1.22    0.95    0.70    0.55    0.40    312     480     790     1150    2268
tr_full        342     20      360      0.4410       1.18    0.92    0.68    0.53    0.39    312     480     790     1150    2268
rational_eps   12      20      160      0.4421       1.30    0.98    0.73    0.57    0.41    312     480     790     1150    2268
mlp            722     20      820      0.9391       2.80    2.10    1.60    1.20    0.85    312     480     790     1150    2268
dls            0       -       12       0.0412*      0.035   0.037   0.039   0.040   0.043   300     470     780     1100    2200
```
(*on subset where DLS converges; also report success rate.)

---

## 5) Unit Tests (PyTest names)
 - [x] `tests/test_seeding.py::test_seeding_reproducible`
 - [x] `tests/test_generator.py::test_stratified_buckets_nonempty`
 - [x] `tests/test_generator.py::test_force_exact_singularities`
 - [x] `tests/test_models.py::test_tr_multi_forward_shapes`
 - [x] `tests/test_trainer.py::test_persistent_optimizers`
 - [x] `tests/test_metrics.py::test_bucketed_mse_schema`
 - [x] `tests/test_metrics.py::test_pole_metrics_2d`
 - [x] `tests/test_serialization.py::test_json_npz_roundtrip`
 - [x] `tests/test_comparator.py::test_parity_same_split_and_loss`

---

## 6) Risks & Guardrails

- **Multivariate instability:** Use **shared‑Q** and constrain Q via `Q(0)=1` + L2 on denominator coeffs.
- **Hybrid not switching:** If `near_pole_ratio<0.05` after stratification, increase `--singular_ratio` or lower `--min-detj`.
- **Comparator drift:** Centralize bucket edges and loss in a small config (§8). Print bucket counts for **every** method.
- **Repro flakiness:** One seeding function; record seeds everywhere.

---

## 7) Profile Defaults

```yaml
profiles:
  quick:
    epochs: 20
    batch_size: 1024
    log_every: 200
    metrics:
      pole: false
      residual_consistency: false
  full:
    epochs: 100
    batch_size: 2048
    log_every: 50
    metrics:
      pole: true
      residual_consistency: true
```

---

## 8) Typed Config for Experiments (YAML)

```yaml
experiment:
  name: rr2d_ablation_v1
  seed: 123
  dataset:
    path: results/robotics/2025-09-12/datasets/rr2d_seed123.npz
    stratify_by_detj: true
    bucket_edges: [1e-5, 1e-4, 1e-3, 1e-2]
    singular_ratio: {train: 0.05, test: 0.10}
    min_detj: 1e-6
    force_exact_singularities: true
  models: [tr_basic, tr_full, rational_eps, mlp, dls]
  trainer:
    profile: quick
    epochs: 20
    batch_size: 1024
    optimizers:
      main: {kind: Adam, lr: 1e-3}
      pole: {kind: Adam, lr: 1e-3}
  outputs:
    save_json: true
    console_table: true
```

---

## 9) Acceptance Checklist (tick before claiming success)

- [ ] **Coverage:** B0–B3 have **non‑zero** counts in **train/val/test** (B0 includes exact zeros).
- [ ] **Parity:** All methods share identical splits, **validation‑based** hyperparameter selection (e.g., ε grid search on **val only**), loss, and bucket edges; table printed with counts.
- [ ] **Repro:** Re‑run with same seed reproduces metrics within tolerance. Torch set with `torch.use_deterministic_algorithms(True)` and CUDNN deterministic; dataloader workers seeded.
- [ ] **Performance:** `tr_full(quick)` ≤ **2×** `tr_basic(quick)` wall‑clock under identical device/dtype.
- [ ] **Evidence near poles:** Per‑bucket MSEs reported; `near_pole_ratio` and `saturating_ratio` > 0.
- [ ] **Artifacts:** JSON files contain `global`, `metrics`, `diagnostics` exactly as schema; JSON uses string `"inf"` for infinity and encodes TR special values (`+inf`, `-inf`, `phi`).
- [ ] **Docs:** HOW‑TO updated; commands in §3 run end‑to‑end on a fresh venv.
- [ ] **Versioning:** Results include git commit hash, library versions (Python, NumPy, Torch), device info.

---

## 10) Audit Fixes & Clarifications (Added)

**A. Serialization & JSON**
- Use strings for non‑finite values: `"inf"`, `"-inf"`, and `"phi"` (TR nullity). Avoid bare `NaN/Infinity` which are non‑standard JSON.
- Add `"schema_version": "1.0"` at `global` level and provide a small upgrader if schema changes.

**B. Determinism & Environment**
- Enforce: `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
- Seed dataloader workers via `worker_init_fn` using the global seed.
- Record: git commit, hostname, device, dtype, BLAS/MKL info.

**C. Baseline Fairness**
- ε‑baseline: select ε via **validation grid search** (logspace) and **never** on test. Record the grid and the chosen ε.
- DLS: report success rate; provide **two** MSEs: (i) on converged subset; (ii) on **all** test points (counting failures with a large sentinel error or exclude with explicit note).

**D. Bucketing & Metrics**
- B0 now **includes** exact zeros. Also log **log10‑bucket histograms** for sanity.
- Define angle metrics modulo 2π; when measuring PLE vs θ₂ lines, use wrapped distances.
- Add **relative error** on outputs and **angular MAE/MedAE (degrees)** for interpretability.

**E. Loss & Weights (Multi‑head)**
- Total loss: `L = L_main + w_pole * L_pole + w_rej * (#nonREAL)` with default `w_pole=0.1`, `w_rej=0.0–0.05`.
- Clip each head’s grad separately if needed; record per‑head LR/schedule.

**F. Robust Jacobian & det(J)**
- Compute det(J) with numerically stable formulas; clamp tiny negatives to 0 when within tolerance.
- Log both **|det(J)|** and model **|Q|** (for TR), so `near_pole_ratio` can be defined on either.

**G. Unit/Property Tests (Additions)**
- Property: **Totality** — random inputs never raise and never emit NaN in TR paths.
- Property: **Hybrid switch** triggers when `|Q| < τ` on synthetic cases.
- Test: **No leakage** — ε grid search reads **val** only.

**H. Performance Guidance**
- Prefer shared‑Q and low total degree; profile with `torch.compile` only if TR ops are compatible; otherwise ensure JIT off.
- Use pinned memory and prefetch; vectorize DLS; ensure identical dtype/device across methods in comparator.

**I. CLI & Config Hygiene**
- Replace `+∞` in configs with `"inf"`; accept `--bucket-edges` or `--auto-log10 N` to auto‑create N bins.
- Write a small **dataset hash** (e.g., SHA256 of NPZ) into results for parity checks.

**J. Docs Notes**
- Clarify venv creation under PEP‑668 (e.g., `python -m venv .venv && . .venv/bin/activate` or `uv venv`), and warn about system Python interference.
