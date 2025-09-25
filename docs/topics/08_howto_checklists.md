# How‑To Checklists

Concise, repeatable steps for common ZeroProof workflows.

## Train a TR‑Rational with Poles
- Create model: `HybridTRRational(d_p, d_q, basis=ChebyshevBasis())`.
- Create schedule: `create_default_schedule(warmup_epochs=5)`.
- Loop epochs with `with schedule.apply(epoch):` and forward on batches.
- Track tags from outputs; update `AdaptiveLambda` policy.
- Log `HybridGradientContext.get_statistics()` per batch.
- Save checkpoints after coverage stabilizes.

## Enable Hybrid Gradients in Existing Code
- Import `create_default_schedule` and create schedule.
- Set global mode if needed: `GradientModeConfig.set_mode(GradientMode.HYBRID)`.
- Wrap training step with `with schedule.apply(epoch): ...`.
- Tune `delta_init/delta_final` and `saturating_bound` from stats.

## Use Adaptive Coverage Control
- Instantiate `AdaptiveLambda(AdaptiveLossConfig(target_coverage=0.95, learning_rate=0.01, lambda_rej_min=0.1))`.
- After each batch, call `policy.update(tags)`.
- Read `policy.get_statistics()`; expect `coverage_gap→0` and λ above `lambda_rej_min`.

## Evaluate Pole Metrics
- Create `IntegratedEvaluator(EvaluationConfig(...), true_poles=[...])`.
- Call `evaluate_model(model, x_values)`; inspect PLE, sign consistency, residual.
- Enable plots with `enable_visualization=True`, set `plot_frequency`.

## Normalize Without ε (TR‑Norm)
- Add `TRNorm(num_features)` or `TRLayerNorm(normalized_shape)`.
- No eps parameter; zero‑variance features bypass to β deterministically.

## Debug Tag Distribution
- During training, keep counts of REAL/PINF/NINF/PHI.
- Investigate persistent PHI: check 0/0 patterns; inspect basis and parameterization.
- Investigate coverage dips: increase λ or adjust schedule δ.

## References
- Autodiff Modes: `docs/topics/03_autodiff_modes.md`.
- Layers & Variants: `docs/topics/04_layers.md`.
- Training Policies: `docs/topics/05_training_policies.md`.
- Sampling & Curriculum: `docs/topics/06_sampling_curriculum.md`.
- Evaluation & Metrics: `docs/topics/07_evaluation_metrics.md`.

## Run Robotics IK (RR arm) — TR vs Baselines

Follow these steps to reproduce the RR inverse kinematics example near singularities (θ2≈0 or π) and compare with baselines.

1) Environment (PEP‑668 friendly)
- Python 3.9+ with NumPy installed. Matplotlib optional (for plots in demos).
- Create and activate an isolated venv (PEP‑668 safe):
  - POSIX: `python -m venv .venv && . .venv/bin/activate`
  - Windows: `py -m venv .venv && .venv\\Scripts\\activate`
  - Or with uv: `uv venv && . .venv/bin/activate`
- Install package in editable mode: `pip install -e .`
- Optional: install dev deps you need for tests: `pip install pytest`

2) Quick sanity check (prints kinematics and a tiny training comparison)
```bash
python examples/robotics/demo_rr_ik.py
```
- Shows det(J) and singularity checks; runs a small training comparison.

3) Generate a dataset (JSON) with near‑pole coverage and buckets
```bash
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
```
- Output: `data/rr_ik_dataset.json` (directory created if missing)
- Prints summary stats and per‑bucket counts for |det(J)|.
- Buckets (by |det(J)|): B0 [0,1e−5], B1 (1e−5,1e−4], B2 (1e−4,1e−3], B3 (1e−3,1e−2], B4 (1e−2, inf)
- JSON contains `metadata.bucket_edges`, `train_bucket_counts`, `test_bucket_counts`, and when stratified also
  `stratified_by_detj`, `train_ratio`, and `ensured_buckets_nonzero`. If you pass split‑specific singular ratios
  (`--singular_ratio_split a:b`) they are recorded as well. A `seed` field is saved for reproducibility.

4) Train TR‑Rational model (ZeroProof)
```bash
python examples/robotics/rr_ik_train.py \
  --dataset data/rr_ik_dataset.json \
  --model tr_rat \
  --epochs 80 \
  --learning_rate 1e-2 \
  --degree_p 3 --degree_q 2 \
  --output_dir runs/ik_experiment
```
- Default enables: hybrid schedule, tag loss, pole head, residual consistency, coverage enforcement.
- Output JSON: `runs/ik_experiment/results_tr_rat.json`
- Console prints final test MSE and training summary.

Key CLI flags (policy/hybrid/contracts)

| Flag | Description |
|------|-------------|
| `--no_tr_policy` | Disable TR policy (use raw TR tags) |
| `--policy_ulp_scale` | ULP scale for τ guard bands (used by default/auto policy) |
| `--no_policy_det_reduction` | Disable deterministic pairwise reductions |
| `--hybrid_aggressive` | More aggressive hybrid schedule (smaller bound) |
| `--hybrid_delta_init/final` | Initial/final `delta` for schedule |
| `--use_contract_safe_lr` | Enable contract‑safe LR clamp |
| `--contract_c` | Contract constant `c` for LR clamp |
| `--loss_smoothness_beta` | Loss smoothness `β` assumption for LR clamp |

Training metrics captured per‑epoch include: `flip_rate`, `saturating_ratio`, `tau_q_on/off`, `q_min_epoch`, `curvature_bound`, and `mask_bandwidth`. The `layer_contract` is included in the training summary and top‑level results.

Plot training curves:

```bash
python scripts/plot_training_curves.py \
  --results runs/ik_experiment/results_tr_rat.json \
  --outdir runs/ik_experiment
```

Optional — supervise pole head with analytic teacher and log PLE each epoch:
```bash
python examples/robotics/rr_ik_train.py \
  --dataset data/rr_ik_dataset.json \
  --model tr_rat \
  --epochs 80 \
  --learning_rate 1e-2 \
  --degree_p 3 --degree_q 2 \
  --supervise-pole-head --teacher_pole_threshold 0.1 \
  --output_dir runs/ik_experiment
```
- Adds `training_summary.pole_head_loss_history`, `training_summary.ple_history`, and `training_summary.final_ple`.

5) Train baselines on the same dataset
- MLP:
```bash
python examples/robotics/rr_ik_train.py \
  --dataset data/rr_ik_dataset.json \
  --model mlp \
  --epochs 80 \
  --hidden_dim 64 \
  --learning_rate 1e-2 \
  --output_dir runs/ik_experiment
```
- Epsilon‑regularized rational (rat_eps):
```bash
python examples/robotics/rr_ik_train.py \
  --dataset data/rr_ik_dataset.json \
  --model rat_eps \
  --epochs 80 \
  --degree_p 3 --degree_q 2 \
  --learning_rate 1e-2 \
  --output_dir runs/ik_experiment
```
- Outputs: `results_mlp.json`, `results_rat_eps.json` in the same output dir.

6) Compare results (parity across baselines)
- All methods run on identical splits, same loss, and same buckets. Bucketed MSE is included in JSON under `near_pole_bucket_mse` (edges, bucket_mse, bucket_counts).
- Pole metrics (2D) are included under `pole_metrics` (ple, sign_consistency, slope_error, residual_consistency).
- DLS outputs per‑sample records with error and status.
For a complete apples‑to‑apples quick run:
```bash
python experiments/robotics/run_all.py \
  --dataset data/rr_ik_dataset.json \
  --profile quick \
  --models tr_basic tr_full rational_eps mlp dls \
  --max_train 2000 --max_test 500 \
  --output_dir results/robotics/quick_run
```
- Saves `comprehensive_comparison.json` with bucketed metrics and a compact console table.
 - In quick mode, the comparator creates a stratified test subset by |det(J)|≈|sin(theta2)| ensuring B0–B3 have
   non‑zero counts when available, recomputes bucket MSE on the subset, and aligns DLS to the same subset.

7) Quick profile vs full profile
- Quick: fewer epochs (default MLP≈2, Rat≈5, ZeroProof≈5), stratified subsampling (`--max_train/--max_test`), and DLS uses a
  single‑step vectorized path on the same test subset. Great for iteration.
- Full: more epochs and full DLS iterations; use `--profile full` or per‑model epoch overrides.

Bench metrics (per‑epoch)
- Hybrid trainer prints and records per‑epoch timings: `avg_step_ms`, `data_time_ms`, `optim_time_ms`, and `batches`.
- These are returned in training summaries under `bench_history` (and surfaced in higher‑level training outputs).

Tips
- Increase `--n_samples` for more robust metrics; reduce for quicker runs.
- Use `--no_hybrid`, `--no_tag_loss`, etc., to ablate TR features in tr_rat runs.
- Set seeds for reproducibility; dataset/experiments accept `--seed`. Internally, `zeroproof/utils/seeding.py::set_global_seed` is used.
