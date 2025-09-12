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

1) Environment
- Python 3.9+ with NumPy installed. Matplotlib optional (for plots in demos).
- From the repo root, optional dev install: `pip install -e .`

2) Quick sanity check (prints kinematics and a tiny training comparison)
```bash
python examples/robotics/demo_rr_ik.py
```
- Shows det(J) and singularity checks; runs a small training comparison.

3) Generate a dataset (JSON)
```bash
python examples/robotics/rr_ik_dataset.py \
  --n_samples 2000 \
  --singular_ratio 0.35 \
  --displacement_scale 0.1 \
  --singularity_threshold 1e-3 \
  --output data/rr_ik_dataset.json
```
- Output: `data/rr_ik_dataset.json` (directory created if missing)
- Prints summary stats (counts, |det(J)|, condition number max)

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

6) Compare results
- Inspect final test MSE from console or open the JSON result files.
- For deeper comparisons (including DLS reference and plots), consider:
```bash
python examples/baselines/compare_all.py
```

Tips
- Increase `--n_samples` for more robust metrics; reduce for quicker runs.
- Use `--no_hybrid`, `--no_tag_loss`, etc., to ablate TR features in tr_rat runs.
- Results are non‑deterministic unless you set global seeds; see spec repro checklist in `complete_v2.md`.
