# Policy & Contracts: Guard Bands, Deterministic Reductions, and Safeguards

This topic consolidates TR policy usage (guard bands + hysteresis), deterministic reductions, and second‑order safeguards (layer contracts + curvature bounds).

## TRPolicy (Guard Bands & Hysteresis)

- TRPolicy defines ULP‑scaled guard bands `tau_Q_on/off`, `tau_P_on/off`, keeps signed zeros (directional limits), and toggles deterministic reductions.
- Model‑aware thresholds (recommended): `enable_policy_from_model(model, ulp_scale)` resolves `tau_*` from basis/coeff sensitivity proxies.
- Hysteresis prevents chattering between MR↔SAT; batch quantiles (e.g., q_p10, g90) steer global mode updates.

```python
from zeroproof.training import enable_policy_from_model
enable_policy_from_model(model, ulp_scale=4.0, deterministic_reduction=True)
```

## Deterministic Reductions (pairwise trees)

- When policy enables deterministic reductions, the library uses pairwise summation across:
  - Rational P/Q, TR‑Norm mean/variance, TR‑softmax normalization, dense sums, and regularizers.
- This reduces order‑sensitivity and supports reproducibility guarantees.

## Second‑Order Safeguards (Contracts & Bounds)

- Layers publish a conservative contract `{B_k, H_k, G_max, H_max, depth_hint}` used for monitoring and optional LR clamping.
- The trainer logs a curvature bound derived from (B_k,H_k,G_max,depth_hint) and from per‑batch |Q| stats.

Optional LR clamp (off by default):

```bash
python examples/robotics/rr_ik_train.py \
  --dataset data/rr_ik_dataset.json --model tr_rat \
  --use_contract_safe_lr --contract_c 1.0 --loss_smoothness_beta 1.0
```

## Metrics & Plots

- Summaries/JSON include: `flip_rate`, `saturating_ratio`, `tau_q_on/off`, `q_min_epoch`, `curvature_bound`, and `layer_contract`.
- Plot training curves:

```bash
python scripts/plot_training_curves.py --results runs/ik_experiment/results_tr_rat.json
```

