# Hybrid + Policy Quickstart

This note shows how to combine the hybrid gradient schedule with a TR policy
that provides ULP‑scaled guard bands and on/off hysteresis.

## TL;DR

1) Enable a default policy using the current precision (float64 by default):

```python
import zeroproof as zp
from zeroproof.training import enable_default_tr_policy

# Guard bands ≈ 4 × machine epsilon; pairwise reductions enabled
enable_default_tr_policy(ulp_scale=4.0, deterministic_reduction=True)
```

2) Use the hybrid trainer with schedule + policy:

```python
from zeroproof.training import HybridTRTrainer, HybridTrainingConfig

cfg = HybridTrainingConfig(
    use_hybrid_gradient=True,
    # Optionally let the trainer set a default policy for you
    use_tr_policy=True,
    policy_ulp_scale=4.0,
    policy_deterministic_reduction=True,
)
trainer = HybridTRTrainer(model, optimizer, cfg)
```

3) Model‑aware thresholds (recommended)

Let the policy derive `tau_Q/P` from the model’s local sensitivity scales:

```python
from zeroproof.training import enable_policy_from_model

# Scales depend on basis/coeff norms; see layer.estimate_local_scales()
enable_policy_from_model(model, ulp_scale=4.0, deterministic_reduction=True)
```

## How it works

- The hybrid scheduler still provides a time‑varying local threshold `delta`.
- When a `TRPolicy` is active, the hybrid controller uses policy guard bands and
  batch quantiles for hysteresis:
  - Enter SAT if q_p10 ≤ tau_Q_on (or g90 ≥ g_on if provided)
  - Exit to MR if q_p10 ≥ tau_Q_off (and g90 ≤ g_off if provided)
- During a batch, HYBRID uses the schedule threshold or `tau_Q_on` to trigger
  local saturation; the batch end updates the global mode using hysteresis.

Deterministic reductions:

- When `deterministic_reduction=True`, rational P/Q, TR‑Norm mean/var, softmax normalization, and dense sums use pairwise reduction trees for order‑stable aggregations.

## Metrics to watch

- `near_pole_ratio`: Fraction of gradient calls flagged near poles.
- `policy_mode`: "MR" or "SAT".
- `q_p10/q_p50/q_p90`: Batch |Q| quantiles against `tau_q_on/off`.

## Notes

- Tags from rational layers use policy classification when a policy is set; values
  still use exact TR algebra.
- `deterministic_reduction=True` switches P/Q to pairwise summation to stabilize
  reduction order.
- To disable the policy, call `TRPolicyConfig.set_policy(None)`.
