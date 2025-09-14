# Topic 7: Evaluation & Metrics (Poles, Coverage, Stability)

How to measure pole learning quality, stability, and coverage; and how to log/visualize results.

## What to Measure
- Pole Localization Error (PLE): Distance between predicted and true poles.
- Sign Consistency: Correctness of ±∞ transitions around poles.
- Asymptotic Behavior: Fit of log|y| vs −log|Q| near poles.
- Residual Consistency: R(x)=Q(x)·y(x)−P(x) near poles should be small.
- Coverage Breakdown: REAL ratios near/mid/far from poles; tag distribution.

## Pole Metrics API
- High‑level evaluator and metrics container: `zeroproof/utils/pole_metrics.py:1`.
- Use when you have true poles (synthetic/analytic) or to compare predicted pole sets.

Example (sketch)
```python
from zeroproof.utils.pole_metrics import PoleEvaluator

true_poles = [0.0, 1.5]
peval = PoleEvaluator(true_poles=true_poles)
metrics = peval.evaluate(x_values, y_values, Q_values, P_values=None,
                         predicted_poles=maybe_predicted)
print(metrics.ple, metrics.sign_consistency)
```

## Integrated Evaluation
- One‑stop evaluation with plotting and JSON logging: `zeroproof/utils/evaluation_api.py:1` (IntegratedEvaluator).
- Configurable thresholds (near/mid distances), which metrics to compute, and visualization cadence.

Usage
```python
from zeroproof.utils.evaluation_api import IntegratedEvaluator, EvaluationConfig

config = EvaluationConfig(enable_visualization=True, plot_frequency=5)
evaluator = IntegratedEvaluator(config=config, true_poles=[0.0])
metrics = evaluator.evaluate_model(model, x_values)
```

Outputs
- Metrics dataclass with PLE, sign consistency, asymptotic slope error/correlation, residual stats, coverage breakdown.
- Optional plots (saved to `plot_dir`), periodic by `plot_frequency`.
- Log file `evaluation_log.json` with timestamped entries.

## Robotics IK (2D) Metrics & Buckets

For the RR‑arm (θ2 singularities at {0, π}), use the 2D helpers and bucketed MSE by |det(J)|≈|sin θ2| to quantify near‑pole behavior.

- 2D metrics helper: `zeroproof/metrics/pole_2d.py`
  - `compute_ple_to_lines(test_inputs, predictions)`: PLE vs θ2∈{0, π}
  - `compute_pole_metrics_2d(test_inputs, predictions)`: bundle of PLE, sign consistency, slope error, residual consistency
- Bucketed test MSE by |det(J)| bins (B0–B4): include both MSE and counts per bucket
  - Edges (default): [0, 1e‑5, 1e‑4, 1e‑3, 1e‑2, inf]
  - Recommended: ensure B0–B3 have non‑zero counts for robust near‑pole evaluation
- Comparator driver outputs both bucketed MSE and 2D pole metrics in JSON
  - Quick profile stratifies the test subset by |det(J)| and aligns DLS to the same subset for parity

## Coverage and Tag Distribution
- Use coverage trackers during training to monitor REAL vs non‑REAL; break down near‑pole coverage using Hybrid stats.
- Code: `zeroproof/training/coverage.py:1` and Hybrid context in `autodiff/hybrid_gradient.py:1`.

## Diagnostics & Reports
- DiagnosticMonitor: batch/epoch histories, q_min/q_mean, tag counts, gradient summaries, exports.
  - Code: `zeroproof/training/sampling_diagnostics.py:480`.
- Verification report: see `docs/verification_report.md:1` for a consolidated spec/behavior check.

## Tests and Reproducibility
- Property tests exercise totality, reduction modes, AD invariants.
  - See `tests/unit/` and `tests/property/`.
- Repro checklist: seeds and deterministic flags (per spec `complete_v2.md:1`).

## Practical Guidance
- Align evaluator thresholds (near/mid) with Hybrid δ for consistent analysis.
- Visualize both y(x) and 1/|Q(x)| overlays to spot missed poles.
- Track PLE over time; expect O(1/√n) decay with sufficient near‑pole samples.
- Persist metrics and plots per checkpoint to compare schedules and policies.
