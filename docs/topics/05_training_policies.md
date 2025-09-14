# Topic 5: Training Policies (Adaptive Loss, Coverage, Tag Loss)

How ZeroProof trains models that may emit non‑REAL values, while keeping learning stable and coverage on target.

## Problem Setting
- Outputs can be REAL, PINF, NINF, or PHI; only REAL contributes to the main loss.
- We want high REAL coverage (e.g., 95%) without destabilizing gradients.
- Mask‑REAL ensures stability; policies manage incentives and measurements.

## Adaptive Rejection Penalty (λ_rej)
- Treat λ_rej as a Lagrange multiplier to meet target coverage c*.
- Update: λ ← λ + η_λ (c* − c_actual), with momentum, warmup, optional decay.
- Floors: `lambda_rej_min` prevents trivial rejection; `lambda_min` soft floor.
- Stats: Tracks λ history, coverage (batch, cumulative, window).
- Code: `zeroproof/training/adaptive_loss.py:1` (AdaptiveLossConfig, AdaptiveLambda).

Usage
```python
from zeroproof.training.adaptive_loss import AdaptiveLambda, AdaptiveLossConfig
from zeroproof.core import TRTag

cfg = AdaptiveLossConfig(target_coverage=0.95, learning_rate=0.01,
                         initial_lambda=1.0, momentum=0.9,
                         lambda_rej_min=0.1, update_frequency=10)
policy = AdaptiveLambda(cfg)

# After a batch
tags = [TRTag.REAL, TRTag.PINF, TRTag.REAL]
policy.update(tags)
stats = policy.get_statistics()
print(stats["current_coverage"], policy.lambda_rej)
```

## Coverage Tracking
- Batch and cumulative coverage, with optional sliding window.
- Tag distribution counts maintained for REAL/PINF/NINF/PHI.
- Multi‑output tracker supports aggregation across heads.
- Code: `zeroproof/training/coverage.py:1` (CoverageTracker, MultiOutputCoverageTracker).

## Tag‑Aware Loss (Optional)
- Cross‑entropy between predicted tags and actual tags to encourage correct tag prediction/exploration.
- Adaptive tag loss weight can increase when coverage approaches 100%.
- Code: `zeroproof/training/tag_loss.py:1`.

## Integration with Layers
- TRRational accepts `adaptive_loss_policy` to read λ and report tags.
- Hybrid layers expose q_min and near‑pole stats; combine with coverage to tune schedules.
- For pole head variants, include pole loss/regularizer in total loss as needed.

## Trainer Utilities
- Generic Trainer and HybridTrainer orchestrate epochs, schedule updates, and metric logging.
- Code: `zeroproof/training/trainer.py:1`, `zeroproof/training/hybrid_trainer.py:1`.
 - Bench metrics (per‑epoch): Hybrid trainer prints and records `avg_step_ms`, `data_time_ms`, `optim_time_ms`, and `batches`. These are returned in training summaries under `bench_history`. Logging cadence is controlled by `log_interval` (CLI `--log_every` in robotics examples).

## Practical Tips
- Start with target_coverage ~0.9–0.98; too high can slow learning early.
- Use momentum and update_frequency to smooth λ evolution; dead‑band reduces oscillations.
- Inspect stats: coverage_gap near zero, λ stable above `lambda_rej_min`.
- Combine with Hybrid gradients if pole learning is a goal.

## See Also
- Guide: `docs/adaptive_loss_guide.md:1`
- Concepts: `docs/topics/03_autodiff_modes.md:1` (Hybrid decisions pair well with coverage)
- Theory clarifications: `complete_v2.md:1` (loss policy and determinism)
