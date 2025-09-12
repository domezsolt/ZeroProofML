# Topic 6: Sampling & Curriculum (Near‑Pole Focus)

Strategies to expose the model to informative near‑pole data without destabilizing training.

## Why Sampling Matters
- Poles are defined by Q(x)=0; informative gradients live near pole neighborhoods.
- Uniform sampling under‑represents near‑poles; importance/active sampling boosts coverage of hard regions.

## Importance Sampling (1/|Q|^p)
- Weight samples by 1/(|Q(x)|^power), with clipping for stability.
- Temperature and max_weight guard distribution sharpness.
- Hybrid batches: mix importance and uniform subsets.
- Code: `zeroproof/training/sampling_diagnostics.py:1` (ImportanceSampler, ImportanceSamplerConfig).

Usage
```python
from zeroproof.training.sampling_diagnostics import ImportanceSampler, ImportanceSamplerConfig

sampler = ImportanceSampler(ImportanceSamplerConfig(weight_power=2.0, max_weight=100.0))
# Given pools x_pool, q_pool (tensors)
batch_x, idx = sampler.sample_batch(x_pool, q_pool, batch_size=128)
```

## Active Sampling (Grid Refinement)
- Maintain a grid over the input domain; refine where |Q| is small.
- Controls: refinement_threshold, max_refinement_level, bounds, memory caps.
- Tracks refinement history and q statistics.
- Code: `zeroproof/training/sampling_diagnostics.py:200` (ActiveSampler, ActiveSamplerConfig).

## Hybrid Strategies
- Combine uniform + importance; or alternate active refinement epochs.
- Balance exploration (coverage of space) and exploitation (near‑poles).

## Curriculum Learning
- Stage difficulty: far → mid → near pole regions based on |Q| bands.
- Works well with Hybrid gradient schedules to avoid early instability.
- Reference concepts: `concept_250908.md: Part VI`.

## Diagnostics
- Monitor q_min (batch/epoch), mean |Q|, near‑pole ratio, sample diversity.
- DiagnosticMonitor exports histories and JSON snapshots.
- Code: `zeroproof/training/sampling_diagnostics.py:480` (DiagnosticMonitor).

## Practical Tips
- Clip |Q| with min_q_abs in weights to avoid extreme peaking.
- Keep an importance_batch_ratio < 1 to retain coverage of easy regions.
- Use adaptive_delta in Hybrid schedule to align sampler focus and gradient mode.
- Persist sampler state for reproducibility when needed.

## See Also
- Autodiff Hybrid: `docs/topics/03_autodiff_modes.md:1`
- Training Policies: `docs/topics/05_training_policies.md:1`
- Conceptual background: `concept_250908.md:1` (Sampling strategies)
