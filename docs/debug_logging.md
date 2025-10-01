# Debug Logging Guide

This guide shows how to enable and use ZeroProofML’s structured logging during
experiments, and how to inspect the resulting artifacts.

## Enable Logging

ZeroProofML uses Python’s `logging` and a structured logger for training.

- Basic logging to console:

```python
import logging

# INFO is a good default; use DEBUG for more detail
logging.basicConfig(level=logging.INFO)
```

- Per‑module control (examples):

```python
import logging

logging.getLogger("zeroproof").setLevel(logging.INFO)
logging.getLogger("zeroproof.utils.evaluation_api").setLevel(logging.DEBUG)
logging.getLogger("zeroproof.utils.plotting").setLevel(logging.WARNING)
```

## Structured Training Logs

Use `zeroproof.utils.logging.StructuredLogger` to capture per‑step metrics in a
machine‑readable way, including coverage, policy/hybrid stats, timing, and
anti‑illusion metrics.

```python
from zeroproof.utils.logging import StructuredLogger, log_training_step

logger = StructuredLogger(run_dir="runs/demo")
logger.set_config({"max_epochs": 10, "batch_size": 256})
logger.set_model_info({"model": "TRRational d_p=3 d_q=2"})

for epoch in range(1, 11):
    # ... training step code ...
    # metrics can include: loss, coverage, lambda_rej, avg_step_ms, delta,
    # saturating_ratio, q_p10/p50/p90, flip_rate, etc.
    metrics = {"loss": 0.01 * epoch, "coverage": 0.95}
    logger.log_metrics(metrics, epoch=epoch, step=epoch)

# Save JSON and CSV
json_path = logger.save()
csv_path = logger.save_csv()
summary_path = logger.export_summary()
print("Saved:", json_path, csv_path, summary_path)
```

Artifacts are written under `run_dir`:

- `*_logs.json` — full session with all per‑step metrics
- `*_metrics.csv` — flat table for quick spreadsheet/plotting
- `summary.json` — compact summary with aggregates

## Trainer/Examples Integration

- Many trainer paths print timing and hybrid policy metrics to console each
  epoch when verbose; see `HybridTRTrainer._log_epoch`.
- The RR‑IK CLI exposes a convenience flag to print policy/hybrid metrics:

```bash
python examples/robotics/rr_ik_train.py --dataset data/rr_ik.json \
  --log_policy_console
```

## Common Fields

Metrics commonly logged by training utilities include:

- `loss`, `coverage`, `lambda_rej`, `tag_loss`, `pole_loss`
- Timing: `avg_step_ms`, `data_time_ms`, `optim_time_ms`, `batches`
- Hybrid: `saturating_ratio`, `flip_rate`, `tau_q_on`, `tau_q_off`, `tau_p_on`, `tau_p_off`
- Near‑pole stats: `q_min`, `q_p10/p50/p90`, `g_p10/p50/p90`, `near_pole_ratio`
- Second‑order proxies: `curvature_proxy`, `grad_max`, `gn_proxy`

Tip: log level `INFO` shows concise progress; use `DEBUG` for more granular
details during development.

