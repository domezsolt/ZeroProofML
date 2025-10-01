"""
Overhead envelope reporting for Hybrid vs Mask‑REAL baseline.

Provides a utility to compare per-epoch timing and hybrid activation stats
between a Mask‑REAL baseline and the current Hybrid configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable, Optional
import logging

from ..autodiff.grad_mode import GradientMode, GradientModeConfig
from ..autodiff.hybrid_gradient import HybridGradientContext

logger = logging.getLogger(__name__)

def _collect_learning_rates(trainer) -> Dict[str, float]:
    lrs: Dict[str, float] = {}
    try:
        lrs['main'] = float(trainer.optimizer.learning_rate)
    except Exception:
        pass
    try:
        if getattr(trainer, 'frontend_optimizer', None) is not None:
            lrs['frontend'] = float(trainer.frontend_optimizer.learning_rate)
    except Exception:
        pass
    try:
        if getattr(trainer, 'head_optimizers', None) is not None:
            # store first as representative and apply equally
            if trainer.head_optimizers:
                lrs['head'] = float(trainer.head_optimizers[0].learning_rate)
    except Exception:
        pass
    return lrs


def _apply_learning_rates(trainer, lrs: Dict[str, float]) -> None:
    try:
        if 'main' in lrs:
            trainer.optimizer.learning_rate = lrs['main']
    except Exception:
        pass
    try:
        if 'frontend' in lrs and getattr(trainer, 'frontend_optimizer', None) is not None:
            trainer.frontend_optimizer.learning_rate = lrs['frontend']
    except Exception:
        pass
    try:
        if 'head' in lrs and getattr(trainer, 'head_optimizers', None) is not None:
            for opt in trainer.head_optimizers:
                opt.learning_rate = lrs['head']
    except Exception:
        pass


def _set_all_learning_rates(trainer, value: float) -> None:
    try:
        trainer.optimizer.learning_rate = value
    except Exception:
        pass
    try:
        if getattr(trainer, 'frontend_optimizer', None) is not None:
            trainer.frontend_optimizer.learning_rate = value
    except Exception:
        pass
    try:
        if getattr(trainer, 'head_optimizers', None) is not None:
            for opt in trainer.head_optimizers:
                opt.learning_rate = value
    except Exception:
        pass


def _run_epoch_generic(trainer, data_loader: List[Tuple[List, List]]) -> Dict[str, Any]:
    """Run a generic epoch over data_loader and return timing/metrics.

    Uses per-batch internal APIs to support both scalar and vector inputs.
    Learning rates should already be set by caller (often 0.0 for benchmarking).
    """
    import time
    total_step_ms = 0.0
    total_optim_ms = 0.0
    n_batches = 0
    for (inputs, targets) in data_loader:
        t0 = time.perf_counter()
        try:
            # Scalar path expects List[TRScalar]
            result = trainer._train_batch(inputs, targets, None)  # type: ignore[arg-type]
        except Exception:
            # Vector path expects List[List[TRNode/TRScalar floats]]
            result = trainer._train_batch_multi(inputs, targets)  # type: ignore[arg-type]
        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000.0
        optim_ms = float(result.get('optim_ms', 0.0)) if isinstance(result, dict) else 0.0
        total_step_ms += step_ms
        total_optim_ms += optim_ms
        n_batches += 1
    # Gather hybrid stats after epoch
    stats = HybridGradientContext.get_statistics()
    avg_step_ms = total_step_ms / max(1, n_batches)
    return {
        'avg_step_ms': avg_step_ms,
        'optim_time_ms': total_optim_ms / max(1, n_batches),
        'batches': float(n_batches),
        'saturating_ratio': stats.get('saturating_ratio', 0.0),
        'saturating_activations': stats.get('saturating_activations', None),
        'total_gradient_calls': stats.get('total_gradient_calls', None),
        'mask_real_activations': stats.get('mask_real_activations', None),
    }


def overhead_report(trainer, data_loader: List[Tuple[List, List]] ) -> Dict[str, Any]:
    """
    Compare Mask‑REAL baseline vs Hybrid epoch timings and stats.

    The trainer is used for both runs; we temporarily set learning rates to 0
    to avoid changing parameters during the benchmark. Original learning rates
    and hybrid schedule are restored afterward.
    """
    # Snapshot schedule and learning rates
    orig_schedule = getattr(trainer, 'hybrid_schedule', None)
    orig_lrs = _collect_learning_rates(trainer)

    # Baseline: Mask‑REAL (no hybrid schedule)
    try:
        trainer.hybrid_schedule = None
    except Exception:
        pass
    HybridGradientContext.reset()
    GradientModeConfig.set_mode(GradientMode.MASK_REAL)
    _set_all_learning_rates(trainer, 0.0)
    try:
        baseline = trainer.train_epoch(data_loader)
    except Exception:
        baseline = _run_epoch_generic(trainer, data_loader)

    # Hybrid: restore schedule and run HYBRID
    if orig_schedule is not None:
        try:
            trainer.hybrid_schedule = orig_schedule
        except Exception:
            pass
    HybridGradientContext.reset()
    if orig_schedule is not None:
        HybridGradientContext.set_schedule(orig_schedule)
        HybridGradientContext.update_epoch(getattr(trainer, 'epoch', 0))
    GradientModeConfig.set_mode(GradientMode.HYBRID)
    _set_all_learning_rates(trainer, 0.0)
    try:
        hybrid = trainer.train_epoch(data_loader)
    except Exception:
        hybrid = _run_epoch_generic(trainer, data_loader)

    # Restore learning rates and schedule
    _apply_learning_rates(trainer, orig_lrs)
    try:
        trainer.hybrid_schedule = orig_schedule
    except Exception:
        pass

    # Compute overhead summary
    b_ms = float(baseline.get('avg_step_ms', float('nan')))
    h_ms = float(hybrid.get('avg_step_ms', float('nan')))
    slowdown = (h_ms / b_ms) if (b_ms > 0 and h_ms == h_ms) else float('nan')
    sat_ratio = float(hybrid.get('saturating_ratio', 0.0))
    mask_acts = hybrid.get('mask_real_activations', None)
    sat_acts = hybrid.get('saturating_activations', None)
    total_calls = hybrid.get('total_gradient_calls', None)

    report = {
        'baseline': {
            'avg_step_ms': b_ms,
            'batches': baseline.get('batches'),
        },
        'hybrid': {
            'avg_step_ms': h_ms,
            'batches': hybrid.get('batches'),
            'saturating_ratio': sat_ratio,
            'saturating_activations': sat_acts,
            'total_gradient_calls': total_calls,
            'mask_real_activations': mask_acts,
            'mask_bandwidth': (float(mask_acts) / float(total_calls)) if isinstance(mask_acts, (int, float)) and isinstance(total_calls, (int, float)) and total_calls > 0 else None,
        },
        'slowdown_x': slowdown,
    }

    # Print compact summary
    try:
        sat_part = f" sat={sat_ratio:.3f}"
        if isinstance(sat_acts, (int, float)) and isinstance(total_calls, (int, float)):
            sat_part += f" ({sat_acts}/{total_calls})"
        logger.info(
            "Overhead: baseline=%.2fms hybrid=%.2fms slowdown=%.2fx;%s",
            b_ms, h_ms, slowdown, sat_part
        )
    except Exception:
        pass

    return report


def compare_tr_vs_float(
    name: str,
    tr_func: Callable[[], Any],
    float_func: Callable[[], Any],
    iterations: int = 10000,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Compare runtime of a TR function vs a pure-float function.

    Returns a dict with mean runtimes and slowdown factor.
    """
    import time
    def timeit(fn: Callable[[], Any]) -> float:
        times: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(iterations):
                fn()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return sum(times) / len(times)
    tr_s = timeit(tr_func)
    fl_s = timeit(float_func)
    return {
        'name': name,
        'tr_sec': tr_s,
        'float_sec': fl_s,
        'slowdown_x': (tr_s / fl_s) if fl_s > 0 else float('inf'),
        'iterations': iterations,
        'repeats': repeats,
    }
