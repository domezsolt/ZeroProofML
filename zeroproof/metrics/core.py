# MIT License
# See LICENSE file in the project root for full license text.
"""
Core metrics for TR-Rational layers and hybrid controller.

Provides helpers to compute per-batch diagnostics required by the
policy/coverage controller and acceptance criteria:
- q_min and q quantiles
- distance estimator quantiles (|Q|/|Q'| if available)
- hybrid controller statistics (flip rate, %SAT time, thresholds)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from ..autodiff.hybrid_gradient import HybridGradientContext


def compute_q_stats(layer, xs: Any) -> Dict[str, float]:
    """Compute q_min and robust quantiles of |Q(x)| for a batch."""
    q_vals = np.array(layer.get_q_values(xs), dtype=float)
    if q_vals.size == 0:
        return {
            "q_min": math.inf,
            "q_mean": math.nan,
            "q_p10": math.nan,
            "q_p50": math.nan,
            "q_p90": math.nan,
        }
    return {
        "q_min": float(np.min(q_vals)),
        "q_mean": float(np.mean(q_vals)),
        "q_p10": float(np.percentile(q_vals, 10)),
        "q_p50": float(np.percentile(q_vals, 50)),
        "q_p90": float(np.percentile(q_vals, 90)),
    }


def compute_distance_stats(layer, xs: Any) -> Dict[str, float]:
    """Compute robust quantiles for the distance estimator d(x)."""
    try:
        d_vals = np.array(layer.estimate_distance_batch(xs), dtype=float)
    except Exception:
        # If layer doesn't implement estimator, fall back to q_vals
        d_vals = np.array(layer.get_q_values(xs), dtype=float)
    if d_vals.size == 0:
        return {
            "d_p10": math.nan,
            "d_p50": math.nan,
            "d_p90": math.nan,
        }
    return {
        "d_p10": float(np.percentile(d_vals, 10)),
        "d_p50": float(np.percentile(d_vals, 50)),
        "d_p90": float(np.percentile(d_vals, 90)),
    }


def hybrid_stats() -> Dict[str, float]:
    """Expose hybrid controller statistics for logging/monitoring."""
    return HybridGradientContext.get_statistics()
