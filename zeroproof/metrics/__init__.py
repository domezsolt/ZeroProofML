# MIT License
# See LICENSE file in the project root for full license text.
"""Metrics for domain-specific evaluations (robotics, poles, etc.)."""

from .core import compute_distance_stats, compute_q_stats, hybrid_stats
from .identifiability import compute_sylvester_smin
from .pole_2d import (
    compute_ple_to_lines,
    compute_pole_metrics_2d,
    compute_residual_consistency,
    compute_sign_consistency_rate,
    compute_slope_error_near_pole,
)
from .pole_3r import (
    compute_ple_to_3r_lines,
    compute_pole_metrics_3r,
    compute_residual_consistency_3r,
    compute_sign_consistency_rate_3r,
)

__all__ = [
    "compute_pole_metrics_2d",
    "compute_residual_consistency",
    "compute_sign_consistency_rate",
    "compute_slope_error_near_pole",
    "compute_ple_to_lines",
    # 3R variants
    "compute_pole_metrics_3r",
    "compute_residual_consistency_3r",
    "compute_sign_consistency_rate_3r",
    "compute_ple_to_3r_lines",
    # Core stats
    "compute_q_stats",
    "compute_distance_stats",
    "hybrid_stats",
    # Identifiability
    "compute_sylvester_smin",
]
