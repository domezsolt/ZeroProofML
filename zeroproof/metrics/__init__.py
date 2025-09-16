"""Metrics for domain-specific evaluations (robotics, poles, etc.)."""

from .pole_2d import (
    compute_pole_metrics_2d,
    compute_residual_consistency,
    compute_sign_consistency_rate,
    compute_slope_error_near_pole,
    compute_ple_to_lines,
)
from .pole_3r import (
    compute_pole_metrics_3r,
    compute_residual_consistency_3r,
    compute_sign_consistency_rate_3r,
    compute_ple_to_3r_lines,
)

__all__ = [
    'compute_pole_metrics_2d',
    'compute_residual_consistency',
    'compute_sign_consistency_rate',
    'compute_slope_error_near_pole',
    'compute_ple_to_lines',
    # 3R variants
    'compute_pole_metrics_3r',
    'compute_residual_consistency_3r',
    'compute_sign_consistency_rate_3r',
    'compute_ple_to_3r_lines',
]
