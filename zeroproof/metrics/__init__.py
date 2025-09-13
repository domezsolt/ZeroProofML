"""Metrics for domain-specific evaluations (robotics, poles, etc.)."""

from .pole_2d import (
    compute_pole_metrics_2d,
    compute_residual_consistency,
    compute_sign_consistency_rate,
    compute_slope_error_near_pole,
    compute_ple_to_lines,
)

__all__ = [
    'compute_pole_metrics_2d',
    'compute_residual_consistency',
    'compute_sign_consistency_rate',
    'compute_slope_error_near_pole',
    'compute_ple_to_lines',
]

