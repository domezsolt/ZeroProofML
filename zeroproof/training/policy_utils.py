"""
Helpers for configuring TRPolicy in training.

Provides a simple function to enable a default TRPolicy using the current
precision (float32/float64) and a scale factor for ULP-based guard bands.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple

from ..core import PrecisionConfig
from ..policy import TRPolicy, TRPolicyConfig


def enable_default_tr_policy(
    ulp_scale: float = 4.0,
    deterministic_reduction: bool = True,
    g_on: Optional[float] = None,
    g_off: Optional[float] = None,
) -> TRPolicy:
    """
    Enable a default TRPolicy using current precision and simple ULP-scaled bands.

    Args:
        ulp_scale: Multiplier for machine epsilon to set guard bands (τ ≈ ulp_scale * eps)
        deterministic_reduction: Use deterministic pairwise reductions for P/Q
        g_on: Optional sensitivity trigger to enter SAT (e.g., 1/|Q| 90th percentile)
        g_off: Optional sensitivity trigger to exit SAT

    Returns:
        The configured TRPolicy
    """
    eps = float(PrecisionConfig.get_epsilon())
    ulp = max(0.0, float(ulp_scale)) * eps

    pol = TRPolicy(
        rounding_mode="nearest_even",
        keep_signed_zero=True,
        deterministic_reduction=bool(deterministic_reduction),
        g_on=g_on,
        g_off=g_off,
    )
    pol.resolve_thresholds(ulp=ulp, local_scale_q=1.0, local_scale_p=1.0)
    TRPolicyConfig.set_policy(pol)
    return pol


class _SupportsLocalScales(Protocol):
    def estimate_local_scales(self) -> Tuple[float, float]:
        ...


def enable_policy_from_model(
    model: Any,
    ulp_scale: float = 4.0,
    deterministic_reduction: bool = True,
    g_on: Optional[float] = None,
    g_off: Optional[float] = None,
) -> TRPolicy:
    """
    Enable a TRPolicy whose guard bands are auto-resolved from the model.

    Uses the model's local sensitivity proxies to scale τ thresholds:
      local_scale_q ≈ 1 + B · ||φ||₁, local_scale_p ≈ 1 + B · ||θ||₁

    Args:
        model: Layer/model exposing estimate_local_scales() -> (scale_q, scale_p)
        ulp_scale: Multiplier for machine epsilon to set base guard bands
        deterministic_reduction: Use deterministic pairwise reductions for P/Q
        g_on: Optional sensitivity trigger to enter SAT (e.g., 1/|Q| quantile)
        g_off: Optional sensitivity trigger to exit SAT

    Returns:
        The configured TRPolicy
    """
    eps = float(PrecisionConfig.get_epsilon())
    ulp = max(0.0, float(ulp_scale)) * eps

    # Fallback scales when model lacks estimator
    try:
        # Try Protocol method if available
        local_scale_q, local_scale_p = getattr(model, "estimate_local_scales")()
    except Exception:
        local_scale_q, local_scale_p = 1.0, 1.0

    pol = TRPolicy(
        rounding_mode="nearest_even",
        keep_signed_zero=True,
        deterministic_reduction=bool(deterministic_reduction),
        g_on=g_on,
        g_off=g_off,
    )
    pol.resolve_thresholds(
        ulp=ulp,
        local_scale_q=float(max(1.0, local_scale_q)),
        local_scale_p=float(max(1.0, local_scale_p)),
    )
    TRPolicyConfig.set_policy(pol)
    return pol
