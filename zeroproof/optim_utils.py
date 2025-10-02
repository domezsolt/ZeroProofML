# MIT License
# See LICENSE file in the project root for full license text.
"""
Optimizer safety utilities (batch-safe learning rate helpers).

This module provides helper functions for selecting conservative learning rates
based on batch curvature proxies and simple sufficient conditions for common
optimizers (SGD, heavy-ball, Adam).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BatchCurvatureProxy:
    """
    Proxy bounds for safe step selection on a batch.

    L_hat is a bound on local curvature; eta_safe <= 1 / L_hat is sufficient
    for non-exploding updates under basic conditions.
    """

    L_hat: float


def batch_safe_lr(
    B_psi: float,
    q_min: float,
    y_max: float,
    alpha: float = 0.0,
    safety: float = 1.0,
) -> BatchCurvatureProxy:
    """
    Compute a simple batch curvature proxy and return safe-step helper.

    L_batch = (B_psi^2 / q_min^2) * (1 + y_max^2) + alpha

    Args:
        B_psi: Bound on basis features on the batch
        q_min: Minimum |Q(x)| over the batch (strictly positive)
        y_max: Maximum |y(x)| over the batch
        alpha: L2 regularization for denominator parameters
        safety: Optional multiplicative safety factor (>=1)

    Returns:
        BatchCurvatureProxy with L_hat
    """
    q_min = max(q_min, 1e-12)
    L_hat = (B_psi * B_psi / (q_min * q_min)) * (1.0 + y_max * y_max) + max(0.0, alpha)
    return BatchCurvatureProxy(L_hat * max(1.0, safety))


def eta_sgd(proxy: BatchCurvatureProxy) -> float:
    """Safe constant step for SGD: eta <= 1 / L_hat."""
    return 1.0 / max(proxy.L_hat, 1e-12)


def eta_heavy_ball(proxy: BatchCurvatureProxy, beta1: float) -> float:
    """
    Safe step for heavy-ball momentum (sufficient condition):
        eta <= 2 (1 - beta1) / L_hat
    """
    beta1 = min(max(beta1, 0.0), 0.999999)
    return 2.0 * (1.0 - beta1) / max(proxy.L_hat, 1e-12)


def eta_adam(proxy: BatchCurvatureProxy, beta1: float, beta2: float) -> float:
    """
    Safe step for Adam (sufficient condition):
        eta <= (1 - beta1) / (sqrt(1 - beta2) * L_hat)
    """
    import math

    beta1 = min(max(beta1, 0.0), 0.999999)
    beta2 = min(max(beta2, 0.0), 0.999999)
    denom = math.sqrt(max(1.0 - beta2, 1e-12)) * max(proxy.L_hat, 1e-12)
    return (1.0 - beta1) / denom
