"""
Saturating gradient operations for transreal autodifferentiation.

This module implements gradient computations that saturate (bound) near
singularities instead of zeroing out like the Mask-REAL rule.
"""

from typing import List, Optional, Tuple

from ..core import TRScalar, TRTag, real, tr_add, tr_div, tr_mul, tr_pow_int, tr_sqrt
from .grad_mode import GradientModeConfig
from .tr_node import TRNode


def saturate_value(x: TRScalar, bound: Optional[float] = None) -> TRScalar:
    """
    Saturate a transreal value using TR arithmetic.

    For a value x, computes x / (|x| + bound) which smoothly
    saturates to ±1 as |x| → ∞.

    Args:
        x: Value to saturate
        bound: Saturation bound (uses global default if None)

    Returns:
        Saturated value
    """
    if bound is None:
        bound = GradientModeConfig.get_saturation_bound()

    if x.tag != TRTag.REAL:
        # Non-REAL values saturate to their sign
        if x.tag == TRTag.PINF:
            return real(1.0)
        elif x.tag == TRTag.NINF:
            return real(-1.0)
        else:  # PHI
            return real(0.0)

    # For REAL values, compute x / (|x| + bound)
    # This avoids division by zero and smoothly saturates
    from ..core import tr_abs

    abs_x = tr_abs(x)
    denominator = tr_add(abs_x, real(bound))
    return tr_div(x, denominator)


def saturating_reciprocal(x: TRScalar, bound: Optional[float] = None) -> TRScalar:
    """
    Compute 1/x with saturation near zero.

    Instead of 1/x, computes 1/(x² + bound²)^(1/2) * sign(x)
    which saturates to ±1/bound as x → 0.

    Args:
        x: Value to take reciprocal of
        bound: Saturation bound

    Returns:
        Saturated reciprocal
    """
    if bound is None:
        bound = GradientModeConfig.get_saturation_bound()

    if x.tag != TRTag.REAL:
        # Handle non-REAL cases
        if x.tag in (TRTag.PINF, TRTag.NINF):
            return real(0.0)  # 1/∞ = 0
        else:  # PHI
            return real(0.0)  # Neutral value

    # Compute x² + bound²
    x_squared = tr_mul(x, x)
    bound_squared = real(bound * bound)
    denominator_squared = tr_add(x_squared, bound_squared)

    # Take square root
    denominator = tr_sqrt(denominator_squared)

    # Compute result with correct sign
    from ..core import tr_sign

    sign_x = tr_sign(x)
    result = tr_div(sign_x, denominator)

    return result


def saturating_div_grad(x: TRNode, y: TRNode, grad_output: TRNode) -> Tuple[TRScalar, TRScalar]:
    """
    Compute gradients for division with saturation.

    For z = x/y:
    - ∂z/∂x = 1/y (saturated)
    - ∂z/∂y = -x/y² (saturated)

    Args:
        x: Numerator node
        y: Denominator node
        grad_output: Gradient from downstream

    Returns:
        Tuple of (grad_x, grad_y)
    """
    # Get saturation bound
    bound = GradientModeConfig.get_saturation_bound()

    # Gradient w.r.t. x: grad_output / y
    # Use saturating reciprocal for 1/y
    recip_y = saturating_reciprocal(y.value, bound)
    grad_x = tr_mul(grad_output.value, recip_y)

    # Gradient w.r.t. y: -grad_output * x / y²
    # First compute y² + bound² for saturation
    y_squared = tr_mul(y.value, y.value)
    bound_squared = real(bound * bound)
    denominator = tr_add(y_squared, bound_squared)

    # Compute -x/denominator
    neg_x = tr_mul(real(-1.0), x.value)
    ratio = tr_div(neg_x, denominator)

    # Final gradient
    grad_y = tr_mul(grad_output.value, ratio)

    return grad_x, grad_y


def saturating_rational_grad(
    P: TRNode, Q: TRNode, psi: List[TRNode], grad_output: TRNode
) -> Tuple[List[TRScalar], List[TRScalar]]:
    """
    Compute gradients for rational function P/Q with saturation.

    Args:
        P: Numerator polynomial value
        Q: Denominator polynomial value
        psi: Basis function values
        grad_output: Gradient from downstream

    Returns:
        Tuple of (theta_grads, phi_grads)
    """
    bound = GradientModeConfig.get_saturation_bound()

    # For θ (numerator params): ∂y/∂θ_k = ψ_k/Q (saturated)
    recip_Q = saturating_reciprocal(Q.value, bound)
    theta_grads = []
    for psi_k in psi:
        grad_k = tr_mul(grad_output.value, tr_mul(psi_k.value, recip_Q))
        theta_grads.append(grad_k)

    # For φ (denominator params): ∂y/∂φ_k = -P*ψ_k/Q² (saturated)
    # Compute Q² + bound² for saturation
    Q_squared = tr_mul(Q.value, Q.value)
    bound_squared = real(bound * bound)
    denominator = tr_add(Q_squared, bound_squared)

    # Compute -P/denominator
    neg_P = tr_mul(real(-1.0), P.value)
    ratio = tr_div(neg_P, denominator)

    phi_grads = []
    for psi_k in psi:
        grad_k = tr_mul(grad_output.value, tr_mul(psi_k.value, ratio))
        phi_grads.append(grad_k)

    return theta_grads, phi_grads


def saturating_log_grad(x: TRNode, grad_output: TRNode) -> TRScalar:
    """
    Compute gradient for log with saturation.

    For y = log(x), ∂y/∂x = 1/x (saturated)

    Args:
        x: Input node
        grad_output: Gradient from downstream

    Returns:
        Gradient w.r.t. x
    """
    recip_x = saturating_reciprocal(x.value)
    return tr_mul(grad_output.value, recip_x)


def saturating_sqrt_grad(x: TRNode, grad_output: TRNode) -> TRScalar:
    """
    Compute gradient for sqrt with saturation.

    For y = sqrt(x), ∂y/∂x = 1/(2*sqrt(x)) (saturated)

    Args:
        x: Input node
        grad_output: Gradient from downstream

    Returns:
        Gradient w.r.t. x
    """
    # Compute 2*sqrt(x)
    sqrt_x = tr_sqrt(x.value)
    two_sqrt_x = tr_mul(real(2.0), sqrt_x)

    # Saturating reciprocal
    recip = saturating_reciprocal(two_sqrt_x)
    return tr_mul(grad_output.value, recip)


def saturating_pow_grad(x: TRNode, k: int, grad_output: TRNode) -> TRScalar:
    """
    Compute gradient for integer power with saturation.

    For y = x^k, ∂y/∂x = k*x^(k-1) (with saturation for negative powers)

    Args:
        x: Base node
        k: Integer exponent
        grad_output: Gradient from downstream

    Returns:
        Gradient w.r.t. x
    """
    if k == 0:
        # Derivative of x^0 is 0
        return real(0.0)
    elif k == 1:
        # Derivative of x^1 is 1
        return grad_output.value
    elif k > 0:
        # k*x^(k-1) - no saturation needed for positive powers
        coeff = real(float(k))
        x_pow = tr_pow_int(x.value, k - 1)
        grad_x = tr_mul(coeff, x_pow)
        return tr_mul(grad_output.value, grad_x)
    else:
        # Negative power: k*x^(k-1) = k/x^(1-k)
        # This needs saturation
        coeff = real(float(k))
        x_pow = tr_pow_int(x.value, 1 - k)  # Positive exponent

        # Saturating division
        recip = saturating_reciprocal(x_pow)
        grad_x = tr_mul(coeff, recip)
        return tr_mul(grad_output.value, grad_x)
