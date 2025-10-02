# MIT License
# See LICENSE file in the project root for full license text.
"""
Second-order safeguards and curvature bound utilities.

This module provides lightweight helpers to estimate conservative bounds
on second-order curvature (e.g., Hessian operator norms) for TR models,
with a focus on Mask-REAL (MR) vs. Saturating (SAT) gradient modes.

The design follows the extended ZeroProofML theory:
 - On REAL regions (MR), second-order derivatives coincide with classical
   values; users can supply per-primitive bounds (B_k, H_k).
 - On SAT regions, bounded surrogates ensure finite curvature; we expose
   surrogate bounds (G_max, H_max) driven by the global saturation bound.

These functions provide practical envelopes rather than exact constants;
they are intended for monitoring, safety checks, and batch-safe step sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .autodiff.grad_mode import GradientModeConfig


@dataclass
class SecondOrderContract:
    """
    Second-order contract summarizing conservative bounds.

    Attributes:
        B_k: Upper bound on first derivatives of REAL primitives (aggregate)
        H_k: Upper bound on second derivatives of REAL primitives (aggregate)
        G_max: Upper bound for SAT surrogate magnitude/first-derivative proxy
        H_max: Upper bound for SAT surrogate second-derivative proxy
        depth_hint: Approximate composition depth (used to combine bounds)
    """

    B_k: float = 1.0
    H_k: float = 1.0
    G_max: float = 1.0
    H_max: float = 1.0
    depth_hint: int = 4


def saturating_surrogate_bounds(bound: Optional[float] = None) -> Tuple[float, float]:
    """
    Derive surrogate bounds (G_max, H_max) from the global saturation bound.

    For common SAT surrogates (reciprocal/log/sqrt in this repo):
      - 1/sqrt(x^2 + b^2) ≤ 1/b for all x (first-derivative magnitude proxy)
      - d/dx 1/sqrt(x^2 + b^2) = -x/(x^2+b^2)^(3/2) has magnitude ≤ 1/b^2

    We return (G_max, H_max) = (max(1, 1/b), max(1, 1/b^2)) to stay safe when
    b is small and to avoid degenerate zeros.
    """
    if bound is None:
        bound = GradientModeConfig.get_saturation_bound()
    b = float(bound if bound is not None and bound > 0 else 1.0)
    g = max(1.0, 1.0 / b)
    h = max(1.0, 1.0 / (b * b))
    return g, h


def combine_path_bound(
    contract: SecondOrderContract, mr_paths: int = 1, sat_paths: int = 0
) -> float:
    """
    Combine per-primitive bounds into a coarse global curvature bound.

    We use a product-of-factors envelope inspired by the theory's path-wise
    composition: C_H ≲ C0 * Π_k c_k. Here we approximate with
        c_MR = (B_k^2 + H_k) and c_SAT = (G_max^2 + H_max),
    raised to a depth hint, scaled by the number of effective paths.

    Args:
        contract: SecondOrderContract
        mr_paths: Number of effective REAL-only composition paths
        sat_paths: Number of effective SAT-surrogate composition paths

    Returns:
        Curvature bound scalar (unitless envelope)
    """
    c_mr = max(1.0, (contract.B_k**2) + contract.H_k)
    c_sat = max(1.0, (contract.G_max**2) + contract.H_max)
    depth = max(1, int(contract.depth_hint))
    # Aggregate: (mr_paths * c_mr^depth + sat_paths * c_sat^depth)
    return float(mr_paths) * (c_mr**depth) + float(sat_paths) * (c_sat**depth)


def estimate_contract_for_tr_rational(layer) -> SecondOrderContract:
    """
    Heuristic contract for TRRational-like layers.

    Uses basis bound and parameter L1 norms as proxies for MR bounds and the
    global saturation bound for SAT bounds.

    Returns:
        SecondOrderContract with conservative defaults.
    """
    # Basis bound B (monomial basis typically exposes .bound)
    try:
        B = float(getattr(getattr(layer, "basis", None), "bound", 1.0))
    except Exception:
        B = 1.0

    # Parameter L1 norms (REAL entries only)
    def _l1(nodes) -> float:
        s = 0.0
        try:
            for n in nodes:
                v = getattr(n, "value", None)
                if v is not None and getattr(v, "tag", None) is not None:
                    if v.tag.name == "REAL":  # safe string check
                        try:
                            s += abs(float(v.value))
                        except Exception:
                            pass
        except Exception:
            return 0.0
        return s

    theta_l1 = _l1(getattr(layer, "theta", []))
    phi_l1 = _l1(getattr(layer, "phi", []))

    # MR aggregate bounds: proportional to basis influence and parameter norms
    # B_k ~ O(1 + B * (||theta||_1 + ||phi||_1)), H_k set to similar scale
    scale = 1.0 + B * (theta_l1 + phi_l1)
    B_k = max(1.0, scale)
    H_k = max(1.0, scale)

    # SAT surrogate bounds from global saturation bound
    G_max, H_max = saturating_surrogate_bounds()

    # Depth hint: number of primitive compositions in P/Q evaluation is
    # roughly deg(P)+deg(Q)+constant; use a safe minimum of 4
    try:
        d_p = int(getattr(layer, "d_p", 2) or 2)
        d_q = int(getattr(layer, "d_q", 1) or 1)
        depth_hint = max(4, d_p + d_q + 2)
    except Exception:
        depth_hint = 6

    return SecondOrderContract(B_k=B_k, H_k=H_k, G_max=G_max, H_max=H_max, depth_hint=depth_hint)


def curvature_bound_for_batch(
    layer, xs: Any, mr_paths: int = 1, sat_paths: int = 1
) -> Dict[str, float]:
    """
    Compute a conservative curvature bound envelope for a batch.

    Heuristics:
      - If q_min (min |Q| over batch) is small, SAT dominates; otherwise MR.
      - Combine both with provided path counters to remain conservative.

    Args:
        layer: TRRational-like layer exposing get_q_values(xs)
        xs: batch inputs
        mr_paths: weight for REAL-only path composition
        sat_paths: weight for SAT-surrogate path composition

    Returns:
        Dict with 'q_min', 'contract' fields and 'curvature_bound'.
    """
    # q_min estimation
    try:
        q_vals = layer.get_q_values(xs)
        q_min = min([float(v) for v in q_vals]) if q_vals else float("inf")
    except Exception:
        q_min = float("inf")

    contract = estimate_contract_for_tr_rational(layer)

    # If clearly away from poles, prefer MR; near poles, include SAT weight
    mr_w = mr_paths
    sat_w = sat_paths
    try:
        thr = GradientModeConfig.get_local_threshold()
    except Exception:
        thr = None
    threshold = float(thr) if thr is not None else 1e-6
    if q_min >= threshold:
        sat_w = max(0, sat_paths - 1)  # damp SAT when far from poles

    bound = combine_path_bound(contract, mr_paths=mr_w, sat_paths=sat_w)
    return {
        "q_min": float(q_min),
        "B_k": float(contract.B_k),
        "H_k": float(contract.H_k),
        "G_max": float(contract.G_max),
        "H_max": float(contract.H_max),
        "depth_hint": float(contract.depth_hint),
        "curvature_bound": float(bound),
    }


def gauss_newton_bound(grad_norm_sq: float, y_max: float, eps: float = 1e-12) -> float:
    """
    Simple Gauss–Newton spectral envelope using gradient norm proxy.

    For squared error losses, a coarse spectral bound is O(‖J‖^2) where J is
    the Jacobian; we use grad_norm_sq as a proxy for ‖J‖^2 and return
    max(grad_norm_sq, 1) to avoid degenerate zeros.
    """
    try:
        g2 = float(grad_norm_sq)
    except Exception:
        g2 = 1.0
    g2 = max(g2, eps)
    return g2
