"""
TR-Softmax surrogate using rational Padé approximations for exp.

This module provides a softmax-like function built from TR-safe operations.
We avoid true exp; instead we use a Padé (2,2)-style rational approximation
around 0 for exp(z):

    exp(z) ≈ (1 + a1 z + a2 z^2) / (1 - b1 z + b2 z^2)

With small coefficients for stability over moderate z. Inputs are shifted by
max(logits) to reduce dynamic range, similar to log-sum-exp stabilization.

The final softmax is r_i / sum_j r_j, where r_i is the rational approximation
applied to the shifted logits. Division and reductions use TR ops, and the
global deterministic-reduction policy ensures reproducible aggregation.
"""

from __future__ import annotations

from typing import List

from ..autodiff import TRNode, tr_abs, tr_add, tr_div, tr_mul, tr_sub
from ..core import TRScalar, TRTag, real, tr_max


def pade_exp_approx(z: TRNode) -> TRNode:
    """Rational decay surrogate for exp(z) on shifted logits (z ≤ 0 typically).

    We use a simple decreasing rational r(z) = 1 / (1 + c t + d t^2) where
    t = max(0, -z) computed via t = (|z| - z)/2. This preserves r(0)=1 and
    produces small values for large negative z, sufficient for softmax behavior.
    """
    one = TRNode.constant(real(1.0))
    half = TRNode.constant(real(0.5))
    c = TRNode.constant(real(1.0))
    d = TRNode.constant(real(0.01))

    # t = (|z| - z)/2  (equals max(0, -z))
    abs_z = tr_abs(z)
    diff = tr_sub(abs_z, z)
    t = tr_mul(half, diff)
    t2 = tr_mul(t, t)
    den = tr_add(one, tr_add(tr_mul(c, t), tr_mul(d, t2)))
    return tr_div(one, den)


def tr_softmax(logits: List[TRNode]) -> List[TRNode]:
    """TR-safe softmax surrogate using rational exp and TR reductions.

    Args:
        logits: List of TRNodes (can contain non-REAL; non-REALs pass through TR ops)
    Returns:
        List of TRNodes summing to 1 (in REAL regions), with TR divisions handling edge cases.
    """
    if not logits:
        return []

    # Policy: force one-hot if any +INF present
    try:
        from ..policy import TRPolicyConfig

        pol = TRPolicyConfig.get_policy()
        if pol is not None and getattr(pol, "softmax_one_hot_infinity", False):
            idx = None
            for i, x in enumerate(logits):
                if getattr(x, "tag", None) == TRTag.PINF:
                    idx = i
                    break
            if idx is not None:
                out: List[TRNode] = []
                for i in range(len(logits)):
                    out.append(
                        TRNode.constant(real(1.0)) if i == idx else TRNode.constant(real(0.0))
                    )
                return out
    except Exception:
        pass

    # Shift by max to stabilize
    # Compute max over REAL values; if non-REAL present, tr_max handles via TR semantics
    max_val_scalar = tr_max([x.value for x in logits])  # returns TRScalar
    max_val = TRNode.constant(max_val_scalar)

    shifted: List[TRNode] = [x - max_val for x in logits]
    # Apply rational exp approx
    r: List[TRNode] = [pade_exp_approx(z) for z in shifted]

    # Sum as nodes to keep AD path with optional deterministic pairwise reduction
    def _pairwise_sum(nodes: List[TRNode]) -> TRNode:
        if not nodes:
            return TRNode.constant(real(0.0))
        if len(nodes) == 1:
            return nodes[0]
        mid = len(nodes) // 2
        left = _pairwise_sum(nodes[:mid])
        right = _pairwise_sum(nodes[mid:])
        return tr_add(left, right)

    use_pairwise = False
    try:
        from ..policy import TRPolicyConfig

        pol = TRPolicyConfig.get_policy()
        use_pairwise = bool(pol and pol.deterministic_reduction)
    except Exception:
        use_pairwise = False
    if use_pairwise:
        r_sum_node = _pairwise_sum(r)
    else:
        r_sum_node = r[0]
        for ri in r[1:]:
            r_sum_node = tr_add(r_sum_node, ri)
    # Normalize
    out: List[TRNode] = []
    for ri in r:
        out.append(tr_div(ri, r_sum_node))
    return out
