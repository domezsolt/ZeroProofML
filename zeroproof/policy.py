# MIT License
# See LICENSE file in the project root for full license text.
"""
Transreal tag policy and guard-band classification.

This module centralizes policy configuration for tag decisions (REAL/INF/NULL)
around poles in rational layers, including basic guard bands and optional
hysteresis. It does not alter TR algebra; instead, it provides a deterministic
classification used by layers, controllers, and metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .core import TRScalar, TRTag


@dataclass
class TRPolicy:
    """
    Tagging and numerical policy for TR layers.

    Contracts and semantics:
    - Tags: REAL, PINF, NINF, PHI. Forward computations obey TR algebra; policy
      only classifies between these where guard bands apply.
    - Guard bands (hysteresis):
      - Enter non-REAL region when |Q| <= tau_Q_on
      - Return to REAL region only when |Q| >= tau_Q_off (tau_Q_off > tau_Q_on)
      - Within the band, choose INF vs PHI via |P| thresholds (tau_P_on/off)
    - Reduction modes: Core reductions support STRICT (default) and DROP_NULL
      behaviors. STRICT propagates PHI/⊥ and enforces ∞ + (−∞) = PHI; DROP_NULL
      ignores PHI/⊥ and reduces over remaining values. Deterministic reductions
      (pairwise trees/compensated sums) can be toggled via this policy.
    - Hybrid schedule knobs: Hysteresis can be extended with extra on/off
      thresholds (τ_on/τ_off) and schedule deltas (δ) at higher levels; this
      policy exposes tau_* and plugs into layers that consult it.

    Rounding/reduction flags are placeholders for backends; classification
    itself only uses the tau_* thresholds.
    """

    tau_Q_on: float = 0.0
    tau_Q_off: float = 0.0
    tau_P_on: float = 0.0
    tau_P_off: float = 0.0

    rounding_mode: str = "nearest_even"
    keep_signed_zero: bool = True
    deterministic_reduction: bool = True

    # Optional sensitivity triggers (unused in base classification)
    g_on: Optional[float] = None
    g_off: Optional[float] = None

    # Softmax policy: when True and any logit is +INF, force one-hot at
    # the first +INF index (deterministic tie-breaker) instead of propagating
    # non-REAL tags through the softmax surrogate.
    softmax_one_hot_infinity: bool = False

    def resolve_thresholds(
        self, ulp: float, local_scale_q: float = 1.0, local_scale_p: float = 1.0
    ) -> None:
        """
        Derive guard bands from a unit-in-last-place (ULP) and local sensitivities.

        The thresholds are scaled as:
            tau_Q_on  = c_on  * ULP * local_scale_q
            tau_Q_off = c_off * ULP * local_scale_q
            tau_P_on  = c_on  * ULP * local_scale_p
            tau_P_off = c_off * ULP * local_scale_p

        where local_scale_q/p can reflect coefficient norms (e.g., 1 + B·||φ||₁
        and 1 + B·||θ||₁ using a basis bound B) or estimates of ∥∇Q∥, ∥∇P∥.

        Updates the instance in-place.
        """
        # Ensure nonnegative inputs
        ulp = max(0.0, float(ulp))
        local_scale_q = max(1.0, float(local_scale_q))
        local_scale_p = max(1.0, float(local_scale_p))

        base_q = ulp * local_scale_q
        base_p = ulp * local_scale_p

        # Set modest hysteresis by default (factor 2)
        self.tau_Q_on = base_q
        self.tau_Q_off = 2.0 * base_q

        self.tau_P_on = base_p
        self.tau_P_off = 2.0 * base_p


class TRPolicyConfig:
    """Global access to the currently active TR policy."""

    _active_policy: Optional[TRPolicy] = None

    @classmethod
    def set_policy(cls, policy: Optional[TRPolicy]) -> None:
        cls._active_policy = policy
        # Propagate deterministic reduction preference to core reductions
        try:
            from .core.reductions import set_deterministic_reduction

            if policy is not None:
                set_deterministic_reduction(bool(policy.deterministic_reduction))
            else:
                set_deterministic_reduction(False)
        except Exception:
            pass

    @classmethod
    def get_policy(cls) -> Optional[TRPolicy]:
        return cls._active_policy


def _sign_from_value(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    # x==0.0: use IEEE signed zero if available
    return -1 if math.copysign(1.0, x) < 0.0 else 1


def classify_tag_with_policy(
    policy: TRPolicy,
    p: TRScalar,
    q: TRScalar,
    default_tag: TRTag,
    prev_policy_tag: Optional[TRTag] = None,
) -> TRTag:
    """
    Classify output tag (REAL/PINF/NINF/PHI) from P, Q using guard bands.

    - Outside guard band (|Q| >= tau_Q_off): return REAL
    - Inside band (|Q| < tau_Q_on): decide between INF vs PHI based on |P|
      and sign(p)/sign(q).
    - Between ON/OFF thresholds: apply simple hysteresis using prev_policy_tag.

    If P or Q are non-REAL, fall back to the provided default_tag (TR algebra).
    """
    # Fallback to TR semantics if inputs are non-REAL
    if p.tag != TRTag.REAL or q.tag != TRTag.REAL:
        return default_tag

    q_abs = abs(float(q.value))
    p_abs = abs(float(p.value))

    # Determine current thresholds using hysteresis
    if prev_policy_tag in (TRTag.PINF, TRTag.NINF, TRTag.PHI):
        tau_q_enter = policy.tau_Q_on
        tau_q_exit = policy.tau_Q_off
        tau_p_th = policy.tau_P_on  # within band
        # If safely outside OFF threshold, go REAL
        if q_abs >= tau_q_exit:
            return TRTag.REAL
        # Still in band; choose INF vs PHI
        if q_abs <= tau_q_enter:
            # Decide INF vs PHI by |P|
            if p_abs >= tau_p_th:
                sign = _sign_from_value(float(p.value)) * _sign_from_value(float(q.value))
                return TRTag.PINF if sign >= 0 else TRTag.NINF
            else:
                return TRTag.PHI
        # Between enter/exit thresholds: keep previous tag to avoid chatter
        return prev_policy_tag
    else:
        # Previous REAL or unknown
        tau_q_enter = policy.tau_Q_on
        tau_q_exit = policy.tau_Q_off
        tau_p_th = policy.tau_P_on
        if q_abs < tau_q_enter:
            # In band; classify
            if p_abs >= tau_p_th:
                sign = _sign_from_value(float(p.value)) * _sign_from_value(float(q.value))
                return TRTag.PINF if sign >= 0 else TRTag.NINF
            else:
                return TRTag.PHI
        elif q_abs >= tau_q_exit:
            return TRTag.REAL
        else:
            # Between thresholds; stay REAL by default
            return TRTag.REAL
